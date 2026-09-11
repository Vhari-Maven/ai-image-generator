"""Sprite-sheet slicer + postprocess stage.

Splits a sheet image into per-cell sprite PNGs given a grid spec.
Designed for the pixel-art expression / pose sheet workflow: a single
sheet with N rows × M cols of character portraits or pose frames is
produced by a single generation call, then cut into individual files
for game-engine / asset-library use.

Two modes:

- **`bbox`** (default) — alpha-aware "detect-then-assign". The slicer
  finds connected blobs of opaque pixels in the WHOLE image first,
  picks the `rows × cols` largest, then assigns each blob to a grid
  cell based on its centroid position. This is robust to characters
  whose silhouettes spill across grid boundaries — a foot extending
  across the cell line stays attached to its character because the
  whole character is one connected component, not because we sliced
  on an exact grid line.

  The earlier "grid-first then detect inside each cell" approach broke
  on this: a foot crossing the boundary either got clipped (naive
  grid) or got dropped as a stray (cell-local connected components).

- **`grid`** — chop on regular cell boundaries with no awareness of
  alpha. Matches some game-engine sprite-sheet conventions (engines
  that look up by exact pixel offset on a uniform grid). Faster than
  bbox; clips any character whose silhouette spills past its cell.

Optional `--uniform` (bbox only) pads every sprite to the largest
bbox dimensions so the set has a consistent canvas size — important
for sprite-engine setups where every frame is positioned the same on
a unit-sized canvas.

The slicer reads alpha for the bbox pass — input must be RGBA. Inputs
without an alpha channel raise a clear error rather than silently
treating every pixel as opaque.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from prompts.parser import ArtPrompt


@dataclass
class SliceConfig:
    cols: int
    rows: int
    mode: str = "bbox"  # "bbox" or "grid"
    padding: int = 0
    uniform: bool = False

    def __post_init__(self) -> None:
        if self.cols < 1 or self.rows < 1:
            raise ValueError(
                f"grid dimensions must be >= 1 (got cols={self.cols}, rows={self.rows})"
            )
        if self.mode not in ("bbox", "grid"):
            raise ValueError(
                f"slice mode must be 'bbox' or 'grid' (got {self.mode!r})"
            )
        if self.padding < 0:
            raise ValueError(f"padding must be >= 0 (got {self.padding})")


@dataclass
class _Blob:
    """One connected opaque component, with absolute-image bbox + mask + centroid."""
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in source-image coords
    mask: np.ndarray  # (h, w) bool, sized to bbox; True for blob pixels
    centroid: Tuple[float, float]  # (cx, cy) in source-image coords
    size: int  # pixel count


def _find_blobs(arr: np.ndarray, n_keep: int) -> List[_Blob]:
    """Find the `n_keep` largest connected opaque blobs in the whole image.

    Connected-component labeling runs once on the whole image, so a
    character whose silhouette crosses any grid line is still a single
    blob. We rank by pixel count and keep the top `n_keep`, discarding
    smaller blobs (model-rendering noise, dust puffs, stray-pixel
    artifacts).
    """
    from scipy import ndimage

    mask = arr[..., 3] > 0
    if not mask.any():
        return []

    # 4-connectivity is dense enough for character silhouettes; pixel-art
    # diagonal-only connections are rare and not relevant here.
    labels, n_blobs = ndimage.label(mask)
    if n_blobs == 0:
        return []

    sizes = ndimage.sum_labels(mask, labels, index=range(1, n_blobs + 1))
    # Descending sort, take top n_keep.
    order = np.argsort(sizes)[::-1][:n_keep]

    blobs: List[_Blob] = []
    for idx in order:
        label_id = int(idx) + 1
        blob_mask_full = labels == label_id
        rows = np.any(blob_mask_full, axis=1)
        cols = np.any(blob_mask_full, axis=0)
        rmin = int(np.argmax(rows))
        rmax = len(rows) - 1 - int(np.argmax(rows[::-1]))
        cmin = int(np.argmax(cols))
        cmax = len(cols) - 1 - int(np.argmax(cols[::-1]))

        bbox = (cmin, rmin, cmax + 1, rmax + 1)
        cropped_mask = blob_mask_full[rmin:rmax + 1, cmin:cmax + 1]
        # Centroid: use the bbox center rather than pixel-mean centroid.
        # For asymmetric silhouettes (lifted leg, raised arm) the mean
        # centroid drifts toward the heavier side and can mis-assign
        # the character to the wrong grid cell. Bbox center stays
        # anchored to the character's spatial extent.
        cx = (cmin + cmax) / 2.0
        cy = (rmin + rmax) / 2.0
        blobs.append(
            _Blob(
                bbox=bbox,
                mask=cropped_mask,
                centroid=(cx, cy),
                size=int(sizes[idx]),
            )
        )
    return blobs


def _assign_to_grid(
    blobs: List[_Blob], rows: int, cols: int, w: int, h: int
) -> List[Optional[_Blob]]:
    """Assign each blob to a grid cell by centroid; return row-major list.

    A cell may end up with multiple candidate blobs — for instance, if
    the model rendered two characters close together that both centroid
    into the same cell. In that case keep the larger blob and warn.
    Cells with no blob assigned end up as None in the output.
    """
    cell_w = w / cols
    cell_h = h / rows
    grid: List[List[Optional[_Blob]]] = [[None] * cols for _ in range(rows)]

    for blob in blobs:
        cx, cy = blob.centroid
        c = min(int(cx / cell_w), cols - 1)
        r = min(int(cy / cell_h), rows - 1)
        existing = grid[r][c]
        if existing is None:
            grid[r][c] = blob
        elif blob.size > existing.size:
            print(
                f"  warning: cell ({r},{c}) had two candidate blobs "
                f"(sizes {existing.size:,} and {blob.size:,}); "
                f"keeping the larger"
            )
            grid[r][c] = blob
        else:
            print(
                f"  warning: cell ({r},{c}) had two candidate blobs "
                f"(sizes {existing.size:,} and {blob.size:,}); "
                f"keeping the larger"
            )

    return [grid[r][c] for r in range(rows) for c in range(cols)]


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    padding: int,
    bounds: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Expand a bbox by `padding` on each side, clipped to image bounds."""
    w, h = bounds
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(w, x1 + padding),
        min(h, y1 + padding),
    )


def slice_sheet(
    src: Path,
    config: SliceConfig,
    output_dir: Optional[Path] = None,
    name_template: str = "{stem}_{idx:03d}.png",
) -> List[Path]:
    """Slice `src` into `cols * rows` sprite PNGs.

    Returns the list of written paths in row-major order (top-left first).
    """
    image = Image.open(src)
    if image.mode != "RGBA":
        # Grid mode tolerates non-alpha; bbox mode does not. Be strict
        # about it so callers don't get silent "every pixel is opaque"
        # bbox results.
        if config.mode == "bbox":
            raise ValueError(
                f"{src.name}: bbox-mode slicing requires an RGBA image "
                f"(got mode={image.mode}). Use --mode grid for non-alpha "
                f"sheets."
            )
        image = image.convert("RGBA")
    arr = np.array(image, dtype=np.uint8)
    h, w = arr.shape[:2]

    cell_w = w // config.cols
    cell_h = h // config.rows
    if cell_w == 0 or cell_h == 0:
        raise ValueError(
            f"image too small to slice {config.cols}×{config.rows}: "
            f"{w}×{h} → cells would be {cell_w}×{cell_h}"
        )

    n_cells = config.rows * config.cols

    # Pass 1: build per-cell (bbox, mask) tuples, in row-major order.
    cells: List[Optional[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]] = []
    if config.mode == "grid":
        for r in range(config.rows):
            for c in range(config.cols):
                x0 = c * cell_w
                y0 = r * cell_h
                x1 = w if c == config.cols - 1 else (c + 1) * cell_w
                y1 = h if r == config.rows - 1 else (r + 1) * cell_h
                cells.append(((x0, y0, x1, y1), None))
    else:
        # Detect-then-assign: find blobs globally, then route each to a
        # cell by centroid. Robust to characters whose silhouettes cross
        # grid lines (e.g. a forward foot extending into the next cell).
        blobs = _find_blobs(arr, n_keep=n_cells)
        assigned = _assign_to_grid(blobs, config.rows, config.cols, w, h)
        for blob in assigned:
            if blob is None:
                cells.append(None)
                continue
            bbox = blob.bbox
            if config.padding:
                bbox = _expand_bbox(bbox, config.padding, (w, h))
            cells.append((bbox, blob.mask))

    # Compute uniform canvas dimensions if requested.
    uniform_size: Optional[Tuple[int, int]] = None
    if config.uniform:
        sizes = [
            (cell[0][2] - cell[0][0], cell[0][3] - cell[0][1])
            for cell in cells if cell is not None
        ]
        if sizes:
            uniform_size = (max(s[0] for s in sizes), max(s[1] for s in sizes))

    # Pass 2: write per-cell PNGs.
    out_dir = output_dir if output_dir is not None else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for idx, cell in enumerate(cells, start=1):
        out_path = out_dir / name_template.format(stem=src.stem, idx=idx)

        if cell is None:
            print(f"  [{idx:03d}] empty cell — no character assigned, skipping")
            continue

        bbox, blob_mask = cell
        x0, y0, x1, y1 = bbox
        sprite = arr[y0:y1, x0:x1].copy()

        if blob_mask is not None:
            # Zero alpha on any pixel that's not in the target blob —
            # excludes pixels of *other* blobs that happen to land
            # inside this character's bbox (e.g. a neighbor's hand
            # that crosses into this character's bbox).
            mh, mw = blob_mask.shape
            sh, sw = sprite.shape[:2]
            if (mh, mw) == (sh, sw):
                sprite[..., 3] = np.where(blob_mask, sprite[..., 3], 0)
            else:
                # Padding expanded the bbox beyond the mask; the mask
                # sits centered (left-anchored if odd diff) inside the
                # padded sprite. Padded edges stay transparent.
                offset_y = (sh - mh) // 2
                offset_x = (sw - mw) // 2
                sub = sprite[offset_y:offset_y + mh, offset_x:offset_x + mw]
                sub[..., 3] = np.where(blob_mask, sub[..., 3], 0)
                sprite[:offset_y, :, 3] = 0
                sprite[offset_y + mh:, :, 3] = 0
                sprite[:, :offset_x, 3] = 0
                sprite[:, offset_x + mw:, 3] = 0

        if uniform_size is not None:
            sprite = _center_on_canvas(sprite, uniform_size)

        Image.fromarray(sprite, mode="RGBA").save(out_path, optimize=True)
        sw, sh = sprite.shape[1], sprite.shape[0]
        print(f"  [{idx:03d}] {sw}×{sh}  →  {out_path}")
        written.append(out_path)

    return written


def _center_on_canvas(
    sprite: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """Place `sprite` centered on a transparent canvas of `size` (w, h)."""
    target_w, target_h = size
    sh, sw = sprite.shape[:2]
    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x = (target_w - sw) // 2
    y = (target_h - sh) // 2
    canvas[y:y + sh, x:x + sw] = sprite
    return canvas


# ----------------------------------------------------------------------
# Postprocess stage — auto-slice during generation
# ----------------------------------------------------------------------

_GRID_DIMS_RE = re.compile(r"^(\d+)\s*x\s*(\d+)$", re.IGNORECASE)


def parse_slice_spec(value: str) -> SliceConfig:
    """Parse a `.prompts` slice_grid value into a SliceConfig.

    Format: `<cols>x<rows>` followed by zero or more space-separated
    flags. Recognized flags:

      - `bbox` / `grid`             — slice mode (default: bbox)
      - `uniform`                   — pad all sprites to max bbox dims
      - `padding=N`                 — bbox padding in pixels

    Examples:
      "4x2"
      "4x2 uniform"
      "4x2 grid"
      "4x2 uniform padding=4"
    """
    raw = value.strip().strip('"').strip("'").strip()
    if not raw:
        raise ValueError("slice_grid: empty value")

    tokens = raw.split()
    dims_match = _GRID_DIMS_RE.match(tokens[0])
    if not dims_match:
        raise ValueError(
            f"slice_grid: first token must be <cols>x<rows> "
            f"(got {tokens[0]!r})"
        )
    cols = int(dims_match.group(1))
    rows = int(dims_match.group(2))

    mode = "bbox"
    padding = 0
    uniform = False
    for token in tokens[1:]:
        tok = token.lower()
        if tok in ("bbox", "grid"):
            mode = tok
        elif tok == "uniform":
            uniform = True
        elif tok.startswith("padding="):
            try:
                padding = int(tok.split("=", 1)[1])
            except ValueError as e:
                raise ValueError(
                    f"slice_grid: bad padding value in {token!r}"
                ) from e
        else:
            raise ValueError(
                f"slice_grid: unknown flag {token!r} "
                f"(allowed: bbox, grid, uniform, padding=N)"
            )

    return SliceConfig(
        cols=cols, rows=rows, mode=mode, padding=padding, uniform=uniform
    )


class SliceStage:
    """Auto-slice a freshly-generated sheet into per-cell sprites.

    Gated by the `.prompts` header field `slice_grid`. The value is a
    grid spec (e.g. `4x2`, `4x2 uniform`) parsed by `parse_slice_spec`.

    Source-image selection: if a sibling `<stem>-cutout.png` exists
    (produced by `remove_background` or `pixel_art_knockout`), slice
    that — sprites need transparent backgrounds, and the cutout is
    almost always what the user wants. Otherwise slice the source PNG
    directly.

    Output layout: per-cell PNGs land in `<stem>-sprites/` next to the
    source, named `<stem>_{idx:02d}.png` in row-major order. Predictable
    location keeps generated sprite directories from polluting the
    source dir.

    Registration order: this stage must come AFTER any knockout stages
    in postprocess/__init__.py, so that the cutout PNG exists by the
    time SliceStage looks for it.
    """

    name = "slice_grid"

    def applies(self, prompt: "ArtPrompt") -> bool:
        flag = getattr(prompt, "slice_grid", None)
        return flag is not None and str(flag).strip() != ""

    def apply(self, image_path: Path, prompt: "ArtPrompt", config) -> None:
        spec = str(getattr(prompt, "slice_grid"))
        slice_config = parse_slice_spec(spec)

        # Prefer the cutout if a knockout stage produced one.
        cutout_path = image_path.with_name(f"{image_path.stem}-cutout.png")
        source = cutout_path if cutout_path.exists() else image_path

        out_dir = image_path.with_name(f"{image_path.stem}-sprites")
        slice_sheet(
            source,
            slice_config,
            output_dir=out_dir,
            name_template=f"{image_path.stem}_{{idx:02d}}.png",
        )
