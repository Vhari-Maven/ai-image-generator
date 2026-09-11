"""Pixel-art-friendly chroma knockout.

Standard chroma-key bg-removal (see remove_bg.py) is tuned for soft
painted-anime edges: a tolerance band blends matte→figure, dead-pixel
cleanup eats stragglers, edge-trim crops to silhouette. Every one of
those moves is hostile to pixel art:

- A tolerance band produces gradient alpha at silhouette edges,
  destroying pixel-art's hard binary edges.
- Dead-pixel cleanup eats single-pixel features (catchlights, highlights,
  intentional 1px details) that are load-bearing in the register.
- Edge trim breaks sprite-sheet grid alignment.

This stage does the opposite: exact-color match, zero tolerance, no
cleanup, no trim. Any pixel matching the chroma color *exactly* gets
alpha 0; everything else stays opaque. Result: hard binary alpha that
preserves pixel-art crispness, and a canvas with the same dimensions
as the input (so a 4×2 sheet stays cuttable on its original grid).

The stage is gated by the `.prompts` header field `pixel_art_knockout`,
whose value is the chroma color to knock out (hex, RGB tuple, or a
named convention). Examples:

    pixel_art_knockout: "#FF00FF"
    pixel_art_knockout: magenta
    pixel_art_knockout: "255,0,255"

Writes a sibling `<stem>-cutout.png` next to the source image.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from PIL import Image

if TYPE_CHECKING:
    from prompts.parser import ArtPrompt


_NAMED_COLORS = {
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def _parse_color(value: str) -> Tuple[int, int, int]:
    """Parse a color spec into an (r, g, b) triple in 0..255.

    Accepted forms:
      - "#RRGGBB" / "RRGGBB"
      - "r,g,b" with decimal components
      - a named color from `_NAMED_COLORS`
    """
    # Strip surrounding quotes first — values from the .prompts YAML-ish
    # header may arrive as e.g. `"#FF00FF"` with quotes preserved — then
    # the lstrip("#") below has to see the # at the front.
    raw = value.strip().strip('"').strip("'").strip()
    if not raw:
        raise ValueError("pixel_art_knockout: empty color value")

    # Hex form
    hex_candidate = raw.lstrip("#")
    if len(hex_candidate) == 6 and all(
        c in "0123456789abcdefABCDEF" for c in hex_candidate
    ):
        return (
            int(hex_candidate[0:2], 16),
            int(hex_candidate[2:4], 16),
            int(hex_candidate[4:6], 16),
        )

    # Comma-separated decimal
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            r, g, b = (int(p) for p in parts)
            if all(0 <= c <= 255 for c in (r, g, b)):
                return (r, g, b)

    # Named
    named = raw.lower()
    if named in _NAMED_COLORS:
        return _NAMED_COLORS[named]

    raise ValueError(
        f"pixel_art_knockout: cannot parse color {value!r}. "
        f"Expected hex (#FF00FF), 'r,g,b' (255,0,255), or a named color "
        f"({', '.join(sorted(_NAMED_COLORS))})."
    )


def _spill_mask(rgb, chroma: Tuple[int, int, int], threshold: int):
    """Pixels with the chroma's channel-imbalance signature.

    For magenta (255, 0, 255), high-channels are R and B, low-channel is
    G; a "spill" pixel has R > G + threshold AND B > G + threshold —
    i.e. it's leaning toward magenta even if not pure magenta. The
    inverse holds for green chroma.

    Returns a 2-D bool mask aligned with `rgb`.
    """
    import numpy as np
    high_idx = [i for i, c in enumerate(chroma) if c >= 128]
    low_idx = [i for i, c in enumerate(chroma) if c < 128]
    if not high_idx or not low_idx:
        # Pure black/white chroma carries no channel-imbalance signal.
        return np.zeros(rgb.shape[:2], dtype=bool)

    rgb_i = rgb.astype(np.int16)
    mask = np.ones(rgb.shape[:2], dtype=bool)
    for hi in high_idx:
        for lo in low_idx:
            mask &= (rgb_i[..., hi] - rgb_i[..., lo]) > threshold
    return mask


def _decontaminate_edges(
    arr,
    chroma: Tuple[int, int, int],
    reach: int,
    strength: float,
) -> int:
    """Push edge-band pixel colors away from `chroma` (chroma decontamination).

    Classic VFX despill: a silhouette-edge pixel that's been chroma-tinted
    by the model has high values in the chroma's "high" channels relative
    to its "low" channels. The amount by which it's been pushed equals
    `min((high_i - low_j) for all (high_i, low_j) pairs)` clipped to 0.
    Subtracting that amount from each high channel pushes the pixel back
    toward what it would have been without spill.

    For magenta (255, 0, 255): a pixel (180, 130, 165) has spill = min(50,
    35) = 35. Decontaminate → (145, 130, 130), neutral grey. Coral pink
    (255, 120, 150): spill = min(135, 30) = 30 → (225, 120, 120),
    slightly less saturated coral. Pure white (240, 240, 240): spill = 0,
    untouched. Skin (181, 95, 75): spill = min(86, -20) clipped to 0,
    untouched.

    Operates only on opaque pixels within `reach` of a transparent pixel —
    interior pixels are not touched. Modifies RGB; alpha unchanged.

    `strength` (0.0–1.0) scales how much spill is subtracted. 1.0 is full
    decontamination (typical right answer); lower values produce a
    gentler correction if the full subtract over-desaturates.
    """
    if strength <= 0 or reach <= 0:
        return 0

    import numpy as np
    from scipy import ndimage

    high_idx = [i for i, c in enumerate(chroma) if c >= 128]
    low_idx = [i for i, c in enumerate(chroma) if c < 128]
    if not high_idx or not low_idx:
        return 0

    alpha = arr[..., 3]
    transparent = alpha == 0
    # Edge band: any opaque pixel within `reach` of a transparent pixel.
    edge_band = ndimage.binary_dilation(transparent, iterations=reach)
    candidate = edge_band & (alpha > 0)
    if not candidate.any():
        return 0

    rgb_i = arr[..., :3].astype(np.int16)
    # spill = min over all (high, low) channel-pair gaps, clipped to >= 0
    spill = np.full(arr.shape[:2], 32767, dtype=np.int16)
    for hi in high_idx:
        for lo in low_idx:
            spill = np.minimum(spill, rgb_i[..., hi] - rgb_i[..., lo])
    spill = np.maximum(spill, 0)

    affected = candidate & (spill > 0)
    if not affected.any():
        return 0

    spill_amt = (spill.astype(np.float32) * strength).astype(np.int16)
    for hi in high_idx:
        new_vals = rgb_i[..., hi] - spill_amt
        new_vals = np.where(affected, new_vals, rgb_i[..., hi])
        arr[..., hi] = np.clip(new_vals, 0, 255).astype(np.uint8)

    return int(affected.sum())


def _despill_pass(arr, chroma: Tuple[int, int, int], threshold: int) -> int:
    """One iteration of edge-aware despill.

    Finds opaque pixels adjacent (4-connected) to currently-transparent
    pixels — i.e. silhouette-edge pixels — that match the chroma's spill
    signature, and sets their alpha to 0. Interior character pixels are
    unreachable until the halo around them has been knocked, so the
    pass cannot eat through into the character body in a single
    iteration; multiple iterations naturally peel a multi-pixel halo.

    Returns the number of pixels knocked out this pass.
    """
    import numpy as np
    alpha = arr[..., 3]
    transparent = alpha == 0
    edge_adjacent = np.zeros_like(transparent)
    edge_adjacent[1:, :] |= transparent[:-1, :]
    edge_adjacent[:-1, :] |= transparent[1:, :]
    edge_adjacent[:, 1:] |= transparent[:, :-1]
    edge_adjacent[:, :-1] |= transparent[:, 1:]

    candidate = edge_adjacent & (alpha > 0)
    spill = _spill_mask(arr[..., :3], chroma, threshold)
    knock = candidate & spill
    arr[knock, 3] = 0
    return int(knock.sum())


def knockout_color(
    src: Path,
    dst: Path,
    color: Tuple[int, int, int],
    tolerance: int = 32,
    despill_iterations: int = 4,
    despill_threshold: int = 40,
    interior_pocket_threshold: int = 100,
    decontaminate_reach: int = 2,
    decontaminate_strength: float = 1.0,
) -> int:
    """Knock out a chroma-key backdrop while preserving pixel-art
    hard-alpha invariants.

    Three-pass strategy:

    1. **Tolerance match** — every pixel within `tolerance` per channel
       of `color` is knocked. Catches the bulk of the backdrop. The
       output alpha is binary (0 or 255) — tolerance gates the *match*,
       not the *alpha*. That distinction is what preserves pixel-art
       crispness: a soft tolerance band on alpha would produce gradient
       silhouette edges, which is exactly what pixel art is not.

    2. **Edge-aware despill** — gpt-image-2 paints a 1–2px chroma-tinted
       halo at silhouette edges (pixel values like `#C63EBC` next to a
       magenta backdrop). These don't match the tolerance band but are
       clearly chroma-spill by their channel-imbalance signature. We
       iteratively erode them: find opaque pixels adjacent to transparent
       ones, knock those that match the chroma's spill signature, repeat.
       Stops early when an iteration finds nothing.

    3. **Interior-pocket cleanup** — small pockets of near-pure chroma
       sometimes survive inside the character silhouette: visually
       jarring magenta pixels embedded in hair / clothing where the
       model accidentally painted backdrop color through the character.
       These are out of reach for Pass 2 (no edge-adjacent transparent
       pixel) and not close enough to pure chroma for Pass 1 (often
       like #DD40DD — outside the tolerance band but unmistakably
       chroma-leaning). We catch them with a STRICTER channel-imbalance
       threshold (default 100 vs Pass 2's 40) applied globally, no edge
       constraint. The strict threshold is what makes this safe: legit
       character coral pink and pastel pink fall below it (their R-G
       gap can be high but their B-G gap is small).

    Tolerance defaults catch gpt-image-2's color drift (~30 units off
    pure chroma) without eating legitimate character colors. Pass-2
    threshold (40) is well below typical halo values (100+) and well
    above any non-spill character pixel — pastel pink (R-G≈51, B-G≈17)
    and coral pink (R-G≈111, B-G≈30) both fail the *both must exceed
    threshold* test. Pass-3 threshold (100) is calibrated to catch
    near-pure interior pockets (B-G typically 150+) while leaving
    coral pink (B-G≈30) safely intact.
    """
    image = Image.open(src).convert("RGBA")
    import numpy as np
    arr = np.array(image, dtype=np.uint8)

    # Pass 1: tolerance match.
    target = np.array(color, dtype=np.int16)
    diff = np.abs(arr[..., :3].astype(np.int16) - target).max(axis=-1)
    match = diff <= tolerance
    arr[match, 3] = 0
    matched = int(match.sum())

    # Pass 2: iterative edge-aware despill.
    despilled = 0
    for _ in range(despill_iterations):
        n = _despill_pass(arr, color, despill_threshold)
        if n == 0:
            break
        despilled += n

    # Pass 3: interior-pocket cleanup with a strict global spill mask.
    # Only opaque pixels are considered (Passes 1 and 2 already cleared
    # everything they were going to clear). Strict threshold means only
    # near-pure chroma survives the cut, so character chroma-adjacent
    # colors (coral pink, pastel pink) are unaffected.
    interior = 0
    if interior_pocket_threshold > 0:
        opaque = arr[..., 3] > 0
        strict_spill = _spill_mask(
            arr[..., :3], color, interior_pocket_threshold
        )
        knock = opaque & strict_spill
        arr[knock, 3] = 0
        interior = int(knock.sum())

    # Pass 4: edge decontamination. The remaining silhouette-edge pixels
    # at this point have weak chroma tint that didn't meet Pass 2's
    # threshold but still creates a visible mauve / dusty-pink fringe
    # along the silhouette. This pass doesn't change alpha — it pulls
    # the chroma component out of the RGB so the edge reads as the
    # character's true color rather than a magenta-tinted version of it.
    decontaminated = _decontaminate_edges(
        arr, color, decontaminate_reach, decontaminate_strength
    )

    out = Image.fromarray(arr, mode="RGBA")
    out.save(dst, optimize=True)
    print(
        f"  ✓ pixel_art_knockout: matched={matched:,} px, "
        f"despilled={despilled:,} px, interior={interior:,} px, "
        f"decontaminated={decontaminated:,} px (binary alpha) → {dst}"
    )
    return matched + despilled + interior


class PixelArtKnockoutStage:
    """Exact-color chroma knockout for pixel-art outputs.

    Gated by the `.prompts` header field `pixel_art_knockout`, which
    holds the color spec to knock out (hex / RGB / named).
    """

    name = "pixel_art_knockout"

    def applies(self, prompt: "ArtPrompt") -> bool:
        from .pipeline import flag_enabled
        return flag_enabled(getattr(prompt, "pixel_art_knockout", None))

    def apply(self, image_path: Path, prompt: "ArtPrompt", config) -> None:
        flag = getattr(prompt, "pixel_art_knockout", None)
        color = _parse_color(str(flag))
        out = image_path.with_name(f"{image_path.stem}-cutout.png")
        knockout_color(image_path, out, color)
