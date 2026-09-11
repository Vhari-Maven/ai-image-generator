"""Background removal post-processor.

Pipeline overview
=================

For each input image, ``remove_background`` runs up to nine sequential
stages. Each stage either reshapes the alpha channel (carving the
silhouette) or cleans up RGB residue from chroma-key spill, and is
independently toggleable via ``RemoveBgConfig``. The order matters —
later stages assume earlier ones have already run, and the parameters
are interrelated (which is why config lives in YAML, not on the CLI).

  0. **Resolve chroma key.** Peek at the four corner patches and snap
     the median to a canonical key (green / magenta / cyan / blue / red
     / yellow) if it's close enough. Or honor a fixed hex from config.
     Result feeds stages 2, 4, and 5; if nothing resolves, those stages
     no-op and we degrade gracefully to segmentation + matting alone.

  1. **Segmentation + alpha matting (rembg).** Neural background
     segmentation produces the initial silhouette. With ``alpha_matting``
     enabled, rembg also runs pymatting's closed-form trimap matting,
     giving continuous α at hair / fur edges instead of the binarized
     α of raw segmentation.

  2. **Chroma matte (YCbCr).** Per-pixel α derived from the pixel's
     Cb/Cr distance to the key color, combined with the segmentation α
     via ``min`` so *both* the segmentation silhouette AND the chroma
     physics must agree a pixel is foreground. Catches interior pockets
     segmentation missed and preserves chroma-distant accents (cyan
     stripe vs. green key — far apart on Cb/Cr, close in raw RGB).

  3. **Foreground decontamination (pymatting).** Solves the matting
     equation ``I = α·F + (1−α)·B`` for the unmixed foreground F,
     removing key-color spill from partial-α edge pixels independently
     of which key was used. Note: a no-op on α=255 (the equation
     degenerates to F = I), so does NOT fix interior spill the
     diffusion model painted directly into opaque shadow regions.

  4. **Interior despill.** Threshold-gated, backdrop-hue-aware channel
     clamp on effectively-opaque pixels (α ≥ ``despill_min_alpha``,
     default 250) — catches the tinted shadows decontamination
     can't see. Redistributes the clipped key-channel excess into the
     off-channels so opaque skin desaturates toward neutral instead of
     picking up an opposite-hue halo (the failure mode that retired the
     older blanket-clamp pass). Threshold protects legitimate
     chroma-near-key foreground (teal gem against green key, etc.).

  4.5. **Boundary cleanup.** Proximity-gated two-pass cleanup of opaque
     contamination near the silhouette edge — the gap between despill's
     "small per-pixel excess" and chroma_knockout's "α<255 only" rules.
     Pass A drops saturated key-hue pixels (olive blobs the matte's
     YCbCr radius missed) to α=0. Pass B recolors weakly-tinted opaque
     pixels by sampling the nearest *non-tinted* opaque neighbor's hue
     at the suspicious pixel's luminance — so green-cast hair shadows
     adopt local platinum/cyan tones instead of desaturating toward
     neutral grey (despill's fallback). Both passes only fire on pixels
     within ``boundary_cleanup_radius_px`` of a transparent region, so
     they cannot bite saturated foreground far from any hole.

  5. **Chroma-key knockout.** RGB-distance fallback for residual key
     pockets the chroma matte attenuated but didn't fully clear. Fires
     only on ``[chroma_opaque_threshold, chroma_max_alpha]`` — stays
     out of α=255 territory entirely (matte-confident foreground) and
     out of edge-AA territory (already cleaned by decontamination).

  6. **Dead-pixel fill.** Restores α on isolated transparent specks
     surrounded by opaque neighbors (a 6-of-8 rule by default).

  7. **Alpha-bbox trim.** Crops to silhouette extent, ignoring
     sub-threshold edge residue. Runs before the pocket diagnostic so
     the pocket coordinates reported below refer to the saved file.

  8. **Internal-pocket diagnostic.** Connected-component pass over the
     transparent mask. Components that don't reach the canvas border
     are "pockets" — almost always matte damage where chroma got too
     close to the key. Always reports counts; optionally auto-fills
     pockets below ``pocket_fill_max_size`` and leaves larger ones for
     human review.

Every stage records a marker string into PNG metadata, so finished
images carry a full audit trail (pixels attenuated, edges
decontaminated, pockets filled, etc.) without needing logs.

Configuration
=============

All knobs live in YAML at ``postprocess.remove_background.*``. The
``.prompts`` header field is a switch only (``true`` / ``false`` /
``<model-name>``) and cannot tune parameters — by design, so the
chroma-key and dead-pixel parameters don't drift between ad-hoc CLI
invocations and pipeline runs.

Dependency group
================

``rembg``, ``pymatting``, ``scipy`` and friends ship under the
``bg-removal`` dependency group. If the group isn't installed,
``remove_background`` raises ``BgRemovalNotInstalled`` with an install
hint — the generator pipeline catches it, warns, and continues without
aborting the batch. Modules elsewhere can ``import postprocess.remove_bg``
unconditionally; the heavy deps only load when ``remove_background``
actually runs.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


class BgRemovalNotInstalled(RuntimeError):
    """Raised when bg-removal is requested but its deps aren't installed."""


class SamNotInstalled(BgRemovalNotInstalled):
    """Raised when a ``sam3:<prompt>`` model is requested but the SAM venv
    isn't set up.

    Subclass of ``BgRemovalNotInstalled`` so pipeline handling is shared,
    but the standalone CLI treats it as a per-model failure — a missing
    SAM venv shouldn't abort the rembg half of an A/B run.
    """


# Stage 6 counts a neighbor as "opaque" at α≥250 so faint sub-255 edge
# pixels still count toward the 6-of-8 rule. Tightening to 255 misses
# real dead pixels next to anti-aliased edges. (Despill's own opaque
# threshold is the ``despill_min_alpha`` config knob — see its docs.)
_DEAD_PIXEL_OPAQUE = 250


@dataclass
class RemoveBgConfig:
    """Resolved config for the bg-removal post-process.

    Loaded from YAML at ``postprocess.remove_background.*``. The model
    can be overridden per-prompt by setting ``remove_background:
    <model-name>`` in the .prompts header (any other tunables stay at
    config defaults).
    """
    # --- segmentation model ---
    model: str = "isnet-anime"

    # --- stage 1: alpha matting (rembg trimap / pymatting closed-form) ---
    # Adds a few seconds per image but produces continuous α at
    # hair / fur edges instead of binarized α. Strongly recommended when
    # paired with ``decontaminate_foreground`` — together they fix the
    # "key-tinted halo on hair" symptom that channel clamping cannot.
    alpha_matting: bool = True
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 10

    # --- chroma key resolution (feeds stages 2, 4, 5) ---
    # "auto" → detect from corners per-call (handles mixed-key batches).
    # Hex string (e.g. "#FF00FF") → fixed key for all calls.
    # None / false → disable chroma-aware stages entirely.
    chroma_key: Optional[str] = "auto"

    # --- stage 2: chroma matte (YCbCr) ---
    # Per-pixel α from Cb/Cr distance to the key, combined with the
    # segmentation α via ``combine`` (min | multiply) so both must agree.
    use_chroma_matte: bool = True
    chroma_matte_inner_radius: float = 25.0   # Cb/Cr distance ≤ this → α=0
    chroma_matte_outer_radius: float = 60.0   # Cb/Cr distance ≥ this → α=1
    chroma_matte_combine: str = "min"         # min | multiply

    # --- stage 3: foreground decontamination ---
    # Solves I = α·F + (1−α)·B for the unmixed F via
    # ``pymatting.estimate_foreground_ml``. No-op on α=255.
    decontaminate_foreground: bool = True

    # --- stage 4: interior despill ---
    # Backdrop-hue-aware channel clamp on α=255 pixels. Threshold
    # protects legitimately key-hue-adjacent foreground.
    #
    # ``despill_redistribute`` controls what happens to the clipped
    # excess. With ``0.0`` (the default), spill is simply removed and
    # the corrected pixel desaturates toward neutral grey. Any nonzero
    # value pushes the corrected pixel toward the *complementary* hue
    # of the key — which produces a complementary-cast halo on the
    # silhouette edge, since boundary cleanup (stage 4.5) only catches
    # *key-hue* contamination and is blind to redistributed-hue halos.
    # The complementary-cast halo blends with the background only when
    # the foreground happens to share that hue family (e.g. magenta
    # halo from green key on a pink subject); on a neutral or off-hue
    # background it reads as a wrong-color rim. Keep at 0 unless you
    # have a specific reason to bias toward the complement.
    despill_interior: bool = True
    despill_threshold: int = 8
    despill_redistribute: float = 0.0
    # Minimum α for a pixel to be eligible for despill. Default 250
    # (slightly under fully opaque) so the alpha-matting smoothing band
    # at α=254 is reachable — that band is where strong green-cast hair
    # pixels hide, and at α=254 decontamination's matting equation sees
    # only ~0.4% background contribution and barely adjusts. Tightening
    # to 255 reverts to "α=255 only" behavior. Going much lower than
    # ~240 risks reintroducing the tinted halo on real AA edges that
    # retired the old blanket-clamp pass — those edges are decontamination's
    # job and despill's redistribute logic isn't designed for them.
    despill_min_alpha: int = 250

    # --- stage 6.5: post-knockout boundary cleanup ---
    # Boundary cleanup also runs *after* chroma knockout. The pre-knockout
    # pass at stage 4.5 only sees holes the chroma matte created; the
    # knockout creates new holes, and an opaque key-cast pixel sitting
    # inside one of those new holes (or stranded as a 1-px island) is
    # invisible to the pre-knockout pass — its `near_boundary` test
    # returns False because the surrounding holes don't exist yet. The
    # post-knockout pass closes that gap by re-running the same logic
    # against the alpha state with knockout holes present.
    boundary_cleanup_post: bool = True

    # --- stage 4.5: boundary cleanup ---
    # Proximity-gated cleanup of opaque key-hue contamination near the
    # silhouette edge. Pass A drops saturated key-hue blobs to α=0; pass
    # B recolors weakly-tinted opaque pixels from local non-tinted
    # neighbors (nearest-clean-pixel hue at suspicious-pixel luminance).
    # The proximity gate (dilation of α<threshold by `radius_px`) keeps
    # both passes off saturated foreground far from any hole.
    boundary_cleanup: bool = True
    boundary_cleanup_radius_px: int = 8
    # Pass A drop threshold. The pixel is dropped to α=0 only if its
    # key_excess (key-channel dominance over off-channels) exceeds this.
    # Default 25 — strong enough to identify clear key-spill residue
    # (Δg=+30+ lime blobs) without catching mildly green-tinted hair
    # (Δg=+5 to +25), which falls through to Pass B's nearest-clean-
    # neighbor recolor instead. Lower values were correct for catching
    # bright residue but also dropped soft-edge "thin hair over key"
    # pixels that should be kept and desaturated.
    boundary_cleanup_drop_hue_margin: int = 25
    boundary_cleanup_drop_saturation: int = 20
    boundary_cleanup_recolor_hue_margin: int = 2
    # Minimum α for a pixel to be eligible for either pass. Defaults to
    # 200, deliberately *below* despill's ``despill_min_alpha`` — channel-clamping
    # partial-α gives haloing (decontamination's domain), but boundary
    # cleanup *replaces* RGB / sets α=0, so partial-α is safe and the
    # alpha-matting smoothed band (α≈210–254) is exactly where
    # scattered key-hue specks hide from despill.
    boundary_cleanup_min_alpha: int = 200
    # Pass C — opaque-island removal. Catches small key-tinted connected
    # components disconnected from the silhouette body — the matte's
    # "almost dead" residue (α≈5–30 pure-key blobs that still composite
    # visibly). Symmetric to the existing pocket detector but on opaque
    # islands floating in transparent sea, not transparent islands in
    # opaque foreground. Min-alpha is intentionally low to capture
    # alpha-attenuated residue; size cap protects legitimate small
    # foreground islands (earrings, stray strands).
    boundary_cleanup_island_min_alpha: int = 5
    boundary_cleanup_island_max_size: int = 500
    boundary_cleanup_island_hue_margin: int = 4

    # --- stage 5: chroma-key knockout (RGB-distance fallback) ---
    # The matte does primary attenuation upstream; the knockout's job
    # is narrowed: catch residue in the matte's smoothstep transition
    # band, and stay out of α=255 territory (matte-confident foreground).
    chroma_tolerance: float = 150.0
    chroma_dominance: int = 35
    chroma_off_channel_ceiling: int = 170
    chroma_off_channel_asymmetry: int = 70
    chroma_opaque_threshold: int = 100         # min α to fire on
    chroma_max_alpha: int = 254                # max α to fire on (skip α=255)

    # --- stage 6: dead-pixel fill ---
    # ``None`` disables; otherwise the minimum opaque-neighbor count for
    # an α=0 pixel to be restored to α=255.
    dead_pixel_threshold: Optional[int] = 6
    # Key-awareness: skip restoring α=0 pixels whose own RGB is strongly
    # key-tinted. Without this, dead-pixel fill silently undoes the
    # chroma_knockout's correct decision and resurrects bright lime/key
    # pixels into the middle of foreground (visible as green dots in
    # hair shadow regions). The threshold is the per-pixel key_excess
    # above which restoration is skipped — same units as despill's
    # threshold (G - max(R,B) for green keys, etc.). Set to None to
    # restore the legacy behavior of "fill anything surrounded by α≥250".
    dead_pixel_key_excess_skip: Optional[int] = 8

    # --- stage 7: alpha-bbox trim ---
    trim: bool = True
    trim_threshold: int = 10

    # --- stage 8: internal-pocket diagnostic ---
    # Connected-component pass over the transparent mask. Pockets that
    # don't touch the canvas border are matte damage — always reported,
    # optionally auto-filled below ``pocket_fill_max_size``. Runs after
    # trim so reported pocket coordinates refer to the saved file.
    pocket_detect: bool = True
    pocket_alpha_threshold: int = 10           # α < this counts as "transparent"
    pocket_fill_max_size: Optional[int] = None  # None → warn only; int → auto-fill cap

    # --- output handling ---
    # Move an existing destination file to assets/drafts/ before writing.
    backup: bool = True

    # --- debug ---
    # If set, write {input_stem}-{NN}-{stage_key}.png into this directory
    # after every stage that runs (skipped for disabled / no-op stages).
    # Lets you scrub through intermediates in an image viewer when tuning
    # — the dominant iteration pain when chroma matte or despill misbehave.
    # Not a YAML knob; set via CLI (--dump-stages <dir>) on remove_bg_cli.
    dump_stages_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> "RemoveBgConfig":
        """Build from a (possibly partial) config-dict.

        Unknown keys are ignored but warned about — the tool's whole
        tuning philosophy is "edit the YAML and re-run", so a silently
        dropped typo (``despill_treshold``) reads as "the knob doesn't
        work".
        """
        if not raw:
            return cls()
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            print(
                "  ⚠ unknown postprocess.remove_background config key(s) "
                f"ignored: {', '.join(unknown)}",
                file=sys.stderr,
            )
        kwargs: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f in raw:
                kwargs[f] = raw[f]
        # Normalize: chroma_key=false / null in YAML disables the pass;
        # "auto" / "AUTO" / "Auto" all collapse to lowercase "auto" sentinel.
        ck = kwargs.get("chroma_key")
        if ck in (False, "false", "False", "none", "None", ""):
            kwargs["chroma_key"] = None
        elif isinstance(ck, str) and ck.lower() == "auto":
            kwargs["chroma_key"] = "auto"
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Chroma-key detection
# ---------------------------------------------------------------------------

def parse_hex_color(s: str) -> tuple[int, int, int]:
    """Parse a hex color string like '#00FF00' or '00FF00' into (R, G, B)."""
    s = s.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected 6-digit hex color, got {s!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


# Canonical chroma-key colors. Auto-detect snaps to the nearest of these
# when within ``_CANONICAL_SNAP_DISTANCE`` so the post-hoc knockout's tuned
# tolerances (which assume a canonical key) keep working. Without snap,
# slight model-rendered drift like (250, 4, 233) instead of pure (255, 0,
# 255) shifts the knockout's RGB-distance band just enough that saturated
# pinks fall inside it and get eaten as residue.
_CANONICAL_KEYS = [
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (0, 0, 255),     # blue
    (0, 255, 255),   # cyan
    (255, 0, 0),     # red
    (255, 255, 0),   # yellow
]
_CANONICAL_SNAP_DISTANCE = 60.0   # RGB-space; covers typical generation drift


def _snap_to_canonical(detected_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Snap detected RGB to the nearest canonical chroma key if within range."""
    best_dist = float("inf")
    best_key = detected_rgb
    for key in _CANONICAL_KEYS:
        dist = sum((a - b) ** 2 for a, b in zip(detected_rgb, key)) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = key
    if best_dist <= _CANONICAL_SNAP_DISTANCE:
        return best_key
    return detected_rgb


def detect_chroma_key(arr_module, rgb_image, patch_size: int = 64,
                      max_corner_disagreement: float = 15.0,
                      min_saturation: int = 100):
    """Detect a flat-saturated chroma-key backdrop from a source image's corners.

    Sample the four corner patches, take the median of each, and return
    the overall median if the corners both AGREE (homogeneous backdrop)
    and are SATURATED (chroma plate, not a white/grey/photographic
    background).

    Returns:
        (R, G, B) int tuple if a chroma key was detected, else None —
        caller should fall back to disabling chroma-aware stages on None.

    Args:
        arr_module: numpy module (passed in to keep imports cheap at
            module scope; this file is imported eagerly elsewhere).
        rgb_image: HxWx3 uint8 array.
        patch_size: size in pixels of each corner sample patch. 64 is
            plenty for a centered character render — figures rarely
            reach into corners.
        max_corner_disagreement: maximum per-channel stddev across the
            four corner medians for the corners to count as
            "homogeneous". A flat chroma plate should have stddev near
            0; raise if generators produce slightly noisy plates.
        min_saturation: minimum (max_channel - min_channel) of the
            corner average, to reject pale or grey backgrounds. 100
            (~40% of the 8-bit range) cleanly accepts pure
            green/magenta/cyan/blue/red keys and rejects
            white/grey/skin-toned backgrounds.
    """
    np = arr_module
    h, w = rgb_image.shape[:2]
    if h < patch_size * 2 or w < patch_size * 2:
        return None
    patches = [
        rgb_image[:patch_size, :patch_size],
        rgb_image[:patch_size, -patch_size:],
        rgb_image[-patch_size:, :patch_size],
        rgb_image[-patch_size:, -patch_size:],
    ]
    medians = np.array([np.median(p.reshape(-1, 3), axis=0) for p in patches])
    if float(np.std(medians, axis=0).max()) > max_corner_disagreement:
        return None
    key = np.median(medians, axis=0).astype(int)
    if int(key.max()) - int(key.min()) < min_saturation:
        return None
    detected = (int(key[0]), int(key[1]), int(key[2]))
    return _snap_to_canonical(detected)


# ---------------------------------------------------------------------------
# Output-path utilities
# ---------------------------------------------------------------------------

def backup_existing(output_path: Path) -> Optional[Path]:
    """Copy an existing output to a sibling `drafts/` dir (see backup.py)."""
    from backup import backup_existing as _backup
    return _backup(output_path)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
#
# Each ``_stage_*`` returns ``(image, marker, log)``:
#   - image:  the (possibly modified) PIL.Image to pass to the next stage
#   - marker: "k1=v1,k2=v2,..." string for PNG metadata, or None if the
#             stage was disabled / no-op
#   - log:    one-line human-readable summary for ``--verbose``, or None
#
# Stages short-circuit on disabled config or missing inputs (e.g. no
# resolved chroma key). The orchestrator just chains them.


def _marker(fields: dict) -> str:
    """Format a stage marker as "k1=v1,k2=v2,..." for PNG metadata.

    Values are stringified with ``str()``. A few markers (key_rgb, bbox)
    legitimately contain commas inside their values; that's pre-existing
    ambiguity — readers know to parse those by stage_key, not by
    splitting on commas blindly.
    """
    return ",".join(f"{k}={v}" for k, v in fields.items())


def _stage_resolve_chroma_key(np, src_rgb, config, verbose):
    """Stage 0: resolve config.chroma_key into a concrete (R,G,B) tuple.

    ``src_rgb`` is the source image as an HxWx3 uint8 array (shared with
    the chroma matte). Returns ``(key_rgb_tuple_or_None, marker_or_None)``
    — note this stage returns 2 values, not the standard 3, because
    there's no image to pass through. The other stages all take this as
    input.
    """
    if config.chroma_key == "auto":
        detected = detect_chroma_key(np, src_rgb)
        if detected is not None:
            key_hex = f"#{detected[0]:02X}{detected[1]:02X}{detected[2]:02X}"
            if verbose:
                print(f"  chroma key auto-detected: {key_hex}", file=sys.stderr)
            return detected, _marker({"mode": "auto", "detected": key_hex})
        if verbose:
            print(
                "  chroma key auto-detection failed (corners not homogeneous "
                "or saturated); chroma matte + knockout disabled",
                file=sys.stderr,
            )
        return None, _marker({"mode": "auto", "detected": "none"})
    if config.chroma_key:
        key_rgb = parse_hex_color(config.chroma_key)
        key_hex = f"#{key_rgb[0]:02X}{key_rgb[1]:02X}{key_rgb[2]:02X}"
        return key_rgb, _marker({"mode": "fixed", "key": key_hex})
    return None, None


def _stage_alpha_matting(src, session, config):
    """Stage 1: rembg segmentation, optionally with trimap alpha matting."""
    from rembg import remove
    if not config.alpha_matting:
        return remove(src, session=session), None, None
    cut = remove(
        src, session=session, alpha_matting=True,
        alpha_matting_foreground_threshold=config.alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=config.alpha_matting_background_threshold,
        alpha_matting_erode_size=config.alpha_matting_erode_size,
    )
    marker = _marker({
        "foreground_threshold": config.alpha_matting_foreground_threshold,
        "background_threshold": config.alpha_matting_background_threshold,
        "erode_size": config.alpha_matting_erode_size,
    })
    log = (
        f"alpha matting: trimap fg≥{config.alpha_matting_foreground_threshold} "
        f"bg≤{config.alpha_matting_background_threshold} "
        f"erode={config.alpha_matting_erode_size}"
    )
    return cut, marker, log


# ---------------------------------------------------------------------------
# Stage 1 alternative: SAM 3 segmentation (subprocess shim)
# ---------------------------------------------------------------------------
#
# Selected by setting `model` to "sam3:<prompt>" — e.g. "sam3:woman" or
# "sam3:hair". Shells out to `tools/sam/.venv/bin/sam-mask`, which lives in
# its own uv project (separate dep graph + ROCm wheels — see
# tools/sam/README.md). Stages 2–8 of the pipeline run on the resulting
# alpha exactly as they would on a rembg cutout, so this is a like-for-like
# A/B against the rembg models.

SAM_PREFIX = "sam3:"
# Path is relative to project root (CWD by convention — matches the rest
# of the art-generator's path handling).
SAM_BIN = Path("tools/sam/.venv/bin/sam-mask")


def _is_sam_model(model: str) -> bool:
    return model.startswith(SAM_PREFIX)


def _sam_prompt(model: str) -> str:
    return model[len(SAM_PREFIX):].strip() or "woman"


def _stage_sam_mask(input_path, config, sam_prompt):
    """Stage 1 alternative: SAM 3 segmentation via subprocess.

    Skips alpha-matting params (SAM doesn't expose them). Returns the SAM
    cutout — already RGBA with alpha applied — for downstream stages to
    refine. Errors if the SAM venv binary is missing, with a hint at the
    SAM README's setup notes.
    """
    import subprocess
    import tempfile
    from PIL import Image

    if not SAM_BIN.exists():
        raise SamNotInstalled(
            f"sam-mask binary not found at {SAM_BIN} — the tools/sam venv "
            "isn't set up. See tools/sam/README.md → 'Quick start' "
            "(`/home/node/.local/bin/uv-variant/uv sync` from tools/sam)."
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            str(SAM_BIN),
            str(input_path),
            "--prompt", sam_prompt,
            "--out", str(tmp_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"sam-mask failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        cut = Image.open(tmp_path).convert("RGBA")
        cut.load()  # detach from the file before tmp is unlinked
    finally:
        tmp_path.unlink(missing_ok=True)

    marker = _marker({"backend": "sam3", "prompt": sam_prompt})
    log = f"SAM 3 segmentation: prompt={sam_prompt!r}"
    return cut, marker, log


def _stage_chroma_matte(np, src_rgb, cut, key_rgb, config):
    """Stage 2: per-pixel YCbCr-distance α, combined with segmentation α.

    Catches interior pockets segmentation missed (chroma physics says
    α=0 there) and preserves chroma-distant accents the binary knockout
    used to bite (e.g. cyan stripe vs. green key — far on Cb/Cr, close
    in raw RGB).
    """
    if not (config.use_chroma_matte and key_rgb is not None):
        return cut, None, None
    from PIL import Image
    from postprocess.chroma_matte import chroma_matte, combine
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    cut_arr = np.array(cut)
    alpha_key = chroma_matte(
        np, src_rgb, key_rgb,
        config.chroma_matte_inner_radius,
        config.chroma_matte_outer_radius,
    )
    seg_alpha = cut_arr[..., 3].copy()
    new_alpha = combine(np, seg_alpha, alpha_key, mode=config.chroma_matte_combine)
    attenuated = int(((seg_alpha.astype(int) - new_alpha.astype(int)) > 0).sum())
    cut_arr[..., 3] = new_alpha
    cut = Image.fromarray(cut_arr, mode="RGBA")
    marker = _marker({
        "inner": config.chroma_matte_inner_radius,
        "outer": config.chroma_matte_outer_radius,
        "combine": config.chroma_matte_combine,
        "pixels_attenuated": attenuated,
    })
    log = (
        f"chroma matte (YCbCr): inner={config.chroma_matte_inner_radius} "
        f"outer={config.chroma_matte_outer_radius} "
        f"combine={config.chroma_matte_combine} — "
        f"attenuated {attenuated} pixels"
    )
    return cut, marker, log


def _stage_decontaminate(np, src, cut, config):
    """Stage 3: solve I = α·F + (1−α)·B for unmixed F via pymatting.

    Replaces the cut image's RGB with the estimated foreground while
    keeping its α intact. Source RGB comes from the *original* image
    (before rembg), since rembg's RGB output is unspecified at α=0
    pixels and we want the true mixed observation everywhere.

    No-op on α=255 (matting equation degenerates to F = I), which is
    why ``_stage_despill`` exists to clean what this can't.
    """
    if not config.decontaminate_foreground:
        return cut, None, None
    from PIL import Image
    from pymatting import estimate_foreground_ml

    src_rgb_img = src if src.mode == "RGB" else src.convert("RGB")
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")

    src_arr = np.asarray(src_rgb_img, dtype=np.float64) / 255.0
    cut_arr = np.asarray(cut)
    alpha = cut_arr[..., 3].astype(np.float64) / 255.0

    foreground = estimate_foreground_ml(src_arr, alpha)
    foreground_u8 = np.clip(foreground * 255.0, 0, 255).astype(np.uint8)

    out = cut_arr.copy()
    out[..., :3] = foreground_u8

    edge_count = int(((cut_arr[..., 3] > 0) & (cut_arr[..., 3] < 255)).sum())
    cut = Image.fromarray(out, mode="RGBA")
    marker = _marker({"method": "estimate_foreground_ml", "edge_pixels": edge_count})
    log = f"foreground decontamination: solved unmixed F for {edge_count} partial-α pixels"
    return cut, marker, log


def _stage_despill(np, cut, key_rgb, config):
    """Stage 4: backdrop-hue-aware channel-clamp despill on effectively-opaque pixels.

    For each pixel at α ≥ ``despill_min_alpha`` (default 250, so the
    alpha-matting smoothing band at α=254 is reachable — see the config
    field's docs), identifies the "key channel(s)" of ``key_rgb``
    (e.g. G for green; R+B for magenta) and clips per-pixel excess of
    those channel(s) over the off-channel(s) when it exceeds
    ``despill_threshold``. The clipped amount is redistributed:
    ``despill_redistribute`` fraction is added to each off-channel so
    the result desaturates toward neutral rather than pushing into the
    opposite hue.

    Stays out of partial-α pixels because ``_stage_decontaminate``
    already handles those via the matting equation; running channel-
    clamp on edges is what gave the old blanket pass its tinted halo.

    Threshold gating preserves legitimate chroma-near-key foreground
    (teal gem against green key, magenta heart-print against magenta
    key) — small per-pixel excursions are noise, not spill.
    """
    if not (config.despill_interior and key_rgb is not None):
        return cut, None, None
    from PIL import Image
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    arr = np.array(cut)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]

    # Identify key vs off channels via the same midpoint rule used by
    # the interior knockout — robust to model-rendered key drift.
    key_min = min(key_rgb)
    key_max = max(key_rgb)
    mid = (key_max + key_min) / 2.0
    key_channels = [i for i in (0, 1, 2) if key_rgb[i] > mid]
    off_channels = [i for i in (0, 1, 2) if i not in key_channels]
    if not off_channels or not key_channels:
        return cut, None, None  # white/grey/black "key" — nothing to clamp against

    threshold = config.despill_threshold
    redistribute = config.despill_redistribute
    opaque_mask = (alpha >= config.despill_min_alpha)

    if len(key_channels) == 1:
        # Single-channel key (green, red, blue).
        kc = key_channels[0]
        oa, ob = off_channels
        ref = np.maximum(rgb[..., oa], rgb[..., ob])
        excess = rgb[..., kc] - ref
        spill = np.where((excess > threshold) & opaque_mask, excess, 0.0)
        rgb[..., kc] -= spill
        rgb[..., oa] += redistribute * spill
        rgb[..., ob] += redistribute * spill
    else:
        # Multi-channel key (magenta=R+B, yellow=R+G, cyan=G+B).
        oc = off_channels[0]
        ka, kb = key_channels
        joint = np.minimum(rgb[..., ka] - rgb[..., oc], rgb[..., kb] - rgb[..., oc])
        spill = np.where((joint > threshold) & opaque_mask, joint, 0.0)
        rgb[..., ka] -= spill
        rgb[..., kb] -= spill
        rgb[..., oc] += redistribute * spill * 2.0  # both key channels redistributing into one off

    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    changed = int(((arr[..., :3] != rgb_u8).any(axis=-1) & opaque_mask).sum())
    arr[..., :3] = rgb_u8
    cut = Image.fromarray(arr, mode="RGBA")
    marker = _marker({
        "key_rgb": f"{key_rgb[0]},{key_rgb[1]},{key_rgb[2]}",
        "threshold": threshold,
        "redistribute": redistribute,
        "min_alpha": config.despill_min_alpha,
        "pixels_corrected": changed,
    })
    log = (
        f"interior despill: {changed} opaque pixels desaturated "
        f"(threshold={threshold}, redistribute={redistribute}, "
        f"min_alpha={config.despill_min_alpha})"
    )
    return cut, marker, log


def _stage_boundary_cleanup(np, cut, key_rgb, config):
    """Stage 4.5: proximity-gated cleanup of opaque key-hue contamination.

    Two passes, both gated by *boundary proximity* — only opaque pixels
    within ``boundary_cleanup_radius_px`` of an α<``pocket_alpha_threshold``
    pixel are eligible. The proximity gate is what makes the stage safe;
    saturated-pink heart prints far from any hole are never touched.

    Pass A — *drop*. Opaque pixels whose key channel(s) dominate the
    off-channel(s) by more than ``drop_hue_margin`` AND whose saturation
    (max−min channel range) is at least ``drop_saturation`` are knocked
    to α=0. Catches olive / desaturated-key blobs the YCbCr matte's
    distance threshold missed but which are structurally background
    (hole-adjacent and key-hue-saturated).

    Pass B — *recolor*. Remaining opaque pixels with a weaker key tint
    (``recolor_hue_margin``) are recolored from their nearest *clean*
    neighbor — clean = opaque, near-boundary, and not key-tinted. The
    suspicious pixel's BT.601 luminance is preserved; only its hue
    swaps to the neighbor's. So a green-cast hair shadow next to a
    cyan strand adopts cyan-at-shadow-luma; next to platinum it adopts
    platinum-at-shadow-luma — instead of despill's global push toward
    desaturated grey.

    Pass C — *island drop*. Small connected components of α≥``island_min_alpha``
    pixels that aren't the silhouette body get knocked to α=0 if their
    mean key-hue dominance exceeds ``island_hue_margin``. Catches the
    matte's "almost dead but not quite" residue — pure-key blobs the
    matte attenuated to α≈5–30 but didn't fully clear. These still
    composite visibly against the page background, so leaving them in
    looks like haunted-pixel green ghosts. Symmetric to the existing
    pocket detector (transparent islands in opaque foreground), but on
    opaque islands floating in transparent sea.
    """
    if not (config.boundary_cleanup and key_rgb is not None):
        return cut, None, None
    from PIL import Image
    from scipy import ndimage

    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    arr = np.array(cut)
    rgb = arr[..., :3].astype(np.int32)
    alpha = arr[..., 3]

    # Identify key vs off channels (same midpoint rule as despill / knockout).
    key_min = min(key_rgb)
    key_max = max(key_rgb)
    mid = (key_max + key_min) / 2.0
    key_channels = [i for i in (0, 1, 2) if key_rgb[i] > mid]
    off_channels = [i for i in (0, 1, 2) if i not in key_channels]
    if not off_channels or not key_channels:
        return cut, None, None  # white/grey/black "key" — nothing to cleanup against

    # Per-pixel key-hue dominance score: how much the key channel(s)
    # exceed the off-channel(s). Negative = no key tint.
    if len(key_channels) == 1:
        kc = key_channels[0]
        oa, ob = off_channels
        ref = np.maximum(rgb[..., oa], rgb[..., ob])
        key_excess = rgb[..., kc] - ref
    else:
        ka, kb = key_channels
        oc = off_channels[0]
        key_excess = np.minimum(
            rgb[..., ka] - rgb[..., oc],
            rgb[..., kb] - rgb[..., oc],
        )

    # Boundary proximity: opaque pixels within radius_px of a
    # near-transparent pixel. Reuse pocket_alpha_threshold so this
    # tracks the same "transparent enough to count" definition pockets
    # uses; if the user retunes one, both move together.
    transparent = alpha < config.pocket_alpha_threshold
    near_boundary = ndimage.binary_dilation(
        transparent, iterations=config.boundary_cleanup_radius_px
    )

    # Pass A: drop saturated key-hue pixels. Note this stage's "opaque"
    # is looser than despill's `despill_min_alpha` — see config docs.
    opaque = alpha >= config.boundary_cleanup_min_alpha
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    drop_mask = (
        near_boundary
        & opaque
        & (key_excess > config.boundary_cleanup_drop_hue_margin)
        & (saturation >= config.boundary_cleanup_drop_saturation)
    )
    dropped = int(drop_mask.sum())
    if dropped:
        arr[..., 3][drop_mask] = 0
        alpha = arr[..., 3]
        opaque = alpha >= config.boundary_cleanup_min_alpha

    # Pass B: recolor remaining weakly-tinted opaque pixels from
    # nearest clean neighbor.
    suspicious = (
        near_boundary
        & opaque
        & (key_excess > config.boundary_cleanup_recolor_hue_margin)
    )
    # Clean reference pool: opaque pixels that are NOT key-hue-tinted at
    # all (key_excess <= 0). Restricting to opaque-only avoids sampling
    # transition-band pixels that decontamination is mid-resolving.
    clean = opaque & (key_excess <= 0)

    recolored = 0
    if suspicious.any() and clean.any():
        # Nearest clean-pixel index for every pixel. For clean pixels
        # the answer is themselves; for suspicious it's the closest
        # clean opaque pixel (typically a hair/skin neighbor a few px
        # away). distance_transform_edt with return_indices is the
        # standard vectorized way to do this — much faster than
        # per-pixel windowed sampling.
        _, (ni, nj) = ndimage.distance_transform_edt(~clean, return_indices=True)
        neighbor_rgb = rgb[ni, nj].astype(np.float32)

        # BT.601 luma — match the YCbCr space the chroma matte uses
        # so "preserve luminance" is consistent across stages.
        wts = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        Y_orig = (rgb.astype(np.float32) * wts).sum(axis=-1)
        Y_neigh = (neighbor_rgb * wts).sum(axis=-1)
        # Scale neighbor RGB so its luma matches the suspicious pixel's
        # original luma. ratio < 1 darkens (shadow); > 1 brightens.
        # Floor Y_neigh at 1 to avoid div-by-zero on pure-black neighbors.
        ratio = Y_orig / np.maximum(Y_neigh, 1.0)
        new_rgb = np.clip(neighbor_rgb * ratio[..., None], 0, 255).astype(np.uint8)

        mask3 = np.broadcast_to(suspicious[..., None], rgb.shape)
        arr[..., :3] = np.where(mask3, new_rgb, arr[..., :3])
        recolored = int(suspicious.sum())

    # Pass C: opaque-island removal. Connected components on α≥island_min_alpha;
    # small non-body components with key-hue dominance get zeroed entirely.
    islands_dropped = 0
    island_pixels = 0
    alpha_now = arr[..., 3]
    foreground_island = alpha_now >= config.boundary_cleanup_island_min_alpha
    labels, n_components = ndimage.label(foreground_island)
    if n_components > 1:
        sizes = ndimage.sum(
            foreground_island, labels, index=range(1, n_components + 1)
        )
        body_label = int(np.argmax(sizes)) + 1
        # Vectorized per-component mean key_excess.
        ke_means = ndimage.mean(
            key_excess, labels, index=range(1, n_components + 1)
        )
        # Build a mask of "to-drop" component labels.
        to_drop_labels = []
        for i in range(n_components):
            lbl = i + 1
            if lbl == body_label:
                continue
            sz = int(sizes[i])
            if sz > config.boundary_cleanup_island_max_size:
                continue
            if ke_means[i] > config.boundary_cleanup_island_hue_margin:
                to_drop_labels.append(lbl)
        if to_drop_labels:
            drop_mask_c = np.isin(labels, to_drop_labels)
            island_pixels = int(drop_mask_c.sum())
            islands_dropped = len(to_drop_labels)
            arr[..., 3][drop_mask_c] = 0

    cut = Image.fromarray(arr, mode="RGBA")
    marker = _marker({
        "radius_px": config.boundary_cleanup_radius_px,
        "drop_hue_margin": config.boundary_cleanup_drop_hue_margin,
        "drop_saturation": config.boundary_cleanup_drop_saturation,
        "recolor_hue_margin": config.boundary_cleanup_recolor_hue_margin,
        "island_max_size": config.boundary_cleanup_island_max_size,
        "island_hue_margin": config.boundary_cleanup_island_hue_margin,
        "pixels_dropped": dropped,
        "pixels_recolored": recolored,
        "islands_dropped": islands_dropped,
        "island_pixels": island_pixels,
    })
    log = (
        f"boundary cleanup: dropped {dropped} near-boundary opaque key-hue "
        f"pixels to α=0, recolored {recolored} weakly-tinted pixels from "
        f"local clean neighbors, dropped {islands_dropped} key-tinted "
        f"opaque islands ({island_pixels} pixels)"
    )
    return cut, marker, log


def _stage_chroma_knockout(np, cut, key_rgb, config):
    """Stage 5: RGB-distance knockout for residual key pockets.

    Two rules: (1) RGB within ``chroma_tolerance`` of pure key, OR (2)
    the key channel(s) dominate the off-channel(s). Only fires on α in
    ``[chroma_opaque_threshold, chroma_max_alpha]`` — skipping α=255
    avoids biting saturated accents the matte verified are foreground.

    Generalized over key colors:

      - Single-channel keys (green, red, blue) — one channel at max,
        two off-channels: rule 2 fires when the key channel exceeds
        ``max(off_a, off_b) + dominance_margin``, with
        ``off_channel_ceiling`` and ``off_channel_asymmetry`` guarding
        against pixels that happen to have one bright channel.

      - Multi-channel keys (magenta=R+B, yellow=R+G, cyan=G+B) — two
        channels at max, one off-channel: rule 2 fires when ALL key
        channels exceed ``off + dominance_margin`` AND the off-channel
        is below ``off_channel_ceiling``.

    Without this generalization, magenta ``argmax``'d to just R, and
    the rule fired on any R-dominant pixel — eating warm skin and pink
    clothing wholesale.
    """
    if key_rgb is None:
        return cut, None, None
    from PIL import Image
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    arr = np.array(cut)
    rgb = arr[..., :3].astype(np.int32)
    alpha = arr[..., 3]
    key = np.array(key_rgb, dtype=np.int32)

    distance = np.sqrt(((rgb - key) ** 2).sum(axis=-1))
    near_key = distance < config.chroma_tolerance

    # Identify key channels via the midpoint rule (same as despill) —
    # robust to model-rendered key drift like (250, 4, 233) ≠ (255, 0, 255).
    key_min = min(key_rgb)
    key_max = max(key_rgb)
    threshold = (key_max + key_min) / 2.0
    key_channels = [i for i in (0, 1, 2) if key_rgb[i] > threshold]
    off_channels = [i for i in (0, 1, 2) if i not in key_channels]

    if not off_channels:
        # Degenerate: white/grey/black key — dominance can't apply.
        key_dominant = np.zeros(rgb.shape[:2], dtype=bool)
    elif len(off_channels) == 1:
        off_band = rgb[..., off_channels[0]]
        key_dom = np.ones(rgb.shape[:2], dtype=bool)
        for kc in key_channels:
            key_dom &= rgb[..., kc] > (off_band + config.chroma_dominance)
        key_dominant = key_dom & (off_band < config.chroma_off_channel_ceiling)
    else:
        kc = key_channels[0]
        key_band = rgb[..., kc]
        other_a = rgb[..., off_channels[0]]
        other_b = rgb[..., off_channels[1]]
        other_max = np.maximum(other_a, other_b)
        other_asymmetry = np.abs(other_a - other_b)
        key_dominant = (
            (key_band > (other_max + config.chroma_dominance))
            & (
                (other_max < config.chroma_off_channel_ceiling)
                | (other_asymmetry < config.chroma_off_channel_asymmetry)
            )
        )

    mask = (
        (near_key | key_dominant)
        & (alpha >= config.chroma_opaque_threshold)
        & (alpha <= config.chroma_max_alpha)
    )
    knocked = int(mask.sum())
    arr[..., 3][mask] = 0
    cut = Image.fromarray(arr, mode="RGBA")
    key_hex = f"#{key_rgb[0]:02X}{key_rgb[1]:02X}{key_rgb[2]:02X}"
    marker = _marker({
        "key": key_hex,
        "tolerance": config.chroma_tolerance,
        "dominance": config.chroma_dominance,
        "off_channel_ceiling": config.chroma_off_channel_ceiling,
        "off_channel_asymmetry": config.chroma_off_channel_asymmetry,
        "pixels_knocked": knocked,
    })
    log = f"chroma-key knockout: {knocked} interior pixels removed"
    return cut, marker, log


def _stage_dead_pixels(np, cut, key_rgb, config):
    """Stage 6: restore α on isolated transparent pixels surrounded by opaque ones.

    Key-aware: skips restoration of pixels whose own RGB is strongly
    key-tinted (per ``dead_pixel_key_excess_skip``). Without this guard,
    a chroma_knockout-created hole sitting in a band of α≈254 hair
    pixels would have its newly-α=0 pixel restored to α=255 with the
    untouched lime RGB intact — visible as a green dot in the
    foreground. The skip lets these pixels remain holes for the pocket
    detector / fill to handle, rather than silently resurrecting them.
    """
    if config.dead_pixel_threshold is None:
        return cut, None, None
    from PIL import Image
    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    arr = np.array(cut)
    alpha = arr[..., 3]

    opaque_mask = (alpha >= _DEAD_PIXEL_OPAQUE).astype(np.uint8)
    padded = np.pad(opaque_mask, 1, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    neighbor_count = windows.sum(axis=(2, 3)) - opaque_mask

    is_dead = (alpha == 0) & (neighbor_count >= config.dead_pixel_threshold)

    # Key-awareness gate: drop pixels whose own RGB is strongly key-tinted.
    # Same midpoint rule for key/off channels as despill / knockout, so the
    # three stages stay consistent on what counts as "key-cast".
    skipped_key = 0
    skip_threshold = config.dead_pixel_key_excess_skip
    if skip_threshold is not None and key_rgb is not None and is_dead.any():
        key_min = min(key_rgb)
        key_max = max(key_rgb)
        mid = (key_max + key_min) / 2.0
        key_channels = [i for i in (0, 1, 2) if key_rgb[i] > mid]
        off_channels = [i for i in (0, 1, 2) if i not in key_channels]
        if key_channels and off_channels:
            rgb = arr[..., :3].astype(np.int32)
            if len(key_channels) == 1:
                kc = key_channels[0]
                ref = np.maximum(rgb[..., off_channels[0]], rgb[..., off_channels[1]])
                key_excess = rgb[..., kc] - ref
            else:
                ka, kb = key_channels
                oc = off_channels[0]
                key_excess = np.minimum(
                    rgb[..., ka] - rgb[..., oc],
                    rgb[..., kb] - rgb[..., oc],
                )
            tinted = key_excess > skip_threshold
            skip_mask = is_dead & tinted
            skipped_key = int(skip_mask.sum())
            is_dead = is_dead & ~tinted

    restored = int(is_dead.sum())
    arr[..., 3][is_dead] = 255
    cut = Image.fromarray(arr, mode="RGBA")
    marker = _marker({
        "neighbor_threshold": config.dead_pixel_threshold,
        "pixels_restored": restored,
        "key_excess_skip": skip_threshold if skip_threshold is not None else "off",
        "skipped_key_tinted": skipped_key,
    })
    log = (
        f"dead-pixel fill: {restored} isolated transparent pixels restored"
        + (f", {skipped_key} skipped (key-tinted)" if skipped_key else "")
    )
    return cut, marker, log


def _stage_pockets(np, cut, config):
    """Stage 8: connected-component diagnostic for α=0 clusters enclosed by foreground.

    A component is "external" (real background) if it touches the canvas
    border, "internal" (a pocket) if it does not. Internal pockets are
    almost always matte damage — chroma matte chewing into figure
    regions whose YCbCr chroma is too close to the key, dead-pixel-fill
    unable to restore them because clusters fail the 6-of-8 neighbor test.

    Runs after trim, so reported pocket coordinates refer to the saved
    file. (Trim can't change pocket topology: cropping only removes the
    frame outside the alpha bbox, and any transparent component that
    reached the old border still reaches the new one.)
    """
    if not config.pocket_detect:
        return cut, None, None
    from PIL import Image
    from scipy import ndimage

    if cut.mode != "RGBA":
        cut = cut.convert("RGBA")
    arr = np.array(cut)
    alpha = arr[..., 3]

    transparent = alpha < config.pocket_alpha_threshold
    labels, n_components = ndimage.label(transparent)

    def _build_marker(count, filled, filled_pixels):
        return _marker({
            "alpha_threshold": config.pocket_alpha_threshold,
            "count": count,
            "filled": filled,
            "filled_pixels": filled_pixels,
            "fill_max_size": config.pocket_fill_max_size,
        })

    if n_components == 0:
        return cut, _build_marker(0, 0, 0), "pocket detection: clean (no internal pockets)"

    border_labels = set()
    for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        border_labels.update(int(v) for v in np.unique(edge) if v != 0)

    pocket_labels = sorted(set(range(1, n_components + 1)) - border_labels)
    if not pocket_labels:
        return cut, _build_marker(0, 0, 0), "pocket detection: clean (no internal pockets)"

    sizes_arr = ndimage.sum(transparent, labels, index=pocket_labels)
    centroids_yx = ndimage.center_of_mass(transparent, labels, index=pocket_labels)
    pocket_records = sorted(
        zip(pocket_labels, sizes_arr, centroids_yx), key=lambda r: -r[1]
    )
    sizes = [int(s) for _, s, _ in pocket_records]
    centroids = [(int(round(c[1])), int(round(c[0]))) for _, _, c in pocket_records]

    filled = 0
    filled_pixels = 0
    fill_cap = config.pocket_fill_max_size
    if fill_cap is not None and fill_cap > 0:
        fill_labels = [lbl for lbl, sz, _ in pocket_records if sz <= fill_cap]
        if fill_labels:
            fill_mask = np.isin(labels, fill_labels)
            arr[..., 3][fill_mask] = 255
            filled = len(fill_labels)
            filled_pixels = int(fill_mask.sum())
            cut = Image.fromarray(arr, mode="RGBA")

    count = len(pocket_records)
    total = sum(sizes)
    median = sizes[len(sizes) // 2]
    largest = sizes[0]
    largest_at = centroids[0]
    fill_summary = ""
    if filled:
        fill_summary = (
            f" — filled {filled} pocket(s) ≤{fill_cap}px "
            f"({filled_pixels} pixels restored)"
        )
    log = (
        f"pocket detection: ⚠ {count} internal pocket(s) "
        f"(median={median}px, max={largest}px, total={total}px); "
        f"largest at (x={largest_at[0]}, y={largest_at[1]}){fill_summary}"
    )
    return cut, _build_marker(count, filled, filled_pixels), log


def _stage_trim(cut, src_size, config):
    """Stage 7: crop to the alpha bounding box, ignoring sub-threshold residue."""
    if not config.trim or cut.mode != "RGBA":
        return cut, None, None
    alpha = cut.split()[3]
    mask = alpha.point(lambda v: 255 if v >= config.trim_threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return cut, None, None
    cut = cut.crop(bbox)
    # ``from`` is a Python keyword so we build the dict literally.
    marker = _marker({
        "threshold": config.trim_threshold,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "from": f"{src_size[0]}x{src_size[1]}",
    })
    return cut, marker, None  # no log line (matches prior behavior)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

# PNG metadata key for each stage's marker. Keys are stable — they're
# read by ``prompts-from-image`` and by humans inspecting saved files.
_MARKER_KEYS = {
    "chroma_key_resolution": "chroma_key_resolution",
    "alpha_matting":         "alpha_matting",
    "chroma_matte":          "chroma_matte",
    "decontaminate":         "foreground_decontamination",
    "despill":               "interior_despill",
    "boundary_cleanup":      "boundary_cleanup",
    "chroma_knockout":       "chroma_key_knockout",
    "boundary_cleanup_post": "boundary_cleanup_post",
    "dead_pixels":           "dead_pixel_fill",
    "trim":                  "alpha_trimmed",
    "pockets":               "internal_pockets",
}


def _write_png_with_metadata(cut, output, src_metadata, model, source_name, markers):
    """Write `cut` to `output`, preserving src metadata and adding stage markers."""
    from PIL import PngImagePlugin
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in src_metadata.items():
        if isinstance(v, (str, bytes)):
            pnginfo.add_text(k, v if isinstance(v, str) else v.decode("utf-8", "replace"))
    pnginfo.add_text("bg_removal_model", model)
    pnginfo.add_text("bg_removed_from", source_name)
    for stage_key, png_key in _MARKER_KEYS.items():
        marker = markers.get(stage_key)
        if marker is not None:
            pnginfo.add_text(png_key, marker)
    output.parent.mkdir(parents=True, exist_ok=True)
    cut.save(output, pnginfo=pnginfo, optimize=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _rembg_session(model: str):
    """Cached rembg session per model name.

    ``rembg.new_session`` loads the full ONNX model (~180MB) on every
    call and does no caching of its own — without this, a batch run
    reloads the model once per image.
    """
    from rembg import new_session
    return new_session(model)


def remove_background(input_path: Path, config: RemoveBgConfig,
                      output_path: Optional[Path] = None,
                      verbose: bool = True) -> Path:
    """Remove the background from `input_path`, writing `<stem>-cutout.png`.

    Honors `config` for model selection, chroma-key tuning, dead-pixel
    fill, and trim. If `output_path` is None, defaults to
    `<input>-cutout.png` (siblings the source). Returns the resolved
    output path.

    Raises `BgRemovalNotInstalled` if the `bg-removal` dependency group
    is not installed.
    """
    try:
        import numpy as np
        from PIL import Image
        import rembg  # noqa: F401 — presence check; sessions come from _rembg_session
    except ImportError as e:
        raise BgRemovalNotInstalled(
            f"bg-removal dependencies are not installed (missing: {e.name}). "
            "Install with: uv sync --project tools/ai_art_generator --group bg-removal"
        ) from e

    if not input_path.exists():
        raise FileNotFoundError(f"image not found: {input_path}")

    output = output_path or input_path.with_name(f"{input_path.stem}-cutout.png")
    if config.backup and output.exists():
        backup_path = backup_existing(output)
        if verbose:
            print(f"  📁 Backed up existing file to: {backup_path}", file=sys.stderr)

    src = Image.open(input_path)
    src_metadata = dict(src.info)
    src_size = src.size

    markers: dict[str, Optional[str]] = {}
    dump_dir = config.dump_stages_dir
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    stage_counter = [0]  # mutable cell for closure

    def _step(stage_key: str, result):
        """Record marker, print log, optionally dump intermediate, return image.

        ``result`` is ``(img, marker, log)``. Stages that no-op (returned
        marker=None) are not dumped — there'd be no diff from the prior
        stage's output.
        """
        img, marker, log = result
        markers[stage_key] = marker
        if log and verbose:
            print(f"  {log}", file=sys.stderr)
        stage_counter[0] += 1
        if dump_dir is not None and marker is not None:
            dump_path = dump_dir / f"{input_path.stem}-{stage_counter[0]:02d}-{stage_key}.png"
            img.save(dump_path)
            if verbose:
                print(f"    [dump] {dump_path}", file=sys.stderr)
        return img

    # Stage 0: resolve chroma key (no image to thread through). The RGB
    # array is shared with the chroma matte below — the full-image
    # conversion isn't free, don't do it twice.
    src_rgb = np.asarray(src.convert("RGB"))
    key_rgb_tuple, key_marker = _stage_resolve_chroma_key(np, src_rgb, config, verbose)
    markers["chroma_key_resolution"] = key_marker

    # Stages 1–8: linear pipeline.
    # Stage 1 dispatches on `model` — `sam3:<prompt>` shells out to the
    # standalone SAM tool; anything else uses rembg.
    if _is_sam_model(config.model):
        cut = _step("alpha_matting",
                    _stage_sam_mask(input_path, config, _sam_prompt(config.model)))
    else:
        session = _rembg_session(config.model)
        cut = _step("alpha_matting", _stage_alpha_matting(src, session, config))
    cut = _step("chroma_matte",      _stage_chroma_matte(np, src_rgb, cut, key_rgb_tuple, config))
    cut = _step("decontaminate",     _stage_decontaminate(np, src, cut, config))
    cut = _step("despill",           _stage_despill(np, cut, key_rgb_tuple, config))
    cut = _step("boundary_cleanup",  _stage_boundary_cleanup(np, cut, key_rgb_tuple, config))
    cut = _step("chroma_knockout",   _stage_chroma_knockout(np, cut, key_rgb_tuple, config))
    if config.boundary_cleanup_post:
        cut = _step("boundary_cleanup_post",
                    _stage_boundary_cleanup(np, cut, key_rgb_tuple, config))
    else:
        markers["boundary_cleanup_post"] = None
    cut = _step("dead_pixels",       _stage_dead_pixels(np, cut, key_rgb_tuple, config))
    # Trim before pockets so pocket coordinates refer to the saved file.
    # Safe: dead-pixel/pocket fills only restore interior pixels, which
    # can't extend the alpha bbox.
    cut = _step("trim",              _stage_trim(cut, src_size, config))
    cut = _step("pockets",           _stage_pockets(np, cut, config))

    _write_png_with_metadata(cut, output, src_metadata, config.model, input_path.name, markers)
    return output
