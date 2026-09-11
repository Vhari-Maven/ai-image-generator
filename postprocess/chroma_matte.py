"""YCbCr-space chroma-key matte.

Derives a per-pixel α from each pixel's chroma distance to a known key
color. Unlike a binary chroma-key knockout, this gives a continuous α
that smoothly falls off across a tunable transition band — pure key →
α=0, well-clear-of-key → α=1, in-between gets a partial α.

Operates in YCbCr (BT.601) so chroma distance is independent of
luminance: dark hair near the key in luma still keys correctly, and
hues that look RGB-similar to the key (cyan ≈ green in raw RGB
distance) separate cleanly on the Cb/Cr plane.

Used as a primary matte source alongside the segmentation network's α.
The two are combined (typically `min`) so a pixel must be foreground
*both* per the segmentation silhouette *and* per the chroma physics —
which lets the chroma matte catch interior pockets the segmentation
misses, while the segmentation matte catches non-key clutter the
chroma matte couldn't see.
"""
from __future__ import annotations

from typing import Tuple


# BT.601 RGB→YCbCr matrix (the "video" version with Cb/Cr in [0, 255]).
# Y = 0.299·R + 0.587·G + 0.114·B
# Cb = 128 + (-0.168736·R - 0.331264·G + 0.5·B)
# Cr = 128 + (0.5·R - 0.418688·G - 0.081312·B)
def _rgb_to_ycbcr_chroma(arr_module, rgb):
    """Convert HxWx3 uint8 RGB to (Cb, Cr) float32 arrays in [0, 255]."""
    np = arr_module
    rgb_f = rgb.astype(np.float32)
    R = rgb_f[..., 0]
    G = rgb_f[..., 1]
    B = rgb_f[..., 2]
    Cb = 128.0 + (-0.168736 * R - 0.331264 * G + 0.5 * B)
    Cr = 128.0 + (0.5 * R - 0.418688 * G - 0.081312 * B)
    return Cb, Cr


def _key_chroma(arr_module, key_rgb: Tuple[int, int, int]) -> Tuple[float, float]:
    """Convert a single (R, G, B) key color to scalar (Cb, Cr) in [0, 255]."""
    np = arr_module
    arr = np.array([[list(key_rgb)]], dtype=np.uint8)  # 1x1x3
    Cb, Cr = _rgb_to_ycbcr_chroma(np, arr)
    return float(Cb[0, 0]), float(Cr[0, 0])


def _smoothstep(arr_module, x, edge0, edge1):
    """GLSL smoothstep: 0 below edge0, 1 above edge1, smooth between."""
    np = arr_module
    if edge1 <= edge0:
        # Degenerate band → hard step at edge0.
        return (x > edge0).astype(np.float32)
    t = (x - edge0) / (edge1 - edge0)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def chroma_matte(
    arr_module,
    rgb_image,
    key_rgb: Tuple[int, int, int],
    inner_radius: float,
    outer_radius: float,
):
    """Compute a per-pixel α [0, 1] from YCbCr chroma distance to `key_rgb`.

    Args:
        arr_module: numpy module (passed in to keep this file import-cheap).
        rgb_image: HxWx3 uint8 RGB array.
        key_rgb: (R, G, B) tuple of the chroma-key color.
        inner_radius: chroma distance at or below which α = 0 (pure key).
        outer_radius: chroma distance at or above which α = 1 (clear of key).
                      Must be > inner_radius; band between them gets smoothstep α.

    Returns:
        HxW float32 array of α in [0, 1].
    """
    np = arr_module
    Cb_img, Cr_img = _rgb_to_ycbcr_chroma(np, rgb_image)
    Cb_key, Cr_key = _key_chroma(np, key_rgb)
    dist = np.sqrt((Cb_img - Cb_key) ** 2 + (Cr_img - Cr_key) ** 2)
    return _smoothstep(np, dist, inner_radius, outer_radius)


def combine(arr_module, alpha_seg, alpha_key, mode: str = "min"):
    """Combine a segmentation α (uint8 HxW) with a chroma-key α (float HxW [0,1]).

    Returns a uint8 HxW α array.

    Modes:
        "min":      α_final = min(α_seg, α_key·255)  — strictest; both must agree
                    a pixel is foreground. Default; matches what VFX keyers do
                    when combining a garbage matte with a chroma matte.
        "multiply": α_final = α_seg · α_key  — multiplicative blend; lets either
                    matte attenuate the other. Slightly softer than min in the
                    transition band.
    """
    np = arr_module
    if mode == "min":
        key_u8 = np.clip(alpha_key * 255.0, 0, 255).astype(np.uint8)
        return np.minimum(alpha_seg, key_u8)
    elif mode == "multiply":
        seg_f = alpha_seg.astype(np.float32) / 255.0
        return np.clip(seg_f * alpha_key * 255.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown combine mode: {mode!r} (expected 'min' or 'multiply')")
