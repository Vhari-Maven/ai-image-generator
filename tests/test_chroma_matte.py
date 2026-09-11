"""Unit tests for the YCbCr chroma matte."""
from __future__ import annotations

import numpy as np
import pytest

from postprocess.chroma_matte import chroma_matte, combine

GREEN = (0, 255, 0)


def plate(color, h=4, w=4):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[...] = color
    return arr


class TestChromaMatte:
    def test_pure_key_is_transparent(self):
        alpha = chroma_matte(np, plate(GREEN), GREEN, 25.0, 60.0)
        assert (alpha == 0.0).all()

    def test_chroma_distant_is_opaque(self):
        # White is at Cb=Cr=128, ~136 from green's chroma — well past outer.
        alpha = chroma_matte(np, plate((255, 255, 255)), GREEN, 25.0, 60.0)
        assert (alpha == 1.0).all()

    def test_luma_independence(self):
        # Dark and bright neutral pixels key identically: chroma distance
        # ignores Y by construction.
        dark = chroma_matte(np, plate((40, 40, 40)), GREEN, 25.0, 60.0)
        bright = chroma_matte(np, plate((220, 220, 220)), GREEN, 25.0, 60.0)
        assert np.allclose(dark, bright)

    def test_transition_band_is_partial(self):
        # A near-key green sits between the radii (chroma distance ~40
        # from the key) → smoothstep α in (0, 1).
        alpha = chroma_matte(np, plate((40, 220, 40)), GREEN, 25.0, 60.0)
        assert 0.0 < alpha[0, 0] < 1.0

    def test_degenerate_band_is_hard_step(self):
        # inner == outer → binary matte.
        key_alpha = chroma_matte(np, plate(GREEN), GREEN, 40.0, 40.0)
        far_alpha = chroma_matte(np, plate((255, 255, 255)), GREEN, 40.0, 40.0)
        assert (key_alpha == 0.0).all()
        assert (far_alpha == 1.0).all()


class TestCombine:
    def test_min_takes_strictest(self):
        seg = np.full((2, 2), 200, dtype=np.uint8)
        key = np.full((2, 2), 0.5, dtype=np.float32)  # → 127
        assert (combine(np, seg, key, "min") == 127).all()

    def test_multiply_blends(self):
        seg = np.full((2, 2), 200, dtype=np.uint8)
        key = np.full((2, 2), 0.5, dtype=np.float32)
        assert (combine(np, seg, key, "multiply") == 100).all()

    def test_unknown_mode_raises(self):
        seg = np.zeros((2, 2), dtype=np.uint8)
        key = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            combine(np, seg, key, "screen")
