"""Unit tests for the bg-removal pipeline stages.

Every test drives a stage function directly with a small synthetic
array — no rembg session, no model download, no real image files. The
fixtures encode the failure modes the stage docstrings describe (blanket
clamp halos, magenta eating warm skin, resurrection of key-tinted dead
pixels, ...), so a regression in any tuned rule fails a named test
instead of silently degrading cutouts.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from postprocess.remove_bg import (
    RemoveBgConfig,
    _rembg_session,
    _snap_to_canonical,
    _stage_boundary_cleanup,
    _stage_chroma_knockout,
    _stage_dead_pixels,
    _stage_despill,
    _stage_pockets,
    _stage_trim,
    detect_chroma_key,
    parse_hex_color,
)

GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)


def rgba(h, w, color=(128, 128, 128), alpha=255):
    """Solid HxW RGBA array."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = color
    arr[..., 3] = alpha
    return arr


def img(arr):
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# Hex parsing / canonical snap / key detection
# ---------------------------------------------------------------------------

class TestParseHexColor:
    def test_with_hash(self):
        assert parse_hex_color("#00FF00") == (0, 255, 0)

    def test_without_hash(self):
        assert parse_hex_color("ff00ff") == (255, 0, 255)

    def test_bad_length_raises(self):
        with pytest.raises(ValueError):
            parse_hex_color("#fff")


class TestSnapToCanonical:
    def test_drifted_magenta_snaps(self):
        # Typical model-rendered drift on a magenta plate.
        assert _snap_to_canonical((250, 4, 233)) == (255, 0, 255)

    def test_distant_color_unchanged(self):
        # Olive is ~179 from yellow, outside the 60 snap radius.
        assert _snap_to_canonical((128, 128, 0)) == (128, 128, 0)


class TestDetectChromaKey:
    def test_flat_green_plate(self):
        plate = np.zeros((200, 200, 3), dtype=np.uint8)
        plate[...] = GREEN
        assert detect_chroma_key(np, plate) == GREEN

    def test_drifted_plate_snaps(self):
        plate = np.zeros((200, 200, 3), dtype=np.uint8)
        plate[...] = (10, 245, 12)
        assert detect_chroma_key(np, plate) == GREEN

    def test_grey_plate_rejected(self):
        plate = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert detect_chroma_key(np, plate) is None

    def test_disagreeing_corners_rejected(self):
        plate = np.zeros((200, 200, 3), dtype=np.uint8)
        plate[...] = GREEN
        plate[:64, :64] = (255, 0, 0)  # one red corner
        assert detect_chroma_key(np, plate) is None

    def test_too_small_image(self):
        plate = np.zeros((100, 100, 3), dtype=np.uint8)
        plate[...] = GREEN
        assert detect_chroma_key(np, plate) is None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestRemoveBgConfigFromDict:
    def test_empty_gives_defaults(self):
        assert RemoveBgConfig.from_dict(None) == RemoveBgConfig()
        assert RemoveBgConfig.from_dict({}) == RemoveBgConfig()

    def test_known_key_applied(self):
        cfg = RemoveBgConfig.from_dict({"model": "birefnet-portrait"})
        assert cfg.model == "birefnet-portrait"

    def test_unknown_key_warns_and_is_ignored(self, capsys):
        cfg = RemoveBgConfig.from_dict({"despill_treshold": 20})
        assert cfg.despill_threshold == RemoveBgConfig().despill_threshold
        assert "despill_treshold" in capsys.readouterr().err

    @pytest.mark.parametrize("raw", [False, "false", "none", ""])
    def test_chroma_key_falsey_disables(self, raw):
        assert RemoveBgConfig.from_dict({"chroma_key": raw}).chroma_key is None

    @pytest.mark.parametrize("raw", ["auto", "AUTO", "Auto"])
    def test_chroma_key_auto_normalized(self, raw):
        assert RemoveBgConfig.from_dict({"chroma_key": raw}).chroma_key == "auto"


# ---------------------------------------------------------------------------
# Stage 4: interior despill
# ---------------------------------------------------------------------------

class TestDespill:
    def test_green_spill_clamped_to_off_channel_max(self):
        arr = rgba(4, 4)
        arr[0, 0, :3] = (100, 200, 100)  # +100 green excess
        cut, marker, _ = _stage_despill(np, img(arr), GREEN, RemoveBgConfig())
        out = np.array(cut)
        assert tuple(out[0, 0, :3]) == (100, 100, 100)
        assert "pixels_corrected=1" in marker

    def test_below_threshold_protected(self):
        # +5 excess is noise (teal gem territory), not spill.
        arr = rgba(4, 4, color=(100, 105, 100))
        cut, _, _ = _stage_despill(np, img(arr), GREEN, RemoveBgConfig())
        assert tuple(np.array(cut)[0, 0, :3]) == (100, 105, 100)

    def test_partial_alpha_untouched(self):
        # α below despill_min_alpha is decontamination's domain — the
        # tinted-halo bug that retired the old blanket clamp.
        arr = rgba(4, 4, color=(100, 200, 100), alpha=200)
        cut, _, _ = _stage_despill(np, img(arr), GREEN, RemoveBgConfig())
        assert tuple(np.array(cut)[0, 0, :3]) == (100, 200, 100)

    def test_magenta_key_clamps_jointly(self):
        # Multi-channel key: R and B clamp together against G. The old
        # argmax rule keyed on R alone and ate warm skin.
        arr = rgba(4, 4)
        arr[0, 0, :3] = (200, 100, 190)  # joint excess = min(100, 90) = 90
        cut, _, _ = _stage_despill(np, img(arr), MAGENTA, RemoveBgConfig())
        assert tuple(np.array(cut)[0, 0, :3]) == (110, 100, 100)

    def test_no_key_is_noop(self):
        arr = rgba(4, 4, color=(100, 200, 100))
        cut, marker, _ = _stage_despill(np, img(arr), None, RemoveBgConfig())
        assert marker is None
        assert tuple(np.array(cut)[0, 0, :3]) == (100, 200, 100)


# ---------------------------------------------------------------------------
# Stage 5: chroma-key knockout
# ---------------------------------------------------------------------------

class TestChromaKnockout:
    def test_key_pixel_in_alpha_band_knocked(self):
        arr = rgba(4, 4, color=GREEN, alpha=200)
        cut, marker, _ = _stage_chroma_knockout(np, img(arr), GREEN, RemoveBgConfig())
        assert (np.array(cut)[..., 3] == 0).all()
        assert "pixels_knocked=16" in marker

    def test_fully_opaque_protected(self):
        # α=255 is matte-confident foreground (chroma_max_alpha=254).
        arr = rgba(4, 4, color=GREEN, alpha=255)
        cut, _, _ = _stage_chroma_knockout(np, img(arr), GREEN, RemoveBgConfig())
        assert (np.array(cut)[..., 3] == 255).all()

    def test_low_alpha_edge_protected(self):
        # Below chroma_opaque_threshold: decontamination's territory.
        arr = rgba(4, 4, color=GREEN, alpha=50)
        cut, _, _ = _stage_chroma_knockout(np, img(arr), GREEN, RemoveBgConfig())
        assert (np.array(cut)[..., 3] == 50).all()

    def test_neutral_pixel_kept(self):
        arr = rgba(4, 4, color=(128, 128, 128), alpha=200)
        cut, _, _ = _stage_chroma_knockout(np, img(arr), GREEN, RemoveBgConfig())
        assert (np.array(cut)[..., 3] == 200).all()


# ---------------------------------------------------------------------------
# Stage 4.5 / 6.5: boundary cleanup
# ---------------------------------------------------------------------------

def _boundary_fixture():
    """20x20: left half transparent, right half opaque skin."""
    arr = rgba(20, 20, color=(180, 150, 130))
    arr[:, :10, 3] = 0
    return arr


class TestBoundaryCleanup:
    def test_pass_a_drops_saturated_key_blob(self):
        arr = _boundary_fixture()
        arr[5, 12, :3] = (30, 220, 40)  # lime blob near the hole
        cut, marker, _ = _stage_boundary_cleanup(np, img(arr), GREEN, RemoveBgConfig())
        assert np.array(cut)[5, 12, 3] == 0
        assert "pixels_dropped=1" in marker

    def test_pass_b_recolors_weak_tint_from_clean_neighbor(self):
        arr = _boundary_fixture()
        arr[6, 12, :3] = (140, 160, 130)  # +20 green cast: too weak for A
        cut, marker, _ = _stage_boundary_cleanup(np, img(arr), GREEN, RemoveBgConfig())
        out = np.array(cut)[6, 12, :3].astype(int)
        # Recolored to the skin neighbors' hue: green no longer dominant.
        assert out[1] <= max(out[0], out[2])
        assert "pixels_recolored=" in marker

    def test_pass_c_drops_weakly_tinted_island(self):
        arr = _boundary_fixture()
        # 2x2 island floating in the transparent sea, tint too weak for
        # pass A (excess 20 < 25 with saturation gate) — pass C's case.
        arr[0:2, 0:2, :3] = (120, 140, 110)
        arr[0:2, 0:2, 3] = 255
        cut, marker, _ = _stage_boundary_cleanup(np, img(arr), GREEN, RemoveBgConfig())
        assert (np.array(cut)[0:2, 0:2, 3] == 0).all()
        assert "islands_dropped=1" in marker

    def test_clean_foreground_untouched(self):
        arr = _boundary_fixture()
        cut, _, _ = _stage_boundary_cleanup(np, img(arr), GREEN, RemoveBgConfig())
        out = np.array(cut)
        assert (out[:, 10:, 3] == 255).all()
        assert (out[:, 10:, :3] == (180, 150, 130)).all()


# ---------------------------------------------------------------------------
# Stage 6: dead-pixel fill
# ---------------------------------------------------------------------------

class TestDeadPixels:
    def test_isolated_hole_restored(self):
        arr = rgba(5, 5)
        arr[2, 2, 3] = 0
        cut, marker, _ = _stage_dead_pixels(np, img(arr), GREEN, RemoveBgConfig())
        assert np.array(cut)[2, 2, 3] == 255
        assert "pixels_restored=1" in marker

    def test_cluster_not_restored(self):
        # A 2x2 hole: each pixel has only 5 opaque neighbors (< 6-of-8).
        arr = rgba(6, 6)
        arr[2:4, 2:4, 3] = 0
        cut, _, _ = _stage_dead_pixels(np, img(arr), GREEN, RemoveBgConfig())
        assert (np.array(cut)[2:4, 2:4, 3] == 0).all()

    def test_key_tinted_hole_not_resurrected(self):
        # Restoring a lime pixel undoes the knockout's correct decision —
        # the "green dot in hair shadow" bug the skip gate exists for.
        arr = rgba(5, 5)
        arr[2, 2] = (50, 200, 50, 0)
        cut, marker, _ = _stage_dead_pixels(np, img(arr), GREEN, RemoveBgConfig())
        assert np.array(cut)[2, 2, 3] == 0
        assert "skipped_key_tinted=1" in marker

    def test_skip_gate_off_restores_anything(self):
        arr = rgba(5, 5)
        arr[2, 2] = (50, 200, 50, 0)
        cfg = RemoveBgConfig(dead_pixel_key_excess_skip=None)
        cut, _, _ = _stage_dead_pixels(np, img(arr), GREEN, cfg)
        assert np.array(cut)[2, 2, 3] == 255

    def test_disabled(self):
        arr = rgba(5, 5)
        arr[2, 2, 3] = 0
        cfg = RemoveBgConfig(dead_pixel_threshold=None)
        cut, marker, _ = _stage_dead_pixels(np, img(arr), GREEN, cfg)
        assert marker is None
        assert np.array(cut)[2, 2, 3] == 0


# ---------------------------------------------------------------------------
# Stage 8: internal-pocket diagnostic
# ---------------------------------------------------------------------------

class TestPockets:
    def test_internal_pocket_reported(self):
        arr = rgba(9, 9)
        arr[4:6, 4:6, 3] = 0
        cut, marker, log = _stage_pockets(np, img(arr), RemoveBgConfig())
        assert "count=1" in marker
        assert "⚠" in log
        # Warn-only by default: the hole stays.
        assert (np.array(cut)[4:6, 4:6, 3] == 0).all()

    def test_pocket_filled_under_cap(self):
        arr = rgba(9, 9)
        arr[4:6, 4:6, 3] = 0
        cfg = RemoveBgConfig(pocket_fill_max_size=10)
        cut, marker, _ = _stage_pockets(np, img(arr), cfg)
        assert (np.array(cut)[..., 3] == 255).all()
        assert "filled=1" in marker

    def test_pocket_over_cap_left_for_review(self):
        arr = rgba(9, 9)
        arr[4:6, 4:6, 3] = 0
        cfg = RemoveBgConfig(pocket_fill_max_size=3)
        cut, marker, _ = _stage_pockets(np, img(arr), cfg)
        assert (np.array(cut)[4:6, 4:6, 3] == 0).all()
        assert "filled=0" in marker

    def test_border_touching_region_is_background_not_pocket(self):
        arr = rgba(9, 9)
        arr[:, 0, 3] = 0  # transparent strip on the border
        _, marker, log = _stage_pockets(np, img(arr), RemoveBgConfig())
        assert "count=0" in marker
        assert "clean" in log


# ---------------------------------------------------------------------------
# Stage 7: trim
# ---------------------------------------------------------------------------

class TestTrim:
    def test_crops_to_alpha_bbox(self):
        arr = rgba(10, 10, alpha=0)
        arr[4:7, 4:7, 3] = 255
        cut, marker, _ = _stage_trim(img(arr), (10, 10), RemoveBgConfig())
        assert cut.size == (3, 3)
        assert "bbox=4,4,7,7" in marker

    def test_subthreshold_residue_ignored(self):
        arr = rgba(10, 10, alpha=0)
        arr[4:7, 4:7, 3] = 255
        arr[0, 0, 3] = 5  # residue below trim_threshold=10
        cut, _, _ = _stage_trim(img(arr), (10, 10), RemoveBgConfig())
        assert cut.size == (3, 3)

    def test_fully_transparent_untouched(self):
        arr = rgba(10, 10, alpha=0)
        cut, marker, _ = _stage_trim(img(arr), (10, 10), RemoveBgConfig())
        assert cut.size == (10, 10)
        assert marker is None


# ---------------------------------------------------------------------------
# Session caching
# ---------------------------------------------------------------------------

class TestRembgSessionCache:
    def test_session_reused_per_model(self, monkeypatch):
        calls = []
        fake = types.SimpleNamespace(
            new_session=lambda model: calls.append(model) or f"session-{model}"
        )
        monkeypatch.setitem(sys.modules, "rembg", fake)
        _rembg_session.cache_clear()
        try:
            a = _rembg_session("isnet-anime")
            b = _rembg_session("isnet-anime")
            c = _rembg_session("birefnet-portrait")
            assert a is b
            assert c != a
            assert calls == ["isnet-anime", "birefnet-portrait"]
        finally:
            _rembg_session.cache_clear()
