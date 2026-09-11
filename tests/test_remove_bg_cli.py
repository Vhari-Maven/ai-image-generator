"""Unit tests for the standalone bg-removal CLI's pure helpers."""
from __future__ import annotations

from pathlib import Path

from postprocess.remove_bg_cli import (
    _expected_cutouts,
    _is_source_png,
    _model_slug,
    _parse_models,
)


class TestParseModels:
    def test_strips_and_drops_empties(self):
        assert _parse_models(" a , b ,, c ") == ["a", "b", "c"]


class TestModelSlug:
    def test_plain_rembg_name_roundtrips(self):
        assert _model_slug("birefnet-portrait") == "birefnet-portrait"

    def test_sam_prompt_colon_replaced(self):
        assert _model_slug("sam3:woman") == "sam3-woman"

    def test_awkward_characters_replaced(self):
        assert _model_slug("sam3:long prompt/here") == "sam3-long-prompt-here"


class TestIsSourcePng:
    def test_source_png(self):
        assert _is_source_png(Path("hero.png"))

    def test_own_cutout_skipped(self):
        assert not _is_source_png(Path("hero-cutout.png"))

    def test_model_ab_cutout_skipped(self):
        assert not _is_source_png(Path("hero-cutout-isnet-anime.png"))

    def test_non_png_skipped(self):
        assert not _is_source_png(Path("hero.jpg"))


class TestExpectedCutouts:
    def test_single_model_legacy_path(self):
        assert _expected_cutouts(Path("a/hero.png"), None) == [
            Path("a/hero-cutout.png")
        ]

    def test_models_use_per_model_slugs(self):
        # --skip-existing must look for the names --models actually
        # writes, not the legacy single-model path.
        assert _expected_cutouts(Path("a/hero.png"), ["isnet-anime", "sam3:woman"]) == [
            Path("a/hero-cutout-isnet-anime.png"),
            Path("a/hero-cutout-sam3-woman.png"),
        ]
