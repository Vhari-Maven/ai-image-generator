"""Unit tests for the OpenAI generator's model-capability helpers.

No network: these exercise pure functions and the pre-request parameter
assembly. The `openai` client is constructed with a dummy key, which the
SDK allows without making any call.
"""
from __future__ import annotations

import pytest

from generators.openai_image import (
    OpenAIImageGenerator,
    quality_tiers_for,
    supports_native_transparency,
)


class TestQualityTiers:
    def test_25_models_get_xhigh_and_max(self):
        for m in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
            tiers = quality_tiers_for(m)
            assert "xhigh" in tiers and "max" in tiers

    def test_dated_snapshot_matches_family(self):
        assert "max" in quality_tiers_for("gpt-image-2.5-flare-2026-09-08")

    def test_older_models_capped_at_high(self):
        for m in ("gpt-image-2", "gpt-image-1.5", "gpt-image-1"):
            tiers = quality_tiers_for(m)
            assert "xhigh" not in tiers and "max" not in tiers
            assert tiers == ("auto", "low", "medium", "high")


class TestNativeTransparency:
    @pytest.mark.parametrize("m", [
        "gpt-image-2.5-flare", "gpt-image-2.5-sunburst",
        "gpt-image-2.5-flare-2026-09-08", "gpt-image-1.5", "gpt-image-1",
    ])
    def test_supported(self, m):
        assert supports_native_transparency(m)

    def test_gpt_image_2_is_not(self):
        assert not supports_native_transparency("gpt-image-2")


class _Prompt:
    def __init__(self, remove_background=None):
        self.remove_background = remove_background


@pytest.fixture
def gen(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return OpenAIImageGenerator()


class TestRequestAssembly:
    def test_default_model_is_flare(self, gen):
        assert gen.model_name == "gpt-image-2.5-flare"

    def test_transparency_requested_on_flare(self, gen):
        gen.model_name = "gpt-image-2.5-flare"
        params = {}
        assert gen._maybe_request_native_transparency(_Prompt("true"), params)
        assert params == {"background": "transparent", "output_format": "png"}

    def test_transparency_skipped_on_gpt_image_2(self, gen):
        gen.model_name = "gpt-image-2"
        params = {}
        assert not gen._maybe_request_native_transparency(_Prompt("true"), params)
        assert params == {}

    def test_rembg_model_override_never_uses_native(self, gen):
        gen.model_name = "gpt-image-2.5-flare"
        params = {}
        assert not gen._maybe_request_native_transparency(
            _Prompt("birefnet-portrait"), params
        )
        assert params == {}

    def test_max_quality_rejected_on_gpt_image_2(self, gen, tmp_path):
        gen.model_name = "gpt-image-2"
        prompt = _Prompt()
        prompt.filename = "x.png"
        prompt.prompt = "a cat"
        prompt.id = "x"
        with pytest.raises(ValueError, match="not supported by gpt-image-2"):
            gen.generate_image(prompt, str(tmp_path), quality="max")
