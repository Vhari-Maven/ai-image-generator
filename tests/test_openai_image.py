"""Unit tests for the OpenAI generator's model-capability helpers.

No network: these exercise pure functions and the pre-request parameter
assembly. The `openai` client is constructed with a dummy key, which the
SDK allows without making any call.
"""
from __future__ import annotations

import pytest
from PIL import Image

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
    def test_default_model_is_sunburst(self, gen):
        assert gen.model_name == "gpt-image-2.5-sunburst"

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


class _Event:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class TestStreaming:
    def test_off_outside_the_cloud(self, gen, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_REMOTE", raising=False)
        calls = []
        gen._request(lambda **kw: calls.append(kw) or "plain", {"n": 1})
        assert calls == [{"n": 1}]

    def test_cloud_streams_and_returns_completed_image(self, gen, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_REMOTE", "true")
        calls = []

        def call(**kw):
            calls.append(kw)
            return iter([
                _Event("image_generation.partial_image", b64_json="part"),
                _Event("image_generation.completed", b64_json="full",
                       usage="u"),
            ])

        response = gen._request(call, {"n": 1})
        assert calls == [{"n": 1, "stream": True, "partial_images": 1}]
        assert response.data[0].b64_json == "full"
        assert response.usage == "u"

    def test_several_images_are_not_streamed(self, gen, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_REMOTE", "true")
        calls = []
        gen._request(lambda **kw: calls.append(kw), {"n": 2})
        assert calls == [{"n": 2}]

    def test_stream_without_completed_image_fails(self, gen, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_REMOTE", "true")
        with pytest.raises(RuntimeError):
            gen._request(lambda **kw: iter([]), {"n": 1})


class TestReferenceImages:
    def test_hidden_colour_under_transparency_is_cleared(self, tmp_path):
        # A cut-out render: an opaque pixel, a half-transparent edge pixel,
        # and colour left under a fully transparent one.
        im = Image.new("RGBA", (3, 1))
        im.putpixel((0, 0), (200, 40, 160, 255))
        im.putpixel((1, 0), (90, 80, 70, 128))
        im.putpixel((2, 0), (180, 60, 200, 0))
        path = tmp_path / "ref.png"
        im.save(path)

        sent = OpenAIImageGenerator._reference_file(path)
        assert sent.name == "ref.png"
        out = Image.open(sent)
        assert out.getpixel((0, 0)) == (200, 40, 160, 255)
        assert out.getpixel((1, 0)) == (90, 80, 70, 128)
        assert out.getpixel((2, 0)) == (0, 0, 0, 0)

    def test_opaque_image_is_sent_as_the_file(self, tmp_path):
        path = tmp_path / "ref.jpg"
        Image.new("RGB", (2, 2), (10, 20, 30)).save(path)
        sent = OpenAIImageGenerator._reference_file(path)
        try:
            assert sent.read() == path.read_bytes()
        finally:
            sent.close()
