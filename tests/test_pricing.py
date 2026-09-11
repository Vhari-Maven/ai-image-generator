"""Unit tests for per-image cost lookup."""
from __future__ import annotations

import pytest

from pricing import get_cost_usd


class TestOpenAI:
    def test_flare_text_plus_image_output(self):
        # 100 text tokens @ $5/M + 4160 image tokens @ $30/M
        cost = get_cost_usd("gpt-image-2.5-flare", input_tokens=100, output_tokens=4160)
        assert cost == pytest.approx(0.0005 + 0.1248)

    def test_sunburst_priced_same_as_flare(self):
        a = get_cost_usd("gpt-image-2.5-flare", input_tokens=50, output_tokens=1000)
        b = get_cost_usd("gpt-image-2.5-sunburst", input_tokens=50, output_tokens=1000)
        assert a == b

    def test_dated_snapshot_prices_like_alias(self):
        a = get_cost_usd("gpt-image-2.5-flare", input_tokens=50, output_tokens=1000)
        b = get_cost_usd("gpt-image-2.5-flare-2026-09-08", input_tokens=50, output_tokens=1000)
        assert a == b

    def test_image_input_billed_at_image_rate(self):
        # 1000 input tokens, 600 of them reference-image → 400 text.
        cost = get_cost_usd(
            "gpt-image-2.5-flare",
            input_tokens=1000, output_tokens=0, image_input_tokens=600,
        )
        assert cost == pytest.approx((400 * 5 + 600 * 8) / 1e6)

    def test_image_input_clamped_to_input_total(self):
        cost = get_cost_usd(
            "gpt-image-2", input_tokens=100, output_tokens=0, image_input_tokens=500,
        )
        assert cost == pytest.approx(100 * 8 / 1e6)

    def test_legacy_models_have_distinct_output_rates(self):
        assert get_cost_usd("gpt-image-1.5", input_tokens=0, output_tokens=1_000_000) == 32.0
        assert get_cost_usd("gpt-image-1", input_tokens=0, output_tokens=1_000_000) == 40.0

    def test_missing_usage_returns_none(self):
        assert get_cost_usd("gpt-image-2.5-flare", input_tokens=None, output_tokens=10) is None

    def test_unknown_model_returns_none(self):
        assert get_cost_usd("gpt-image-9", input_tokens=1, output_tokens=1) is None
