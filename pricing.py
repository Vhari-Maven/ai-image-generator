"""
Per-image cost lookup for AI art generators.

Rates verified September 2026 (gpt-image-2.5 launch). Pricing drifts — when OpenAI or Google change
rates, update the tables here. Cost embedded in past PNGs is a historical
record of what the image cost at the time and should not be retroactively
updated.

Sources:
- OpenAI token rates: developers.openai.com/api/docs/pricing
- Gemini per-token rates: ai.google.dev/gemini-api/docs/pricing
"""

from typing import Optional


# OpenAI gpt-image family — per-million-token rates (USD).
# response.usage reports input_tokens (prompt text + any reference images)
# and output_tokens (generated image). When the generator can split out
# reference-image tokens (usage.input_tokens_details.image_tokens) they are
# billed at `image_input`; otherwise all input is billed as text.
_OPENAI_TOKEN_RATES_USD_PER_M = {
    "gpt-image-2.5-flare":    {"text_input": 5.00, "image_input": 8.00,  "image_output": 30.00},
    "gpt-image-2.5-sunburst": {"text_input": 5.00, "image_input": 8.00,  "image_output": 30.00},
    "gpt-image-2":            {"text_input": 5.00, "image_input": 8.00,  "image_output": 30.00},
    "gpt-image-1.5":          {"text_input": 5.00, "image_input": 8.00,  "image_output": 32.00},
    "gpt-image-1":            {"text_input": 5.00, "image_input": 10.00, "image_output": 40.00},
}


def _openai_rates(model: str):
    """Exact match first, then alias-prefix so dated snapshots price like
    their alias (e.g. `gpt-image-2.5-flare-2026-09-08`)."""
    if model in _OPENAI_TOKEN_RATES_USD_PER_M:
        return _OPENAI_TOKEN_RATES_USD_PER_M[model]
    for alias in sorted(_OPENAI_TOKEN_RATES_USD_PER_M, key=len, reverse=True):
        if model.startswith(alias + "-"):
            return _OPENAI_TOKEN_RATES_USD_PER_M[alias]
    return None

# Gemini image models — per-million-token rates (USD).
# Image output billed at the image-output rate; text prompt billed at input rate.
_GEMINI_TOKEN_RATES_USD_PER_M = {
    "gemini-3.1-flash-image-preview": {"input": 0.50, "image_output": 60.00},
    "gemini-3-pro-image-preview":     {"input": 2.00, "image_output": 120.00},
    "gemini-2.5-flash-image":         {"input": 0.30, "image_output": 30.00},
}


def get_cost_usd(
    model: str,
    *,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    image_input_tokens: Optional[int] = None,
) -> Optional[float]:
    """
    Look up the cost in USD for a single generated image.

    OpenAI gpt-image-*: priced per token (text input + image output).
    Requires both input_tokens and output_tokens. `image_input_tokens`,
    when given, is the reference-image share of `input_tokens` and is
    billed at the (higher) image-input rate.

    Gemini: priced per token. If both input_tokens and output_tokens are
    given, computes the exact split. If only total_tokens is given, treats
    the whole count as image output (small overestimate, since text input is
    cheaper than image output).

    Returns None when pricing cannot be determined.
    """
    rates = _openai_rates(model)
    if rates is not None:
        if input_tokens is None or output_tokens is None:
            return None
        image_in = min(image_input_tokens or 0, input_tokens)
        text_in = input_tokens - image_in
        cost = (
            text_in * rates["text_input"]
            + image_in * rates["image_input"]
            + output_tokens * rates["image_output"]
        ) / 1_000_000
        return round(cost, 6)

    if model in _GEMINI_TOKEN_RATES_USD_PER_M:
        rates = _GEMINI_TOKEN_RATES_USD_PER_M[model]
        if input_tokens is not None and output_tokens is not None:
            cost = (
                input_tokens * rates["input"]
                + output_tokens * rates["image_output"]
            ) / 1_000_000
            return round(cost, 6)
        if total_tokens is not None:
            return round(total_tokens * rates["image_output"] / 1_000_000, 6)
        return None

    return None
