"""
Per-image cost lookup for AI art generators.

Rates verified April 2026. Pricing drifts — when OpenAI or Google change
rates, update the tables here. Cost embedded in past PNGs is a historical
record of what the image cost at the time and should not be retroactively
updated.

Sources:
- OpenAI token rates: developers.openai.com/api/docs/pricing
- Gemini per-token rates: ai.google.dev/gemini-api/docs/pricing
"""

from typing import Optional


# OpenAI gpt-image-2 — per-million-token rates (USD).
# Verified against an actual response: gpt-image-2 reports input_tokens
# (text prompt) and output_tokens (generated image) on response.usage.
_OPENAI_TOKEN_RATES_USD_PER_M = {
    "gpt-image-2": {"text_input": 5.00, "image_output": 30.00},
}

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
) -> Optional[float]:
    """
    Look up the cost in USD for a single generated image.

    OpenAI gpt-image-2: priced per token (text input + image output).
    Requires both input_tokens and output_tokens.

    Gemini: priced per token. If both input_tokens and output_tokens are
    given, computes the exact split. If only total_tokens is given, treats
    the whole count as image output (small overestimate, since text input is
    cheaper than image output).

    Returns None when pricing cannot be determined.
    """
    if model in _OPENAI_TOKEN_RATES_USD_PER_M:
        if input_tokens is None or output_tokens is None:
            return None
        rates = _OPENAI_TOKEN_RATES_USD_PER_M[model]
        cost = (
            input_tokens * rates["text_input"]
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
