#!/usr/bin/env python3
"""List available models from Google GenAI API.

Useful when poking at what the API exposes for a given key. The
`--image-only` filter that used to live here was unreliable (Google's
`supported_actions` field doesn't consistently flag image models) and
was removed; for the curated list of models the tool actually uses,
see `pricing.py` / `config.yaml`.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai


def list_all_models(api_key: str | None = None) -> None:
    key = api_key or os.getenv("GOOGLE_AI_API_KEY")
    if not key:
        print("Error: GOOGLE_AI_API_KEY not found in environment or .env file",
              file=sys.stderr)
        sys.exit(1)

    try:
        client = genai.Client(api_key=key)
        print("Fetching available models from Google GenAI API...")
        print("=" * 70)

        models = list(client.models.list())
        if not models:
            print("No models found")
            return

        print(f"\nFound {len(models)} models:\n")
        for model in models:
            print(f"Model: {model.name}")
            if hasattr(model, "display_name"):
                print(f"  Display Name: {model.display_name}")
            if hasattr(model, "description"):
                print(f"  Description: {model.description}")
            if hasattr(model, "supported_actions"):
                print(f"  Supported Actions: {', '.join(model.supported_actions)}")
            print()
    except Exception as e:
        print(f"Error listing models: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List available models from Google GenAI API",
    )
    parser.add_argument("--api-key",
                        help="Google AI API key (overrides GOOGLE_AI_API_KEY env var)")
    args = parser.parse_args()
    list_all_models(args.api_key)


if __name__ == "__main__":
    main()
