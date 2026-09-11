"""Google GenAI image generator.

Wraps the Gemini `generateContent` API for the Nano Banana family
(`gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview`,
`gemini-2.5-flash-image`).
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .base_generator import BaseGenerator

try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

import sys
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from prompts.parser import ArtPrompt  # noqa: E402
from config import get_config  # noqa: E402


class GoogleGenAIGenerator(BaseGenerator):
    """Gemini image generator via `generateContent`."""

    service_name = "genai"
    generator_label = "Google GenAI"
    # Within the 50 RPM default quota, ~8 concurrent requests is safe.
    max_workers = 8

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "Google GenAI libraries not installed. Run: uv sync"
            )

        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google AI API key not provided. Set GOOGLE_AI_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        super().__init__(get_config(config_path))
        self.model_name = self.config.get(
            "genai.model", "gemini-3.1-flash-image-preview"
        )

    def generate_image(
        self,
        prompt: ArtPrompt,
        output_dir: str,
        num_images: int = 1,
        *,
        aspect_ratio: Optional[str] = None,
        **_unused,
    ) -> List[str]:
        """Generate `num_images` images. Gemini returns one per call, so loop."""
        print(f"Generating {num_images} image(s) for: {prompt.filename}")
        print(f"Prompt: {prompt.prompt[:100]}...")

        default_aspect = self.config.get("genai.defaults.aspect_ratio", "1:1")
        final_aspect = self._resolve_param(
            "aspect_ratio", aspect_ratio, prompt, default_aspect
        )
        print(f"  Using aspect ratio: {final_aspect}")

        # Multi-image input: prepend reference images to `contents` ahead
        # of the prompt text. Gemini's generate_content accepts PIL Image
        # objects directly. When `input_images` is empty, contents is
        # text-only as before.
        reference_images = self._load_input_images(prompt)
        if reference_images:
            print(
                f"  Using {len(reference_images)} reference image(s) "
                f"in contents"
            )

        safety_settings = self._build_safety_settings()
        saved_paths: List[str] = []

        for i in range(num_images):
            config_params: Dict[str, Any] = {
                "response_modalities": ["Image"],
                "candidate_count": 1,
                "image_config": types.ImageConfig(aspect_ratio=final_aspect),
            }
            if safety_settings:
                config_params["safety_settings"] = safety_settings

            contents: List[Any] = list(reference_images) + [prompt.prompt]
            response, elapsed = self._timed_call(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_params),
                )
            )

            usage = self._extract_usage(response)
            request_params = {"aspect_ratio": final_aspect}

            for part in response.candidates[0].content.parts:
                if part.inline_data is None:
                    continue
                image = Image.open(BytesIO(part.inline_data.data))

                base_filename = Path(prompt.filename).stem
                filename = (
                    f"{base_filename}_{i+1:03d}.png"
                    if num_images > 1
                    else f"{base_filename}.png"
                )
                save_path = os.path.join(output_dir, filename)

                self._save_image_with_metadata(
                    image,
                    save_path,
                    prompt,
                    response=response,
                    request_params=request_params,
                    usage=usage,
                    elapsed_seconds=elapsed,
                )
                saved_paths.append(save_path)
                cost = self._cost_for(usage)
                cost_str = f"${cost:.4f}" if cost is not None else "$?"
                print(f"  ✓ Saved: {save_path} ({elapsed}s, {cost_str})")

                self._run_postprocess(prompt, save_path)

        return saved_paths

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _service_metadata(self, *, request_params, **_) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if request_params.get("aspect_ratio"):
            meta["aspect_ratio"] = request_params["aspect_ratio"]
        return meta

    def _extract_usage(self, response: Any) -> Dict[str, Optional[int]]:
        """Map Gemini's usage_metadata to the canonical triple.

        usage_metadata is None on safety-blocked / failed responses.
        """
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
        return {
            "input_tokens": getattr(meta, "prompt_token_count", None),
            "output_tokens": getattr(meta, "candidates_token_count", None),
            "total_tokens": getattr(meta, "total_token_count", None),
        }

    # ------------------------------------------------------------------
    # Service-specific helpers
    # ------------------------------------------------------------------

    def _load_input_images(self, prompt: ArtPrompt) -> List[Image.Image]:
        """Open every `prompt.input_images` path as a PIL Image.

        Paths resolve as written: absolute used verbatim, relative resolved
        against CWD (the tool is documented to run from project root).
        Returned images are loaded into memory so the caller can close
        the underlying files immediately — Gemini's SDK serializes the
        bytes into the request itself.
        """
        paths = getattr(prompt, "input_images", None) or []
        loaded: List[Image.Image] = []
        for raw in paths:
            p = Path(raw)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists():
                raise FileNotFoundError(
                    f"input_images path not found: {p} "
                    f"(from prompt '{prompt.id}')"
                )
            with Image.open(p) as img:
                img.load()
                loaded.append(img.copy())
        return loaded

    def _build_safety_settings(self):
        """Build SafetySetting list from per-model config, if present."""
        model_config = (
            self.config.config.get("genai", {})
            .get("models", {})
            .get(self.model_name, {})
        )
        safety_config = model_config.get("safety_settings", {})
        if not safety_config:
            return []

        print(f"  Applying safety settings: {safety_config}")

        category_map = {
            "harassment": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            "hate_speech": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "sexually_explicit": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "dangerous_content": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        }
        threshold_map = {
            "BLOCK_NONE": types.HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_ONLY_HIGH": types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "BLOCK_MEDIUM_AND_ABOVE": types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_LOW_AND_ABOVE": types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }

        settings = []
        for category_name, threshold_str in safety_config.items():
            if category_name in category_map and threshold_str in threshold_map:
                settings.append(
                    types.SafetySetting(
                        category=category_map[category_name],
                        threshold=threshold_map[threshold_str],
                    )
                )
        return settings

    @staticmethod
    def test_connection(api_key: Optional[str] = None) -> bool:
        """Probe the API with a cheap, non-generative call.

        Iterates a single page of `models.list()` to confirm the key is
        valid and the network path is open. No tokens billed.
        """
        try:
            generator = GoogleGenAIGenerator(api_key)
            iterator = iter(generator.client.models.list())
            next(iterator, None)
            print("✓ Google GenAI connection successful")
            return True
        except Exception as e:
            print(f"✗ Google GenAI connection failed: {e}")
            return False
