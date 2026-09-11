"""OpenAI image generator.

Wraps the OpenAI Images API. Defaults to `gpt-image-2.5-flare`; also
supports `gpt-image-2.5-sunburst`, `gpt-image-2`, and `gpt-image-1.5` /
`gpt-image-1`. Native transparent backgrounds on every model except
`gpt-image-2`.
"""
from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .base_generator import BaseGenerator

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import sys
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from prompts.parser import ArtPrompt  # noqa: E402
from config import get_config  # noqa: E402


# Models that accept `background="transparent"` natively on the Images API.
# gpt-image-2 is the odd one out: it never shipped transparency.
TRANSPARENT_CAPABLE_MODELS = (
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-1.5",
    "gpt-image-1",
)

# Quality tiers accepted per model family. gpt-image-2.5 added `xhigh` and
# `max` above the old `high` ceiling (Sept 2026). Anything else is a
# hard error rather than a silent downgrade — the tier drives spend.
_BASE_QUALITY_TIERS = ("auto", "low", "medium", "high")
_GPT_IMAGE_25_QUALITY_TIERS = _BASE_QUALITY_TIERS + ("xhigh", "max")
ALL_QUALITY_TIERS = _GPT_IMAGE_25_QUALITY_TIERS


def quality_tiers_for(model_name: str) -> tuple:
    """Return the quality values the Images API accepts for `model_name`.

    Snapshot ids (e.g. `gpt-image-2.5-flare-2026-09-08`) match by prefix.
    """
    if model_name.startswith("gpt-image-2.5"):
        return _GPT_IMAGE_25_QUALITY_TIERS
    return _BASE_QUALITY_TIERS


def supports_native_transparency(model_name: str) -> bool:
    """True when the model honors `background="transparent"`.

    Prefix-matched so dated snapshots resolve like their alias.
    """
    return any(model_name.startswith(m) for m in TRANSPARENT_CAPABLE_MODELS)


class OpenAIImageGenerator(BaseGenerator):
    """OpenAI image generator (gpt-image family)."""

    service_name = "openai"
    generator_label = "OpenAI"
    # Tier 1 limit: 5 images per minute.
    max_workers = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not installed. Run: uv sync"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        super().__init__(get_config(config_path))
        self.model_name = self.config.get("openai.model", "gpt-image-2.5-flare")

    def generate_image(
        self,
        prompt: ArtPrompt,
        output_dir: str,
        num_images: int = 1,
        *,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        style: Optional[str] = None,
        **_unused,
    ) -> List[str]:
        print(f"Generating {num_images} image(s) for: {prompt.filename}")
        print(f"Prompt: {prompt.prompt[:100]}...")

        defaults = self.config.get("openai.defaults", {}) or {}
        final_size = self._resolve_param(
            "size", size, prompt, defaults.get("size", "1024x1024")
        )
        final_quality = self._resolve_param(
            "quality", quality, prompt, defaults.get("quality", "auto")
        )
        # Style is tracked in metadata but not sent to gpt-image-2.
        final_style = self._resolve_param(
            "style", style, prompt, defaults.get("style", "natural")
        )

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt.prompt,
            "n": num_images,
            "size": final_size,
        }
        allowed = quality_tiers_for(self.model_name)
        if final_quality not in allowed:
            raise ValueError(
                f"quality '{final_quality}' is not supported by "
                f"{self.model_name} (accepts: {', '.join(allowed)})"
            )
        api_params["quality"] = final_quality

        # Native transparent-background path (every model but gpt-image-2).
        # When `remove_background: true` AND the model supports it, request
        # a transparent PNG from the API and skip the rembg postprocess.
        native_transparent = self._maybe_request_native_transparency(
            prompt, api_params
        )

        print(f"  Using size: {final_size}")
        print(f"  Using quality: {final_quality}")

        # Edit / multi-image path: when reference images are provided on the
        # prompt, route through `images.edit` instead of `images.generate`.
        # gpt-image-2 / 1.5 / 1 all support the edit endpoint with one or
        # more inputs.
        input_image_files = self._open_input_images(prompt)
        try:
            if input_image_files:
                api_params["image"] = (
                    input_image_files[0]
                    if len(input_image_files) == 1
                    else input_image_files
                )
                print(
                    f"  Using {len(input_image_files)} reference image(s) "
                    f"→ images.edit"
                )
                response, elapsed = self._timed_call(
                    lambda: self.client.images.edit(**api_params)
                )
            else:
                response, elapsed = self._timed_call(
                    lambda: self.client.images.generate(**api_params)
                )
        finally:
            for fh in input_image_files:
                try:
                    fh.close()
                except Exception:
                    pass
        usage = self._extract_usage(response)
        request_params = {
            "size": final_size,
            "quality": final_quality,
            "style": final_style,
        }

        saved_paths: List[str] = []
        base_filename = Path(prompt.filename).stem

        for idx, image_data in enumerate(response.data):
            image = self._image_from_response(image_data)
            filename = (
                f"{base_filename}_{idx+1:03d}.png"
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

            # Skip rembg when the API already gave us alpha.
            if not native_transparent:
                self._run_postprocess(prompt, save_path)

        return saved_paths

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _service_metadata(self, *, request_params, **_) -> Dict[str, Any]:
        return {
            "size": request_params.get("size"),
            "quality": request_params.get("quality"),
            "style": request_params.get("style"),
        }

    def _extract_usage(self, response: Any) -> Dict[str, Optional[int]]:
        """gpt-image-* reports input/output/total tokens on `response.usage`.

        `input_tokens_details.image_tokens` (present on the edit path, when
        reference images were sent) is surfaced as `image_input_tokens` so
        pricing can bill it at the image-input rate rather than the text
        rate.
        """
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
        details = getattr(usage_obj, "input_tokens_details", None)
        image_input = getattr(details, "image_tokens", None) if details else None
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
        if image_input:
            usage["image_input_tokens"] = image_input
        return usage

    # ------------------------------------------------------------------
    # Service-specific helpers
    # ------------------------------------------------------------------

    def _maybe_request_native_transparency(
        self, prompt: ArtPrompt, api_params: Dict[str, Any]
    ) -> bool:
        flag = getattr(prompt, "remove_background", None)
        if flag is None:
            return False
        is_truthy_bool = str(flag).strip().lower() in ("true", "1", "yes", "on")
        if not is_truthy_bool:
            return False
        if not supports_native_transparency(self.model_name):
            return False
        api_params["background"] = "transparent"
        api_params["output_format"] = "png"
        print(f"  Using native transparent background ({self.model_name})")
        return True

    def _open_input_images(self, prompt: ArtPrompt) -> List[Any]:
        """Open every `prompt.input_images` path as a binary file handle.

        Paths are resolved as written: absolute paths used verbatim, relative
        paths resolved against the current working directory (the tool is
        documented to run from the consuming project's root). Caller closes.
        """
        paths = getattr(prompt, "input_images", None) or []
        opened: List[Any] = []
        for raw in paths:
            p = Path(raw)
            if not p.is_absolute():
                p = Path.cwd() / p
            if not p.exists():
                # Close anything we already opened before raising.
                for fh in opened:
                    try:
                        fh.close()
                    except Exception:
                        pass
                raise FileNotFoundError(
                    f"input_images path not found: {p} "
                    f"(from prompt '{prompt.id}')"
                )
            opened.append(open(p, "rb"))
        return opened

    def _image_from_response(self, image_data: Any) -> Image.Image:
        if image_data.b64_json:
            image_bytes = base64.b64decode(image_data.b64_json)
            return Image.open(BytesIO(image_bytes))
        if image_data.url:
            import requests
            img_response = requests.get(image_data.url)
            img_response.raise_for_status()
            return Image.open(BytesIO(img_response.content))
        raise ValueError("No image data (URL or base64) returned from API")

    @staticmethod
    def test_connection(api_key: Optional[str] = None) -> bool:
        """Probe the API with a cheap, non-generative call (`models.list()`)."""
        try:
            generator = OpenAIImageGenerator(api_key)
            generator.client.models.list()
            print(
                f"✓ OpenAI connection successful "
                f"(model configured: {generator.model_name})"
            )
            return True
        except Exception as e:
            print(f"✗ OpenAI connection failed: {e}")
            return False
