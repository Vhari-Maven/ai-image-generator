"""BaseGenerator — shared plumbing for all AI image generators.

Subclass contract
=================

Concrete generators (GoogleGenAIGenerator, OpenAIImageGenerator) extend
this class and implement two methods:

    generate_image(self, prompt, output_dir, num_images, **call_kwargs)
        → List[str]
        Make the per-image API call(s), save with metadata via
        `_save_image_with_metadata()`, run `_run_postprocess()`, return
        the list of saved paths.

    _service_metadata(self, *, prompt, response, request_params, usage)
        → Dict[str, Any]
        Return the service-specific metadata keys to embed in the PNG
        (size/quality/style for OpenAI, safety_settings or anything
        else service-specific). Generic keys (prompt, description,
        generator, model, generation_time, cost, tokens) are filled in
        by the base class.

    _extract_usage(self, response) → Dict[str, Optional[int]]
        Map the SDK's per-response usage object to the canonical
        `{input_tokens, output_tokens, total_tokens}` triple. Any value
        may be None when the API didn't report it.

Subclasses also set:

    service_name: str   — short slug, e.g. "openai" / "genai"
    generator_label: str — display string embedded in PNG metadata
    max_workers: int    — concurrent threads for batch generation

Everything else — batch threading, save-with-backup, parameter
resolution, post-process invocation, cost lookup, output-dir
resolution — is implemented here and shared.
"""
from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, PngImagePlugin

# Local imports use the parent dir on sys.path (set by package __init__).
import sys
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from prompts.parser import ArtPrompt  # noqa: E402
from pricing import get_cost_usd  # noqa: E402
from backup import backup_existing  # noqa: E402


class BaseGenerator:
    """Shared base for all image generators. See module docstring."""

    # Subclass must override.
    service_name: str = ""
    generator_label: str = ""
    max_workers: int = 1

    def __init__(self, config):
        self.config = config
        # Subclass sets self.model_name in its __init__.
        self.model_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Subclass entry points (override in concrete generators)
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: ArtPrompt,
        output_dir: str,
        num_images: int = 1,
        **call_kwargs,
    ) -> List[str]:
        raise NotImplementedError

    def _service_metadata(
        self,
        *,
        prompt: ArtPrompt,
        response: Any,
        request_params: Dict[str, Any],
        usage: Dict[str, Optional[int]],
    ) -> Dict[str, Any]:
        """Return service-specific PNG metadata. Default: nothing extra."""
        return {}

    def _extract_usage(self, response: Any) -> Dict[str, Optional[int]]:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    # ------------------------------------------------------------------
    # Batch — generic across services
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[ArtPrompt],
        base_output_dir: Optional[str],
        images_per_prompt: int = 1,
        **call_kwargs,
    ) -> Dict[str, List[str]]:
        """Generate images for many prompts in a thread pool.

        `call_kwargs` are forwarded to `generate_image` per prompt — these
        are the CLI-flag overrides (size, quality, aspect_ratio, style)
        that win over per-prompt and config defaults. Failures are isolated
        per prompt; one failed prompt does not halt the batch.
        """
        results: Dict[str, List[str]] = {}

        def run_one(prompt: ArtPrompt) -> Tuple[str, List[str]]:
            try:
                output_dir = self._resolve_output_dir(prompt, base_output_dir)
                paths = self.generate_image(
                    prompt, output_dir, images_per_prompt, **call_kwargs
                )
                return prompt.filename, paths
            except Exception as e:
                print(f"✗ Failed: {prompt.filename}: {e}")
                return prompt.filename, []

        workers = max(1, min(len(prompts) or 1, self.max_workers))
        print(
            f"Using {workers} concurrent thread(s) for "
            f"{self.service_name or 'image'} generation"
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, p): p for p in prompts}
            total = len(prompts)
            completed = 0
            for future in as_completed(futures):
                filename, paths = future.result()
                results[filename] = paths
                completed += 1
                status = "✓ Completed" if paths else "✗ Failed"
                print(f"  {status} ({completed}/{total}): {filename}")

        return results

    # ------------------------------------------------------------------
    # Helpers for concrete subclasses
    # ------------------------------------------------------------------

    def _resolve_param(
        self,
        name: str,
        batch_kwarg: Optional[str],
        prompt: ArtPrompt,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a per-image param with the canonical precedence chain.

        Order, highest priority first:
          1. `batch_kwarg` — CLI flag forwarded through `generate_batch`.
          2. `prompt.overrides[name]` — per-entry value (entry > header).
          3. `default` — service/config default the subclass passes in.
        """
        if batch_kwarg is not None:
            return batch_kwarg
        overrides = getattr(prompt, "overrides", {}) or {}
        if overrides.get(name) is not None:
            return overrides[name]
        return default

    def _timed_call(self, fn: Callable[[], Any]) -> Tuple[Any, float]:
        """Run `fn`, return (response, elapsed_seconds_rounded)."""
        start = time.perf_counter()
        response = fn()
        return response, round(time.perf_counter() - start, 2)

    def _cost_for(self, usage: Dict[str, Optional[int]]) -> Optional[float]:
        """Look up per-image cost from `usage` against the current model."""
        if not self.model_name:
            return None
        return get_cost_usd(
            self.model_name,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            image_input_tokens=usage.get("image_input_tokens"),
        )

    def _save_image_with_metadata(
        self,
        image: Any,
        save_path: str,
        prompt: ArtPrompt,
        *,
        response: Any = None,
        request_params: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Optional[int]]] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        """Assemble standard metadata + service extras and save the PNG."""
        usage = usage or {}
        request_params = request_params or {}

        metadata: Dict[str, Any] = {
            "prompt": prompt.prompt,
            "description": prompt.description,
            "generator": self.generator_label or self.service_name,
            "generated_at": datetime.now().isoformat(),
        }
        if self.model_name:
            metadata["model"] = self.model_name
        if elapsed_seconds is not None:
            metadata["generation_time_seconds"] = elapsed_seconds
        cost = self._cost_for(usage)
        if cost is not None:
            metadata["cost_usd"] = cost
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if usage.get(key) is not None:
                metadata[key] = usage[key]

        metadata.update(
            self._service_metadata(
                prompt=prompt,
                response=response,
                request_params=request_params,
                usage=usage,
            )
        )

        self._save_image_with_backup(image, save_path, metadata)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_output_dir(
        self, prompt: ArtPrompt, base_output_dir: Optional[str]
    ) -> str:
        """Where on disk to land an image for `prompt`.

        When `base_output_dir` is set (collection / --output-dir modes),
        images land flat under it. Otherwise we fall back to the parser's
        pre-computed `prompt.output_path` directory (used by
        --prompts-file / --prompts-dir modes).
        """
        if base_output_dir:
            return base_output_dir
        return os.path.dirname(prompt.output_path)

    def _save_image_with_backup(
        self,
        image_data: Any,
        save_path: str,
        metadata_fields: Dict[str, Any],
    ) -> None:
        """Save with optional backup of the prior file at the same path."""
        if (
            os.path.exists(save_path)
            and self.config.get("output.create_backups", True)
        ):
            backup_path = backup_existing(save_path)
            if backup_path:
                print(f"  📁 Backed up existing file to: {backup_path}")

        pil_image = self._convert_to_pil(image_data)

        png_metadata = PngImagePlugin.PngInfo()
        for key, value in metadata_fields.items():
            if value is not None:
                png_metadata.add_text(key, str(value))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil_image.save(save_path, pnginfo=png_metadata, optimize=True)

    def _run_postprocess(self, prompt: ArtPrompt, save_path: str) -> None:
        """Walk the post-process pipeline against a freshly saved image.

        Each registered stage decides whether to run via its `applies()`
        check (typically reads a field on the prompt). Stage failures are
        logged but never abort the batch — postprocess is "nice to have"
        on top of a successful generation.
        """
        try:
            from postprocess import run_pipeline
        except ImportError as e:
            print(f"  ⚠ postprocess pipeline could not be imported: {e}")
            return
        run_pipeline(Path(save_path), prompt, self.config)

    def _convert_to_pil(self, image_data: Any) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data
        if hasattr(image_data, "data"):
            return Image.open(io.BytesIO(image_data.data))
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        return image_data
