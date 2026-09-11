"""Concrete post-process stages.

Each stage wraps an underlying implementation module (e.g. `remove_bg`)
behind the `PostprocessStage` protocol. Standalone CLIs call into the
underlying module directly, so stage-specific knobs (model A/B,
diagnostics, custom output paths) live there — the stage wrapper is
only the bridge to the auto-during-generation pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .pipeline import flag_enabled
from .remove_bg import (
    BgRemovalNotInstalled,
    RemoveBgConfig,
    remove_background,
)

if TYPE_CHECKING:
    from prompts.parser import ArtPrompt


def _is_truthy(flag) -> bool:
    return flag_enabled(flag)


def _is_bool_truthy(flag) -> bool:
    """True only for explicit bool-like values, not arbitrary strings."""
    return str(flag).strip().lower() in ("true", "1", "yes", "on")


class RemoveBackgroundStage:
    """Background removal via rembg + chroma-key cleanup.

    Gated by the `.prompts` header field `remove_background`:

    - bool-like truthy (`true`, `yes`, ...) → run with config defaults.
    - bool-like falsey or absent              → skip.
    - any other string                        → run, treating the string
      as a rembg model-name override (e.g. `birefnet-portrait`).
    """

    name = "remove_background"

    def applies(self, prompt: "ArtPrompt") -> bool:
        return _is_truthy(getattr(prompt, "remove_background", None))

    def apply(self, image_path: Path, prompt: "ArtPrompt", config) -> None:
        flag = getattr(prompt, "remove_background", None)
        model_override = (
            None if _is_bool_truthy(flag) else str(flag).strip()
        )

        raw_config = config.get("postprocess.remove_background", {}) or {}
        if model_override:
            raw_config = {**raw_config, "model": model_override}
        bg_config = RemoveBgConfig.from_dict(raw_config)

        try:
            out = remove_background(image_path, bg_config, verbose=True)
        except BgRemovalNotInstalled as e:
            # Re-raise as a known type so the pipeline can log a friendlier
            # warning. (Pipeline catches everything; this just preserves the
            # nicer message.)
            print(f"  ⚠ {e}")
            return

        print(f"  ✓ removed background → {out}")
