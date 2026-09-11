"""Post-processing pipeline.

After a generator saves an image, the generator calls
`run_pipeline(image_path, prompt, config)` here. The pipeline walks every
registered stage in declared order, asks each `applies(prompt)`, and runs
`apply()` for the ones that opt in.

A "stage" is one tool-level post-process step (e.g. background removal,
upscale, watermark). Each stage is gated by its own field in the
`.prompts` file header (matching `stage.name`), so adding a new stage is:

  1. Implement the `PostprocessStage` protocol in `postprocess/<name>.py`.
  2. Add a matching `Optional[str]` field on `PromptsFileHeader`.
  3. Register the stage in `postprocess/__init__.py`.

The pipeline driver is for the auto-during-generation path. Standalone
CLIs (e.g. `art-generator-remove-bg`) call into a stage's underlying
implementation directly, since they typically have stage-specific
flags (model A/B comparison, debug dumps, custom output paths) that
don't generalize.

Stage failures are logged and swallowed: post-processing is "nice to
have" on top of a successful generation. One stage failing must not
abort the batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from prompts.parser import ArtPrompt


@runtime_checkable
class PostprocessStage(Protocol):
    """Contract every post-process stage implements.

    `name` matches the `.prompts` file header field that gates the stage,
    e.g. `remove_background`. The pipeline never reads `name` for
    routing — it's only used for log messages — but keeping it aligned
    with the header field avoids confusion.
    """

    name: str

    def applies(self, prompt: "ArtPrompt") -> bool:
        """Return True if this stage should run for `prompt`."""
        ...

    def apply(self, image_path: Path, prompt: "ArtPrompt", config) -> None:
        """Run the stage. May write sibling files next to `image_path`."""
        ...


# Registered stages, in execution order. Order matters when stages
# transform the image (e.g. upscale before bg-removal vs. after); the
# canonical sequence is whatever order stages register here.
_STAGES: List[PostprocessStage] = []


def register(stage: PostprocessStage) -> None:
    """Add a stage to the pipeline. Order of registration = order of execution."""
    if not isinstance(stage, PostprocessStage):
        raise TypeError(
            f"{stage!r} does not satisfy the PostprocessStage protocol "
            f"(needs `name`, `applies()`, `apply()`)"
        )
    _STAGES.append(stage)


def registered_stages() -> List[PostprocessStage]:
    """Return the registered stages, in execution order."""
    return list(_STAGES)


def run_pipeline(image_path: Path, prompt: "ArtPrompt", config) -> None:
    """Walk the pipeline against a freshly generated image.

    Failures are caught per-stage and logged; the batch continues.
    """
    for stage in _STAGES:
        try:
            if not stage.applies(prompt):
                continue
        except Exception as e:
            print(f"  ⚠ {stage.name}.applies() raised: {e}")
            continue

        try:
            stage.apply(image_path, prompt, config)
        except Exception as e:
            print(f"  ⚠ {stage.name} failed: {e}")
