"""Post-processing steps that run after image generation.

The pipeline driver (`pipeline.run_pipeline`) walks every registered
stage in order, asks each `applies(prompt)`, and runs `apply()` for the
ones that opt in. Each stage is gated by its own `.prompts` header field.

To add a new stage:
  1. Implement the `PostprocessStage` protocol (see `pipeline.py`).
  2. Add a matching `Optional[str]` field to `PromptsFileHeader`.
  3. Register the stage below — registration order = execution order.
"""
from .pipeline import PostprocessStage, register, registered_stages, run_pipeline
from .stages import RemoveBackgroundStage
from .pixel_knockout import PixelArtKnockoutStage
from .slice import SliceStage

# Registration order matters: knockout stages run before slicing so the
# cutout PNG exists by the time SliceStage looks for one. RemoveBackground
# and PixelArtKnockout are mutually exclusive in practice (a prompt sets
# one or the other), but their order matters relative to SliceStage.
register(RemoveBackgroundStage())
register(PixelArtKnockoutStage())
register(SliceStage())

__all__ = [
    "PostprocessStage",
    "register",
    "registered_stages",
    "run_pipeline",
    "RemoveBackgroundStage",
    "PixelArtKnockoutStage",
    "SliceStage",
]
