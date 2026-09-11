"""Standalone CLI for the bg-removal post-processor.

Configuration lives entirely in config.yaml under
`postprocess.remove_background.*`. The CLI exposes the input target,
optional output path, A/B model comparison, and a stage-dump diagnostic.

Usage:
    # Single image
    art-generator-remove-bg <image.png>
    art-generator-remove-bg <image.png> -o <output.png>
    art-generator-remove-bg <image.png> --models isnet-anime,birefnet-massive

    # Batch — every PNG in a directory (skips *-cutout*.png siblings)
    art-generator-remove-bg <directory>

    # Batch — every PNG in a collection's output directory
    art-generator-remove-bg --collection <slug>

Batch modes default to overwriting existing `<stem>-cutout.png` files
(`backup: true` in config saves the prior cutout). Pass `--skip-existing`
to skip files where the cutout already exists.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from config import get_config
from postprocess.remove_bg import (
    BgRemovalNotInstalled,
    RemoveBgConfig,
    SamNotInstalled,
    remove_background,
)


def _parse_models(arg: str) -> List[str]:
    """Parse a comma-separated model list, stripping whitespace and dropping empties."""
    return [m.strip() for m in arg.split(",") if m.strip()]


def _model_slug(model: str) -> str:
    """Filesystem-safe slug for a model name.

    `sam3:woman` → `sam3-woman`. Plain rembg names round-trip unchanged.
    Replaces colons (the SAM `sam3:<prompt>` separator) and any other
    awkward characters with `-`.
    """
    out = []
    for ch in model:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def _expected_cutouts(image: Path, models: Optional[List[str]]) -> List[Path]:
    """The cutout path(s) a run over `image` would write.

    Single-model runs write `<stem>-cutout.png`; `--models` runs write
    `<stem>-cutout-<slug>.png` per model. `--skip-existing` skips an
    image only when ALL of these exist — if any is missing, the whole
    image is reprocessed (per-model gap-filling isn't worth the plumbing).
    """
    if models:
        return [
            image.with_name(f"{image.stem}-cutout-{_model_slug(m)}.png")
            for m in models
        ]
    return [image.with_name(f"{image.stem}-cutout.png")]


def _is_source_png(path: Path) -> bool:
    """True for PNGs that are inputs (not our own previously-written cutouts)."""
    if path.suffix.lower() != ".png":
        return False
    # Skip our own outputs: <stem>-cutout.png and -cutout-<model>.png variants.
    return "-cutout" not in path.stem


def _resolve_target(args, config) -> List[Path]:
    """Resolve the CLI's input target (image / directory / collection) to a file list."""
    if args.collection:
        output_template = config.get(
            "paths.output_template", "art/{collection}"
        )
        # Project root: CWD by convention (matches main CLI's default).
        collection_dir = Path.cwd() / output_template.format(
            collection=args.collection
        )
        if not collection_dir.exists():
            print(
                f"error: collection output directory does not exist: "
                f"{collection_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        return sorted(p for p in collection_dir.iterdir() if _is_source_png(p))

    if args.image is None:
        print("error: must pass an image, a directory, or --collection",
              file=sys.stderr)
        sys.exit(1)

    target = args.image
    if target.is_dir():
        return sorted(p for p in target.iterdir() if _is_source_png(p))
    if target.is_file():
        return [target]
    print(f"error: target does not exist: {target}", file=sys.stderr)
    sys.exit(1)


def _process_one(
    image: Path,
    config: RemoveBgConfig,
    *,
    output_path: Optional[Path],
    models: Optional[List[str]],
) -> List[tuple[str, Exception]]:
    """Run bg-removal on one image (possibly across multiple models). Returns failures."""
    failures: List[tuple[str, Exception]] = []
    runs = models or [config.model]

    for model in runs:
        run_config = replace(config, model=model)
        # Multi-model A/B runs always include the model name in the
        # output filename. Single-model invocations keep the legacy
        # <stem>-cutout.png path unless the user passed -o.
        if models:
            out_path: Optional[Path] = image.with_name(
                f"{image.stem}-cutout-{_model_slug(model)}.png"
            )
        else:
            out_path = output_path

        if len(runs) > 1:
            print(f"\n→ model: {model}", file=sys.stderr)

        try:
            out = remove_background(image, run_config, output_path=out_path, verbose=True)
            print(f"wrote {out}", file=sys.stderr)
        except SamNotInstalled as e:
            # Per-model failure, not fatal: a missing SAM venv shouldn't
            # abort the rembg half of an A/B run.
            print(f"error: {e}", file=sys.stderr)
            failures.append((model, e))
        except BgRemovalNotInstalled as e:
            # Missing Python deps — nothing later can succeed.
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            failures.append((model, e))
        except Exception as e:
            print(f"error running {model} on {image.name}: {e}", file=sys.stderr)
            failures.append((model, e))

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the background from one or more images. "
                    "Configuration is read from config.yaml "
                    "(postprocess.remove_background.*); no tuning flags here."
    )
    parser.add_argument(
        "image", type=Path, nargs="?", default=None,
        help="Image file OR directory of images. Omit when using --collection.",
    )
    parser.add_argument(
        "--collection",
        help="Process every PNG in <output_template formatted for this collection>. "
             "Equivalent to passing the collection's output directory as a positional.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: <stem>-cutout.png next to source). "
             "Single-image mode only — cannot combine with batch or --models.",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=None,
        help="Comma-separated model list for A/B comparison. Each entry is "
             "either a rembg model name (e.g. isnet-anime, birefnet-massive) "
             "or `sam3:<prompt>` to shell out to tools/sam (e.g. sam3:woman, "
             "sam3:hair). Each run writes <stem>-cutout-<slug>.png. Overrides "
             "the `model` field in config.yaml.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Batch mode: skip files whose cutout(s) already exist. With "
             "--models, skips only when every per-model cutout exists.",
    )
    parser.add_argument(
        "--dump-stages", type=Path, default=None, metavar="DIR",
        help="Debug: write {stem}-NN-{stage}.png into DIR after each pipeline "
             "stage (chroma matte, decontamination, despill, etc.). Useful for "
             "diagnosing which stage chewed into a figure or left residue.",
    )
    args = parser.parse_args()

    if args.image is None and not args.collection:
        parser.error("must pass an image, a directory, or --collection")
    if args.image is not None and args.collection:
        parser.error("cannot combine a positional target with --collection")

    config = get_config()
    raw_config = config.get("postprocess.remove_background", {})
    bg_config = RemoveBgConfig.from_dict(raw_config)
    if args.dump_stages is not None:
        bg_config = replace(bg_config, dump_stages_dir=args.dump_stages)

    targets = _resolve_target(args, config)
    if not targets:
        print("no source PNGs found", file=sys.stderr)
        sys.exit(1)

    is_batch = len(targets) > 1 or args.collection or (
        args.image is not None and args.image.is_dir()
    )
    if is_batch and args.output is not None:
        parser.error("--output is single-image only (incompatible with batch)")
    if is_batch and args.models:
        # Allowed but loud — the output count is N×M files. Warn but proceed.
        print(
            f"note: --models in batch mode will write "
            f"{len(targets)} × {len(args.models)} files",
            file=sys.stderr,
        )

    if args.skip_existing and not is_batch:
        print("note: --skip-existing has no effect in single-image mode",
              file=sys.stderr)

    all_failures: List[tuple[Path, str, Exception]] = []
    skipped = 0

    for image in targets:
        if is_batch and args.skip_existing:
            if all(p.exists() for p in _expected_cutouts(image, args.models)):
                skipped += 1
                continue
        if is_batch:
            print(f"\n=== {image.name} ===", file=sys.stderr)
        failures = _process_one(
            image, bg_config, output_path=args.output, models=args.models
        )
        for model, exc in failures:
            all_failures.append((image, model, exc))

    if is_batch:
        ran = len(targets) - skipped
        print(
            f"\nProcessed {ran}/{len(targets)} image(s); "
            f"{skipped} skipped, {len(all_failures)} failed.",
            file=sys.stderr,
        )

    if all_failures:
        sys.exit(3)


if __name__ == "__main__":
    main()
