#!/usr/bin/env python3
"""AI Art Generator CLI.

Generates images from `.prompts` files. See README.md / CLAUDE.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add this directory to sys.path so flat imports below work whether the
# CLI is launched as a script or via the project's entry point.
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from prompts.parser import ArtPrompt, PromptsFileParser, PromptsFileHeader  # noqa: E402
from generators.google_genai import GoogleGenAIGenerator  # noqa: E402
from generators.openai_image import OpenAIImageGenerator  # noqa: E402
from config import get_config  # noqa: E402


# Aspect ratios accepted by Gemini 3.x image models.
GENAI_ASPECT_RATIOS = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1",
)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def _build_parser(default_service: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AI art collections from .prompts files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything in a collection
  art-generator --collection my-set

  # One specific image by id (repeatable)
  art-generator --collection my-set --image-id mira-domestic-mug

  # Multiple images per prompt
  art-generator --collection my-set --images-per-prompt 3

  # Override aspect ratio (GenAI) for the whole batch
  art-generator --collection my-set --aspect-ratio 16:9

  # Specific .prompts file or directory
  art-generator --prompts-file prompts/my-set/hero.prompts
  art-generator --prompts-dir   prompts/my-set/

  # Pick model / service per invocation
  art-generator --collection my-set --service genai \\
                --model gemini-3-pro-image-preview

  # Connection sanity check (no images generated)
  art-generator --test-connection
  art-generator --test-connection --service openai

  # Dry-run resolution check (no API call)
  art-generator --collection my-set --dry-run --verbose

  # Discover collections
  art-generator --list-collections
        """,
    )

    src = parser.add_argument_group("source (pick one)")
    src.add_argument("--collection", "-c",
                     help="Collection slug under <prompts_dir> (e.g. my-set)")
    src.add_argument("--prompts-file",
                     help="Path to a single .prompts file")
    src.add_argument("--prompts-dir",
                     help="Path to a directory containing .prompts files")

    sel = parser.add_argument_group("selection")
    sel.add_argument("--image-id", action="append",
                     help="Generate only entries with these IDs (repeatable)")
    sel.add_argument("--prompt-id", action="append", dest="image_id",
                     help="Alias for --image-id (repeatable)")

    img = parser.add_argument_group("image options")
    img.add_argument("--images-per-prompt", "-n", type=int, default=1,
                     help="Number of images per prompt (default: 1)")
    img.add_argument("--aspect-ratio", choices=GENAI_ASPECT_RATIOS,
                     help="Override aspect ratio (GenAI only)")
    # Size accepts arbitrary strings: gpt-image-2+ supports a wide range
    # well beyond the legacy 7-preset list. The .prompts header has always
    # been free-form here; the CLI now matches.
    img.add_argument("--size",
                     help="Override image size for OpenAI (e.g. 1024x1024, "
                          "1536x1024, 1280x960). Accepts any value the model "
                          "supports — see CLAUDE.md for gpt-image size limits.")
    img.add_argument("--quality",
                     choices=("auto", "low", "medium", "high", "xhigh", "max"),
                     help="Override image quality (OpenAI only; xhigh/max "
                          "need a gpt-image-2.5 model)")
    img.add_argument("--style", choices=("natural", "vivid"),
                     help="Override style (metadata only; not sent to gpt-image models)")
    img.add_argument("--input-image", action="append", dest="input_image",
                     help="Reference image to feed into the model alongside "
                          "the prompt. Repeatable. When set, overrides any "
                          "input_images: in the .prompts file. Routes "
                          "OpenAI through images.edit; routes Gemini through "
                          "multi-image generate_content.")

    svc = parser.add_argument_group("service")
    # `default=None` lets us tell apart "user passed nothing" from
    # "user explicitly chose default" — file headers can override.
    svc.add_argument("--service", choices=("genai", "openai"), default=None,
                     help=f"AI service to use (default: {default_service})")
    svc.add_argument("--api-key",
                     help="API key for the service (overrides env / secrets)")
    svc.add_argument("--model",
                     help="Override model name "
                          "(genai: gemini-3.1-flash-image-preview / "
                          "gemini-3-pro-image-preview / gemini-2.5-flash-image; "
                          "openai: gpt-image-2.5-flare / gpt-image-2.5-sunburst / "
                          "gpt-image-2 / gpt-image-1.5 / gpt-image-1)")

    util = parser.add_argument_group("utility")
    util.add_argument("--list-collections", action="store_true",
                      help="List available collections and exit")
    util.add_argument("--test-connection", action="store_true",
                      help="Test API connection and exit")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--project-root",
                      help="Project root (defaults to CWD). Prompts and outputs "
                           "are resolved relative to this path.")
    misc.add_argument("--output-dir",
                      help="Override output directory (else uses paths.output_template)")
    misc.add_argument("--dry-run", action="store_true",
                      help="Resolve prompts and print plan; do not call the API")
    misc.add_argument("--verbose", "-v", action="store_true",
                      help="Verbose output")

    return parser


# ----------------------------------------------------------------------
# Source → prompts
# ----------------------------------------------------------------------

def parse_prompts_from_source(
    args, prompts_parser: PromptsFileParser
) -> Tuple[List[ArtPrompt], Optional[PromptsFileHeader]]:
    """Resolve `args` to (prompts, optional file/collection header)."""

    if args.collection:
        collection_dir = prompts_parser.prompts_dir / args.collection
        if not collection_dir.exists() or not any(collection_dir.glob("**/*.prompts")):
            print(f"Error: No .prompts files found for collection '{args.collection}'")
            print(f"Looked in: {collection_dir}/")
            sys.exit(1)

        # Re-parse each file once so we can reach for headers (parse_collection
        # discards them). Cheap — these are small text files.
        all_entries = []
        all_headers: List[PromptsFileHeader] = []
        for prompts_file in sorted(collection_dir.glob("**/*.prompts")):
            try:
                header, entries = prompts_parser.parse_prompts_file(prompts_file)
            except Exception as e:
                print(f"Warning: Failed to parse {prompts_file}: {e}")
                continue
            if header is not None:
                all_headers.append(header)
            all_entries.extend(entries)

        # If every file in the collection agrees on output_dir, surface that
        # as the collection header so the args layer can adopt it. Mixed
        # values are ambiguous and ignored.
        collection_header: Optional[PromptsFileHeader] = None
        output_dirs = {h.output_dir for h in all_headers if h.output_dir}
        if len(output_dirs) == 1:
            collection_header = PromptsFileHeader(output_dir=output_dirs.pop())
        elif len(output_dirs) > 1:
            print(
                f"Warning: .prompts files in collection '{args.collection}' "
                f"disagree on output_dir ({sorted(output_dirs)!r}); "
                f"falling back to default output template"
            )

        prompts = [
            entry.to_art_prompt(args.collection, prompts_parser.output_template)
            for entry in all_entries
        ]
        return prompts, collection_header

    if args.prompts_file:
        file_path = Path(args.prompts_file)
        if not file_path.exists():
            print(f"Error: File '{file_path}' does not exist")
            sys.exit(1)
        if file_path.suffix != ".prompts":
            print(f"Error: Expected a .prompts file, got '{file_path.suffix}'")
            sys.exit(1)

        header, entries = prompts_parser.parse_prompts_file(str(file_path))
        # Collection name is the parent directory's name. (No more
        # category-subdir special casing — the `characters/`/`backgrounds/`
        # convention is gone.)
        collection_name = file_path.parent.name
        prompts = [
            entry.to_art_prompt(collection_name, prompts_parser.output_template)
            for entry in entries
        ]
        return prompts, header

    if args.prompts_dir:
        dir_path = Path(args.prompts_dir)
        if not dir_path.exists():
            print(f"Error: Directory '{dir_path}' does not exist")
            sys.exit(1)

        collection_name = dir_path.name
        # Use a temp parser pointed at this directory's parent so its
        # `parse_collection` finds the right slug.
        temp_parser = PromptsFileParser(
            str(prompts_parser.project_root),
            output_template=prompts_parser.output_template,
        )
        temp_parser.prompts_dir = dir_path.parent
        entries = temp_parser.parse_collection(collection_name)
        prompts = [
            entry.to_art_prompt(collection_name, prompts_parser.output_template)
            for entry in entries
        ]
        return prompts, None

    print("Error: No valid source specified")
    sys.exit(1)


# ----------------------------------------------------------------------
# Utility commands
# ----------------------------------------------------------------------

def list_available_collections(prompts_parser: PromptsFileParser) -> None:
    collections = prompts_parser.get_available_collections()
    if not collections:
        print("No collections found")
        return
    print("Available collections:")
    for collection in collections:
        try:
            entries = prompts_parser.parse_collection(collection)
            print(f"  {collection}  ({len(entries)} prompt(s))")
        except Exception:
            print(f"  {collection}  (could not parse)")


def test_service_connection(service: str, api_key: Optional[str] = None) -> None:
    print(f"Testing {service} connection...")
    if service == "genai":
        ok = GoogleGenAIGenerator.test_connection(api_key)
    elif service == "openai":
        ok = OpenAIImageGenerator.test_connection(api_key)
    else:
        print(f"Service '{service}' not supported")
        return

    if ok:
        print("Connection test passed!")
    else:
        print("Connection test failed!")
        sys.exit(1)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    config = get_config()
    default_service = config.get("generation.default_service", "openai")

    parser = _build_parser(default_service)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    prompts_dir = config.get("paths.prompts_dir", "prompts")
    output_template = config.get(
        "paths.output_template", "art/{collection}"
    )
    prompts_parser = PromptsFileParser(
        str(project_root), prompts_dir, output_template
    )

    # ---- Utility commands (no source needed) ----
    if args.list_collections:
        list_available_collections(prompts_parser)
        return

    if args.test_connection:
        svc = args.service or default_service
        api_key = args.api_key or config.get_api_key(svc)
        test_service_connection(svc, api_key)
        return

    # ---- Validate source ----
    source_count = sum(
        bool(x) for x in (args.collection, args.prompts_file, args.prompts_dir)
    )
    if source_count == 0:
        print("Error: Must specify one of --collection, --prompts-file, --prompts-dir")
        parser.print_help()
        sys.exit(1)
    if source_count > 1:
        print("Error: Cannot combine --collection / --prompts-file / --prompts-dir")
        sys.exit(1)

    # ---- Parse prompts ----
    try:
        prompts, file_header = parse_prompts_from_source(args, prompts_parser)
    except Exception as e:
        print(f"Error parsing prompts: {e}")
        sys.exit(1)

    if not prompts:
        print("No prompts found from the specified source")
        sys.exit(1)

    # ---- Apply file-level header default for output_dir ----
    # Service + model are resolved per-prompt below (each ArtPrompt carries
    # its own, so one collection can mix engines), so only output_dir merges
    # into args here. Per-image keys (size, aspect_ratio, quality, style) ride
    # on each ArtPrompt.overrides and resolve inside the generator.
    if file_header and not args.output_dir and file_header.output_dir:
        args.output_dir = file_header.output_dir
        print(f"Using output directory from file header: {args.output_dir}")

    # ---- CLI override: --input-image wins over .prompts input_images ----
    if args.input_image:
        for p in prompts:
            p.input_images = list(args.input_image)
        print(f"Using CLI input image(s): {', '.join(args.input_image)}")

    # ---- Filter by image-id ----
    if args.image_id:
        wanted = set(args.image_id)
        selected = [p for p in prompts if p.id in wanted]
        if not selected:
            print(f"Error: No images found with IDs '{', '.join(wanted)}'")
            print("Available image IDs:")
            for p in prompts:
                print(f"  {p.id}: {p.title}")
            sys.exit(1)
        missing = wanted - {p.id for p in selected}
        if missing:
            print(f"Warning: Image IDs not found: {', '.join(sorted(missing))}")
        prompts = selected

    # ---- Resolve each prompt's service + model ----
    # Precedence: CLI flag (--service / --model) > the prompt's own .prompts
    # header > config default. A service/model declared in a .prompts file is
    # NEVER silently swapped for the config default — the default only fills
    # in prompts that declare nothing. (Before this, collection mode dropped
    # the per-file service/model entirely and everything fell to the default.)
    for p in prompts:
        p.service = args.service or p.service or default_service
        p.model = args.model or p.model  # None → generator's own config default

    # ---- Show plan ----
    if args.collection:
        print(f"Collection: {args.collection}")
    elif args.prompts_file:
        print(f"Prompts file: {args.prompts_file}")
    elif args.prompts_dir:
        print(f"Prompts directory: {args.prompts_dir}")
    services_in_play = sorted({p.service for p in prompts})
    print(f"Service(s): {', '.join(services_in_play)}")
    if args.image_id:
        print(f"Specific images: {', '.join(args.image_id)}")
    print(f"Images per prompt: {args.images_per_prompt}")
    if args.aspect_ratio:
        print(f"Aspect ratio override (GenAI): {args.aspect_ratio}")
    if args.size:
        print(f"Size override (OpenAI): {args.size}")
    if args.quality:
        print(f"Quality override (OpenAI): {args.quality}")
    if args.style:
        print(f"Style override (OpenAI, metadata only): {args.style}")
    print(f"Total prompts: {len(prompts)}")

    if args.verbose:
        print("\nPrompts to generate:")
        for p in prompts:
            engine = p.service + (f" / {p.model}" if p.model else "")
            print(f"  {p.id} → {p.filename} [{engine}]: {p.description}")
            if p.input_images:
                print(f"    input_images: {', '.join(p.input_images)}")

    if args.dry_run:
        print("\nDry run — no images will be generated")
        return

    # ---- Resolve base output dir ---- (service-independent)
    if args.output_dir:
        base_output_dir: Optional[str] = args.output_dir
    elif args.collection:
        base_output_dir = str(
            project_root / output_template.format(collection=args.collection)
        )
    else:
        # --prompts-file / --prompts-dir: each prompt carries its own
        # output_path; the generator falls back to it when base is None.
        base_output_dir = None

    if base_output_dir:
        print(f"\nGenerating images to: {base_output_dir}")
    else:
        print("\nGenerating images (paths from .prompts file output_path)")
    print("=" * 50)

    # ---- Generate, grouped by (service, model) ----
    # One generator instance per distinct engine. Grouping is what lets a
    # single run honor a mix of declared engines (e.g. a gpt-image-1.5
    # transparent variant alongside its gpt-image-2 siblings). A group that
    # fails to init or generate is isolated — it doesn't sink the others.
    groups: Dict[Tuple[str, Optional[str]], List[ArtPrompt]] = {}
    for p in prompts:
        groups.setdefault((p.service, p.model), []).append(p)

    results: Dict[str, List[str]] = {}
    for (service, model), group_prompts in groups.items():
        label = service + (f" / {model}" if model else "")
        print(f"\n→ {len(group_prompts)} image(s) on {label}")

        try:
            api_key = args.api_key or config.get_api_key(service)
            if service == "genai":
                generator = GoogleGenAIGenerator(api_key)
            elif service == "openai":
                generator = OpenAIImageGenerator(api_key)
            else:
                print(f"  Error: service '{service}' not supported — "
                      f"skipping {len(group_prompts)} image(s)")
                continue
            if model:
                generator.model_name = model
        except Exception as e:
            print(f"  Error initializing {label} generator: {e} — "
                  f"skipping {len(group_prompts)} image(s)")
            continue

        call_kwargs = {}
        if service == "genai":
            if args.aspect_ratio:
                call_kwargs["aspect_ratio"] = args.aspect_ratio
        elif service == "openai":
            if args.size:
                call_kwargs["size"] = args.size
            if args.quality:
                call_kwargs["quality"] = args.quality
            if args.style:
                call_kwargs["style"] = args.style

        try:
            group_results = generator.generate_batch(
                group_prompts, base_output_dir, args.images_per_prompt,
                **call_kwargs,
            )
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nError during generation on {label}: {e}")
            continue
        results.update(group_results)

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("Generation Summary:")
    total_generated = 0
    failed_count = 0
    for filename, paths in results.items():
        if paths:
            print(f"  ✓ {filename}: {len(paths)} image(s)")
            total_generated += len(paths)
        else:
            print(f"  ✗ {filename}: Failed")
            failed_count += 1
    print(f"\nTotal generated: {total_generated}")
    if failed_count:
        print(f"Failed: {failed_count}")
    print("Done!")


if __name__ == "__main__":
    main()
