"""Parser for `.prompts` files.

A `.prompts` file is a small text format with two sections:

  1. An optional YAML-ish front matter delimited by `---`, holding
     file-level defaults (service, model, size/quality/aspect_ratio,
     remove_background, output_dir).
  2. One or more `=== <id> ===` entry blocks. Each block has its own
     metadata lines (filename, title, description, tags, optional
     per-entry overrides) followed by a blank line and the prompt text.

This module is the format's spec. There is no JSON form — older JSON
prompt files were converted to `.prompts` and the converter was retired.

Public surface:

  - ``ArtPrompt`` — fully-resolved per-image record consumed by generators.
  - ``PromptsFileHeader`` — parsed front matter (file-level defaults).
  - ``PromptsFileEntry`` — single `=== id ===` block, retains a back-ref
    to its file header so per-entry overrides can resolve against it.
  - ``PromptsFileParser`` — reads files / collections from disk.

The `type:` field on entries (legacy `character` / `background`) is
silently ignored if present — kept tolerated so existing files parse
unchanged, but no longer affects routing or defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Per-entry metadata keys recognized as overrides. Anything else on an
# entry's metadata line (filename/title/description/tags/type) has its own
# slot; unrecognized keys are silently dropped.
PER_ENTRY_OVERRIDE_KEYS = ("size", "aspect_ratio", "quality", "style")

# All keys the entry-block parser treats as metadata. Lines whose key isn't
# in this set are taken to be the start of the prompt body — this lets the
# `.prompts` format keep its conventional visual separation between the
# required block (filename/title/description/tags) and an optional
# overrides block (size/style/...) with a blank line between them, without
# the blank line truncating metadata collection.
KNOWN_ENTRY_METADATA_KEYS = (
    {"filename", "title", "description", "type", "tags", "input_images"}
    | set(PER_ENTRY_OVERRIDE_KEYS)
)


def _parse_image_list(value: Optional[str]) -> List[str]:
    """Split a comma-separated `input_images:` value into a list of paths.

    Empty / None returns an empty list. Whitespace around each entry is
    stripped. Order is preserved (multi-image inputs are positionally
    significant — see "Image 1: ..., Image 2: ..." in prompts).
    """
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


@dataclass
class ArtPrompt:
    """A single, fully-resolved image generation request.

    Built from a `PromptsFileEntry` via `to_art_prompt()`. The generator
    consumes this directly — it has no knowledge of file structure.
    """

    id: str
    filename: str
    title: str
    description: str
    prompt: str
    output_path: str
    tags: List[str] = field(default_factory=list)
    # Per-prompt overrides resolved from (entry > file_header). Keys are
    # a subset of PER_ENTRY_OVERRIDE_KEYS. Sit between CLI flags (highest)
    # and config defaults (lowest); see BaseGenerator._resolve_param.
    overrides: Dict[str, str] = field(default_factory=dict)
    # Post-process flag from the .prompts file header. Truthy bool-like
    # strings gate the bg-removal pass; any other string is treated as a
    # rembg model-name override (e.g. "birefnet-portrait").
    remove_background: Optional[str] = None
    # Pixel-art-friendly knockout flag from the .prompts file header.
    # Value is the chroma color spec to remove (hex / "r,g,b" / named).
    pixel_art_knockout: Optional[str] = None
    # Auto-slice flag for sprite-sheet outputs. Value is "<cols>x<rows>"
    # plus optional space-separated flags (e.g. "4x2 uniform", "4x2 grid",
    # "4x2 uniform padding=4"). When set, postprocess slices the source
    # (or its cutout if a knockout stage produced one) into per-cell PNGs
    # at <stem>-sprites/ next to the source. See postprocess/slice.py.
    slice_grid: Optional[str] = None
    # Reference images to feed into the model alongside the prompt. When
    # non-empty, generators route through their edit / multi-image path
    # (OpenAI `images.edit`; Gemini `generate_content` with PIL images
    # prepended to `contents`). Resolved paths — absolute or relative to
    # the consumer's CWD (the tool is documented to run from project root).
    input_images: List[str] = field(default_factory=list)
    # Service + model this image generates on, resolved from (entry >
    # file header). None means "let the runner decide" (CLI flag, then
    # config default). Carrying these per-prompt is what lets one
    # collection mix engines — e.g. a gpt-image-1.5 transparent variant
    # alongside gpt-image-2 siblings — instead of assuming one service
    # for the whole batch.
    service: Optional[str] = None
    model: Optional[str] = None


@dataclass
class PromptsFileHeader:
    """File-level front matter from a `.prompts` file.

    `output_dir`, `service`, and `model` affect generator instantiation
    and batch-level paths and are read at the CLI/args layer. The
    per-image keys (size/aspect_ratio/quality/style) flow through to
    each entry's `overrides` dict.
    """

    output_dir: Optional[str] = None
    aspect_ratio: Optional[str] = None
    image_size: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None
    safety_filter: Optional[str] = None
    allow_people: Optional[str] = None
    service: Optional[str] = None
    model: Optional[str] = None
    remove_background: Optional[str] = None
    # Pixel-art-friendly chroma knockout: exact-color match, no AA, no
    # trim. Value is the chroma color spec to remove ("#FF00FF",
    # "255,0,255", or a named color). See postprocess/pixel_knockout.py.
    pixel_art_knockout: Optional[str] = None
    # Auto-slice spec ("<cols>x<rows> [flags]"). See ArtPrompt docstring.
    slice_grid: Optional[str] = None
    # File-level default for reference inputs; merged with per-entry
    # `input_images` (entry wins, header is fallback). Stored raw as
    # comma-separated string here; resolved to a list in `to_art_prompt`.
    input_images: Optional[str] = None


@dataclass
class PromptsFileEntry:
    """One `=== id ===` block, before resolution into an ArtPrompt."""

    id: str
    filename: str
    title: str
    description: str
    tags: List[str]
    prompt: str
    source_file: str = ""
    entry_overrides: Dict[str, str] = field(default_factory=dict)
    file_header: Optional[PromptsFileHeader] = None
    # Per-entry reference inputs; comma-separated raw string, overrides
    # the file-level header value when set.
    input_images: Optional[str] = None

    def to_art_prompt(
        self,
        collection_name: str,
        output_template: str = "art/{collection}",
    ) -> ArtPrompt:
        """Resolve into a flat ArtPrompt using `output_template`.

        Output path is always flat under the collection base — there is
        no per-category subdir routing anymore.
        """
        base = output_template.format(collection=collection_name)
        output_path = f"{base}/{self.filename}"

        overrides: Dict[str, str] = {}
        for key in PER_ENTRY_OVERRIDE_KEYS:
            value = self.entry_overrides.get(key)
            if value is None and self.file_header is not None:
                value = getattr(self.file_header, key, None)
            if value is not None:
                overrides[key] = value

        remove_background = (
            self.file_header.remove_background
            if self.file_header is not None
            else None
        )
        pixel_art_knockout = (
            self.file_header.pixel_art_knockout
            if self.file_header is not None
            else None
        )
        slice_grid = (
            self.file_header.slice_grid
            if self.file_header is not None
            else None
        )

        # input_images: entry value wins over file header.
        raw_inputs = self.input_images
        if raw_inputs is None and self.file_header is not None:
            raw_inputs = self.file_header.input_images

        # service / model: entry override wins over file header. Left None
        # when neither declares them — the runner fills in CLI flag then
        # config default.
        service = self.entry_overrides.get("service")
        if service is None and self.file_header is not None:
            service = self.file_header.service
        model = self.entry_overrides.get("model")
        if model is None and self.file_header is not None:
            model = self.file_header.model

        return ArtPrompt(
            id=self.id,
            filename=self.filename,
            title=self.title,
            description=self.description,
            prompt=self.prompt,
            output_path=output_path,
            tags=list(self.tags),
            overrides=overrides,
            remove_background=remove_background,
            pixel_art_knockout=pixel_art_knockout,
            slice_grid=slice_grid,
            input_images=_parse_image_list(raw_inputs),
            service=service,
            model=model,
        )


class PromptsFileParser:
    """Read `.prompts` files and collections from disk."""

    def __init__(
        self,
        project_root: str,
        prompts_dir: str = "prompts",
        output_template: str = "art/{collection}",
    ):
        self.project_root = Path(project_root)
        self.prompts_dir = self.project_root / prompts_dir
        self.output_template = output_template

    # ---------- Parsing ----------

    def parse_header(
        self, content: str
    ) -> Tuple[Optional[PromptsFileHeader], str]:
        """Parse front matter; return (header_or_None, rest_of_content)."""
        if not content.startswith("---\n"):
            return None, content

        end_marker = content.find("\n---\n", 4)
        if end_marker == -1:
            return None, content

        header_content = content[4:end_marker]
        remaining = content[end_marker + 5:]

        header_data: Dict[str, str] = {}
        for line in header_content.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                header_data[key] = value

        # Drop unknown keys so PromptsFileHeader's signature stays the
        # source of truth for what the format accepts.
        known = set(PromptsFileHeader.__dataclass_fields__.keys())
        header_data = {k: v for k, v in header_data.items() if k in known}
        return PromptsFileHeader(**header_data), remaining

    def parse_prompts_file(
        self, filepath: str | Path
    ) -> Tuple[Optional[PromptsFileHeader], List[PromptsFileEntry]]:
        """Parse a single `.prompts` file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts file not found: {filepath}")

        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to read file {filepath}: {e}") from e

        header, content = self.parse_header(content)
        entries: List[PromptsFileEntry] = []

        # Split on `=== ` delimiters; the first chunk before any delimiter
        # is whitespace/comments and gets discarded.
        raw_entries = content.split("\n=== ")
        for i, raw_entry in enumerate(raw_entries):
            if not raw_entry.strip():
                continue
            if i == 0 and raw_entry.startswith("=== "):
                raw_entry = raw_entry[4:]
            elif i == 0:
                # First chunk wasn't an entry — skip leading text.
                continue

            lines = raw_entry.strip().split("\n")
            if not lines[0].endswith(" ==="):
                continue
            prompt_id = lines[0].replace(" ===", "").strip()

            # Walk the entry's metadata lines. Blank lines do NOT terminate —
            # they're allowed between the required block and the optional
            # overrides block. Termination happens when we hit a non-blank
            # line whose key isn't a known metadata key (i.e. the prompt
            # body begins). A line without `:` is also prompt body.
            metadata: Dict[str, str] = {}
            prompt_start = len(lines)
            for line_idx, line in enumerate(lines[1:], 1):
                stripped = line.strip()
                if stripped == "":
                    continue
                if ":" not in stripped:
                    prompt_start = line_idx
                    break
                key = stripped.split(":", 1)[0].strip()
                if key not in KNOWN_ENTRY_METADATA_KEYS:
                    prompt_start = line_idx
                    break
                _, value = line.split(":", 1)
                metadata[key] = value.strip()

            prompt_text = "\n".join(lines[prompt_start:]).strip()

            required = ("filename", "title", "description")
            missing = [f for f in required if not metadata.get(f)]
            if missing:
                print(
                    f"Warning: Missing required fields {missing} in entry "
                    f"'{prompt_id}' ({filepath.name})"
                )
                continue
            if not prompt_text:
                print(
                    f"Warning: Missing prompt text in entry '{prompt_id}' "
                    f"({filepath.name})"
                )
                continue

            tags = [
                t.strip()
                for t in metadata.get("tags", "").split(",")
                if t.strip()
            ]
            entry_overrides = {
                key: metadata[key]
                for key in PER_ENTRY_OVERRIDE_KEYS
                if metadata.get(key)
            }

            entries.append(
                PromptsFileEntry(
                    id=prompt_id,
                    filename=metadata["filename"],
                    title=metadata["title"],
                    description=metadata["description"],
                    tags=tags,
                    prompt=prompt_text,
                    source_file=str(filepath),
                    entry_overrides=entry_overrides,
                    file_header=header,
                    input_images=metadata.get("input_images"),
                )
            )

        return header, entries

    def parse_collection(self, collection_name: str) -> List[PromptsFileEntry]:
        """Parse every `.prompts` file under a collection directory."""
        collection_dir = self.prompts_dir / collection_name
        if not collection_dir.exists():
            raise FileNotFoundError(
                f"Collection directory not found: {collection_dir}"
            )

        all_entries: List[PromptsFileEntry] = []
        for prompts_file in sorted(collection_dir.glob("**/*.prompts")):
            try:
                _, entries = self.parse_prompts_file(prompts_file)
                all_entries.extend(entries)
            except Exception as e:
                print(f"Warning: Failed to parse {prompts_file}: {e}")
        return all_entries

    # ---------- Discovery ----------

    def get_available_collections(self) -> List[str]:
        """List collection slugs that contain at least one `.prompts` file."""
        if not self.prompts_dir.exists():
            return []
        collections: List[str] = []
        for child in sorted(self.prompts_dir.iterdir()):
            if child.is_dir() and any(child.glob("**/*.prompts")):
                collections.append(child.name)
        return collections
