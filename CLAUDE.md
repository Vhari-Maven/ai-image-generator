# CLAUDE.md — AI Art Generator

Guidance for Claude Code when working on this tool.

## What it is

A generic CLI for generating AI art collections from `.prompts` files,
wrapping OpenAI's gpt-image family (`gpt-image-2.5-flare` /
`gpt-image-2.5-sunburst`, plus `gpt-image-2`, `gpt-image-1.5`,
`gpt-image-1`) and Google's Gemini image models.
Project-agnostic: paths are configurable, terminology is neutral
(`--collection`, not `--story`/`--artifact`). Currently lives in-tree
under `tools/ai_art_generator/` but designed for clean extraction to
its own repo.

## Supported Models (September 2026)

The default service is **OpenAI** (`generation.default_service`); Google
GenAI is selected per file with `service: genai` or via `--service`.

**Google GenAI** — all images generated through the Gemini API's
`generateContent` endpoint (Nano Banana family):

| Model ID | Nickname | Use case |
|---|---|---|
| `gemini-3.1-flash-image-preview` | Nano Banana 2 | **GenAI default.** Fast + advanced reasoning, strong text rendering |
| `gemini-3-pro-image-preview` | Nano Banana Pro | Highest fidelity, best text rendering, slowest/most expensive |
| `gemini-2.5-flash-image` | Nano Banana | Cheapest, lowest latency, older generation |

Notes:
- Supported aspect ratios (all): `1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 1:4, 4:1, 1:8, 8:1`
- Each call returns one image; the generator loops internally when `--images-per-prompt > 1`
- All outputs include invisible SynthID watermark
- Imagen models and the Vertex AI path were removed when Google deprecated them — do not reintroduce without discussion.

**OpenAI** — gpt-image family via the Images API (`images.generate` /
`images.edit`):

| Model ID | Released | Quality tiers | Native transparency | Use case |
|---|---|---|---|---|
| `gpt-image-2.5-sunburst` | 2026-09-08 | auto, low, medium, high, **xhigh, max** | yes | **Default (service and model).** Most polished output; tighter control across edits. Same token price as flare, ~1.5-2x slower |
| `gpt-image-2.5-flare` | 2026-09-08 | same as sunburst | yes | Same price and features, ~50% lower latency than gpt-image-2. Pick for fast iteration loops |
| `gpt-image-2` | 2026-04 | auto, low, medium, high | **no** | Prior default; keep for reproducing older renders |
| `gpt-image-1.5` | 2025-12 | auto, low, medium, high | yes | Legacy transparent path (pre-2.5) |
| `gpt-image-1` | 2025-04 | auto, low, medium, high | yes | Legacy |

Notes on 2.5:
- Both 2.5 models bill at gpt-image-2's per-token rates (text in $5 /
  image in $8 / image out $30 per 1M). `xhigh` and `max` are new tiers
  above `high`; at 1024×1024 OpenAI's calculator puts them at roughly
  1.8× and 4× the cost of `high`. Passing them to a non-2.5 model is a
  hard error in `openai_image.py` (`quality_tiers_for`), not a downgrade.
- `background: "transparent"` is GA on both (PNG/WebP). `opaque` is a
  new explicit value; the generator still only sends `transparent`.
- Dated snapshots (`gpt-image-2.5-flare-2026-09-08`) are accepted
  anywhere the alias is; capability + pricing lookups prefix-match.
- The Responses API adds `action`, `partial_images` (streaming) and
  `moderation: "low"`; none are wired here yet.

**OpenAI transparent-background path** — every model except `gpt-image-2`:
- Wired up via the existing `remove_background:` flag in the `.prompts` file: when
  the model supports native transparency AND `remove_background: true`,
  the OpenAI generator passes `background="transparent"` + `output_format="png"`
  natively, then **skips** the local rembg postprocess (the PNG is already
  alpha-transparent, no `-cutout.png` is written).
- For `gpt-image-2`, `remove_background: true` still routes through the rembg
  postprocess as before (writes a `-cutout.png` next to the opaque source).
- Non-bool `remove_background:` values (e.g. `remove_background: birefnet-portrait`)
  are explicit rembg-model overrides and always go through the postprocess path,
  even on transparent-capable models. Use this when you want the chroma-key
  pipeline's edge handling over the API's alpha.

Sizes — far more flexible than commonly assumed:

- **Standard presets:** `1024x1024`, `1536x1024`, `1024x1536`, plus `auto`
- **Arbitrary dimensions accepted** within these constraints:
  - Both edges multiples of 16
  - Max single edge 3840px
  - Aspect ratio between 3:1 and 1:3
  - Total pixels between 655,360 and 8,294,400 (≈ 1024×640 to 3840×2160)
  - Anything above ~2560×1440 (3.68M pixels) is marked experimental — higher quality variance
- **Pricing scales with output pixels** — bigger images cost more. At observed
  rates, a 1536×1024 character render at high (or auto-resolves-to-high)
  quality is ~$0.17; a 1024×1024 still-life at auto is ~$0.05–0.10. Pick the
  smallest size that the image actually needs.

Both the `--size` CLI flag and the `.prompts` `size:` field accept any
free-form value the model supports — there is no preset whitelist.

Sources: <https://developers.openai.com/api/docs/models/gpt-image-2.5-flare>,
<https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst>,
<https://developers.openai.com/api/docs/guides/image-generation>,
<https://developers.openai.com/api/docs/pricing>

## Environment Setup

Runs in a `uv`-managed venv. From the tool directory:

```bash
uv sync                          # Install dependencies
uv sync --dev                    # Add dev tools
```

Dependency changes: `uv add <pkg>`, `uv remove <pkg>`, `uv add --dev <pkg>`.

## API Key Configuration

Resolution order in `config.get_api_key()`:

1. **Environment variable** — `GOOGLE_AI_API_KEY` / `OPENAI_API_KEY`
2. **`/run/secrets/`** — `google-api-key` / `openai-api-key` (devcontainer bind mount)
3. **gpg-encrypted file** — `~/.secrets-enc/google-api-key.gpg` /
   `openai-api-key.gpg`, decrypted in memory via `gpg --batch -d` (needs a
   reachable gpg-agent; nothing is written to disk). Directory overridable
   with `ART_SECRETS_ENC_DIR` or `api.secrets_enc_dir` in `config.yaml`.
4. **`config.yaml`** — `api.google_ai_key` / `api.openai_key`

Devcontainers decrypt the same `~/.secrets-enc/` files into podman secrets
mounted at `/run/secrets/` (see the consuming project's
`.devcontainer/secrets.list`). On a host or distrobox with the gpg-agent
socket available, step 3 makes the tool work with no further wiring.

## Project layout (configurable)

The tool reads `paths.prompts_dir` and `paths.output_template` from
`config.yaml`, defaulting to this repo's layout: `prompts/` and
`art/{collection}/`. A `config.yaml` is only needed for
machine-specific overrides.

For each `--collection <slug>`, prompts are read from any `.prompts`
file at any depth under:

```
<project_root>/<paths.prompts_dir>/<slug>/
```

Generated images land flat under:

```
<project_root>/<paths.output_template formatted>/<filename>
```

`<project_root>` defaults to the current working directory; override
with `--project-root`. There is no per-type subdirectory routing —
everything in a collection lands in one directory, named by `filename:`
in each `.prompts` entry.

## Invocation

The tool expects to be run from the consuming project's root directory:

```bash
cd /path/to/your/project
uv run --project /path/to/tools/ai_art_generator art-generator --help
```

Or, when in-tree, from this repo's root:

```bash
cd /path/to/your/project
uv run --project tools/ai_art_generator art-generator --help
```

## Usage Patterns

**Testing connection** (uses a cheap, non-generative `models.list()` probe):

```bash
uv run --project tools/ai_art_generator art-generator --test-connection
uv run --project tools/ai_art_generator art-generator --test-connection --service openai
```

**Single image:**
```bash
uv run --project tools/ai_art_generator art-generator \
  --collection my-set --image-id base

# Higher quality with Nano Banana Pro
uv run --project tools/ai_art_generator art-generator \
  --collection my-set --image-id base --model gemini-3-pro-image-preview

# OpenAI (default gpt-image-2.5-sunburst)
uv run --project tools/ai_art_generator art-generator \
  --collection my-set --image-id base --service openai --quality high

# OpenAI fast iteration
uv run --project tools/ai_art_generator art-generator \
  --collection my-set --image-id base --service openai \
  --model gpt-image-2.5-flare --quality medium
```

**Batch:**
```bash
# Whole collection
uv run --project tools/ai_art_generator art-generator --collection my-set

# Whole collection with a 16:9 aspect override (GenAI)
uv run --project tools/ai_art_generator art-generator \
  --collection my-set --aspect-ratio 16:9
```

**Specific file or directory:**
```bash
uv run --project tools/ai_art_generator art-generator \
  --prompts-file prompts/my-set/base.prompts

uv run --project tools/ai_art_generator art-generator \
  --prompts-dir prompts/my-set/
```

**With reference images (edit / multi-image):**

Pass one or more reference images alongside the prompt to drive an
*edit* / *multi-image* generation rather than a fresh text→image roll.
On OpenAI this routes through `images.edit`; on Gemini the images are
prepended to `contents=[...]` ahead of the prompt text.

Two ways to specify inputs:

1. **In the `.prompts` file** — add an `input_images:` key to the file
   header (file-level default) or to a per-entry block (overrides header).
   Comma-separated list, paths resolved as written (absolute used verbatim,
   relative resolved against CWD — i.e. project root).

   ```
   ---
   service: openai
   model: gpt-image-2
   quality: high
   input_images: web/public/img/joyco-magazine-ad/jx7-hero.png
   ---
   ```

2. **CLI override** — `--input-image PATH` (repeatable). Wins over any
   `input_images:` set in the file. Useful for one-off edits without
   editing the source `.prompts`:

   ```bash
   uv run --project tools/ai_art_generator art-generator \
     --prompts-file prompts/experiments/outfit-swap.prompts \
     --input-image web/public/img/joyco-magazine-ad/jx7-hero.png
   ```

Multiple inputs (multi-image compositing — see §2 of `docs/gpt-image-2.md`
for the "Image 1: ..., Image 2: ..." pattern) are supported by listing
multiple paths in the comma-separated value or by repeating `--input-image`.
Order is positionally significant.

When `input_images` is empty (default), behavior is unchanged: text→image
generation as before.

## Reverse direction: image → .prompts

`prompts-from-image` reads an image's embedded PNG metadata (the same fields
this tool writes on save) and emits a `.prompts` file. Useful when you want
to tweak an existing image's prompt and re-run, or recover a `.prompts`
source that was discarded.

```sh
# print to stdout
uv run --project tools/ai_art_generator prompts-from-image <image.png>

# write to a file
uv run --project tools/ai_art_generator prompts-from-image <image.png> \
  -o prompts/<collection>/characters/<name>.prompts
```

Flags: `-o/--output` to write (errors if exists), `--force` to overwrite,
`--id <slug>` to override the filename-derived entry id.

Carries over verbatim: prompt body, description, model, size, quality,
style, and infers `service` from the `generator` field. Leaves `title`
and `tags` as TODO placeholders for the user to fill in. Errors if the
image has no `prompt` metadata field (i.e. wasn't generated by this tool
or had its metadata stripped). Prints a stderr warning if the embedded
service and model fields look mismatched (e.g. `generator: OpenAI` with
`model: gemini-…` — typically means the metadata was edited).

## Dependencies

Runtime: `google-genai`, `openai`, `Pillow`, `python-dotenv`, `PyYAML`, `tqdm`, `requests`.
Dev: `pytest`, `black`, `ruff`.

## File Structure

```
tools/ai_art_generator/
├── .venv/                   # venv (gitignored)
├── art_generator.py         # CLI entry point
├── prompts_from_image.py    # reverse-direction CLI (image → .prompts)
├── config.py                # Config loader (with /run/secrets/ fallback)
├── config.yaml              # Runtime config (gitignored)
├── config.yaml.example
├── pricing.py               # Per-model token rates → per-image cost lookup
├── list_models.py           # `list_models` — Google API model dump (debugging)
├── pyproject.toml
├── generators/
│   ├── base_generator.py    # Batch threading, save/metadata, post-process hook,
│   │                        # parameter resolution, cost lookup
│   ├── google_genai.py      # Gemini generateContent (Nano Banana family)
│   └── openai_image.py      # OpenAI gpt-image-2.5 / 2 / 1.5 / 1
├── postprocess/
│   ├── __init__.py          # Stage registry — register new stages here
│   ├── pipeline.py          # PostprocessStage protocol + run_pipeline driver
│   ├── stages.py            # Concrete stage wrappers (RemoveBackgroundStage)
│   ├── remove_bg.py         # rembg + chroma-key knockout/spill/dead-pixel/trim
│   └── remove_bg_cli.py     # `art-generator-remove-bg` standalone CLI (single + batch)
└── prompts/
    └── parser.py            # `.prompts` file format spec + parser
```

### Subclass contract for new generators

To add a third provider, subclass `BaseGenerator` and implement:

- `generate_image(prompt, output_dir, num_images, **call_kwargs) → List[str]` —
  do the API call(s), save with `_save_image_with_metadata`, run
  `_run_postprocess`.
- `_extract_usage(response) → {input_tokens, output_tokens, total_tokens}`
  (any value may be `None`).
- `_service_metadata(*, prompt, response, request_params, usage) → dict` —
  return the service-specific PNG metadata fields (e.g. size/quality/style).

Set the class attributes `service_name`, `generator_label`, and `max_workers`.
Everything else (batch threading, parameter resolution, save/backup,
cost lookup, output-dir resolution) is inherited.

## Post-processing

Post-processing runs as a **pipeline of stages**. After every successful
image save, generators call `BaseGenerator._run_postprocess(prompt, save_path)`,
which delegates to `postprocess.run_pipeline(image_path, prompt, config)`.
The pipeline walks every registered stage in declared order, asks each
`applies(prompt)` (reads a `.prompts` switch — per-entry value, else the
file header), and runs
`apply()` for the ones that opt in.

Stages today (registration order = execution order):

1. **`RemoveBackgroundStage`** — gated by `remove_background:`. Runs
   rembg + chroma-key cleanup, writes `<stem>-cutout.png`.
2. **`PixelArtKnockoutStage`** — gated by `pixel_art_knockout:` (value
   = chroma color spec, e.g. `"#FF00FF"` / `magenta` / `255,0,255`).
   Hard-alpha knockout tuned for pixel art, writes `<stem>-cutout.png`.
3. **`SliceStage`** — gated by `slice_grid:` (value = `<cols>x<rows>`
   plus optional space-separated flags: `bbox` / `grid` / `uniform` /
   `padding=N`). Slices the source — or its cutout if a knockout
   stage produced one — into per-cell PNGs at `<stem>-sprites/` next
   to the source, named `<stem>_{idx:02d}.png` in row-major order.
   Examples: `slice_grid: 4x2`, `slice_grid: 4x2 uniform`,
   `slice_grid: 4x2 grid padding=4`.

Every stage switch (`remove_background:`, `pixel_art_knockout:`,
`slice_grid:`), like `service:` / `model:`, may be set in the file header
or in an entry block; **the entry value wins**, so a mixed file can set
`remove_background: true` in the header and `remove_background: false`
on the one opaque scene. Off values are `false`, `no`, `0`, `off`,
`none`, `null`.

Deeper docs for `RemoveBackgroundStage`, gated by `remove_background:`.
Truthy values (`true`, `yes`, …) run with config defaults; any other
string is treated as a rembg model-name override (e.g.
`birefnet-portrait`); off values or omission skip the stage. On success, writes a sibling
`<stem>-cutout.png` next to the source.

**Failures are non-fatal.** A stage's `apply()` raising prints a warning
and the pipeline moves on. The original image is always saved
successfully first.

**Configuration lives in YAML, not the CLI.** All tunables for
`postprocess.remove_background.*` (model, chroma-key color, tolerances,
dead-pixel threshold, trim) come from config.yaml. The `.prompts` header
field is a switch only — there are no CLI flags for the tuning knobs by
design, because the chroma-key and dead-pixel parameters are
interrelated and shouldn't drift between ad-hoc invocations and
pipeline runs.

### Adding a new stage

1. Implement the `PostprocessStage` protocol — `name: str`,
   `applies(prompt) -> bool`, `apply(image_path, prompt, config) -> None`.
   Live in `postprocess/stages.py` (or a separate module if it grows).
2. Add a matching `Optional[str]` field to `PromptsFileHeader` in
   `prompts/parser.py`. The convention is "field name = stage name = the
   `.prompts` header key the user types."
3. Register the stage in `postprocess/__init__.py`. **Order of
   registration = order of execution** — when a stage transforms the
   image (e.g. upscale before bg-removal vs. after), this ordering is
   load-bearing.

The auto-during-generation path picks up the new stage automatically.
Standalone CLIs are stage-specific (see `remove_bg_cli.py`) — wire one
up if the stage has its own knobs (model A/B, debug dumps, custom
output path) that don't fit the pipeline abstraction.

### Standalone CLI

`art-generator-remove-bg` runs the bg-removal stage independently of
generation. Same YAML config, same code path inside `remove_bg.py`.
Three input modes:

```bash
# Single image
art-generator-remove-bg <image.png>
art-generator-remove-bg <image.png> -o <output.png>
art-generator-remove-bg <image.png> --models isnet-anime,birefnet-massive

# Batch — every PNG in a directory (skips *-cutout*.png siblings)
art-generator-remove-bg <directory>

# Batch — every PNG in a collection's output directory
art-generator-remove-bg --collection <slug>
```

Batch defaults to overwriting existing `<stem>-cutout.png` files
(`backup: true` in config saves the prior cutout to a sibling `drafts/`
directory, the same place `output.create_backups` puts prior renders). Pass `--skip-existing`
to skip files that already have a cutout. Useful for tuning chroma-matte
parameters in `config.yaml` and re-processing a whole collection.

**Dependency group.** `rembg` and friends live in the `bg-removal`
dependency group (PEP 735), listed under `[tool.uv.default-groups]` so
in-tree `uv sync` installs them automatically. They are NOT included
when this tool is consumed as a library from another project — keeps
the extraction footprint clean. If a user removes the group manually
and then triggers bg removal, they get a friendly `BgRemovalNotInstalled`
message with the install command.

## Troubleshooting

**"Module not found":** run `uv sync`, and always invoke via `uv run`.

**"API key not found":** verify the resolution chain — env var,
`/run/secrets/<provider>-api-key`, `~/.secrets-enc/<provider>-api-key.gpg`
(try `gpg --batch -q -d` on it by hand; a pinentry prompt means no agent),
or `config.yaml`.

**"Connection refused" / "Connection error":** the devcontainer firewall
allowlist must include `api.openai.com` and `generativelanguage.googleapis.com`
— see [.devcontainer/firewall-extra](../../.devcontainer/firewall-extra)
(applied by the base image's init-firewall.sh on container start).

**Import errors with Google SDK:** use `google-genai` (not `google-generativeai`);
import as `from google import genai`.

## Performance & Threading

- Google GenAI: up to 8 concurrent threads (within the 50 RPM default quota)
- OpenAI gpt-image-*: up to 5 concurrent threads (Tier 1: 5 images/minute, 100K TPM)
- Failures are isolated per prompt — one failure doesn't halt the batch

## Integration Notes

- `--dry-run` previews without generating
- PNG metadata (prompt, description, model, timestamp) is embedded in every saved image
- `test_connection` uses cheap `models.list()` probes — does NOT generate billable images

## Running from Claude Code

Real generation calls take real time — gpt-image-2 at high quality was ~110s
for a single 1536×1024 in observed runs, batches scale roughly linearly within
the per-service thread limit. **Always run actual generation calls with
`run_in_background: true`** and continue the conversation while they finish;
Claude Code will be notified on completion. Foreground-blocking on a
multi-image batch wastes the user's session.

Exceptions — these stay foreground because they're fast and you need the
output to make the next decision:

- `--dry-run` previews
- `--test-connection` probes
- `--list-collections`

Rule of thumb: if the call hits the image API for real, background it.

### When to use `--dry-run`

`--dry-run` is for **verifying the tool itself** after a config change,
parser change, or override-resolution change — confirm prompts are loading,
service/size/quality resolve as expected, output paths are right.

It is NOT a workflow step during normal art-creation iteration. When the
user is iterating on a prompt's content, do not propose a dry-run before
each generation; just run the real generation in the background. Only
reach for dry-run when the tool's plumbing has changed and you want to
verify the new resolution behavior before spending money.

## Extraction-readiness

The tool has no project-specific terminology or hardcoded paths in source.
To extract to its own repo:

1. `git filter-repo --path tools/ai_art_generator/ --path-rename tools/ai_art_generator/:`
2. Add a top-level README, LICENSE, `.gitignore`
3. Consume from digital-haze (or any other project) via `uv add` from path or git URL
4. Delete the in-tree copy
