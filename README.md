# ai-art-generator

A CLI for generating AI art collections from `.prompts` files. Wraps
OpenAI's gpt-image family (`gpt-image-2.5-flare` / `-sunburst`, `gpt-image-2`,
`gpt-image-1.5`, `gpt-image-1`) and Google Gemini image models (Nano Banana family)
behind a single interface.

## Quick start

From this repo:

```bash
# install deps (one time; includes the bg-removal group)
uv sync

# verify keys + connectivity (cheap, non-generative probe)
uv run art-generator --test-connection

# generate a collection under a project root (defaults to the current dir)
uv run art-generator --project-root /path/to/your/project --collection my-set

# or, from a consuming project's root
uv run --project /path/to/ai-image-generator art-generator --collection my-set
```

`sandbox/` is a gitignored project root for trying the tool out here.

Project layout the tool expects (configurable in `config.yaml`):

```
<project_root>/
├── prompts/<collection>/
│   └── *.prompts            # .prompts files at any depth under the collection
└── art/<collection>/        # output (auto-created, flat layout)
```

## API keys

Resolution order: env var → `/run/secrets/<stem>` → gpg-decrypt
`~/.secrets-enc/<stem>.gpg` (in memory, needs a gpg-agent) → `config.yaml`.

| Service | Env var | Secret stem |
|---|---|---|
| Google GenAI | `GOOGLE_AI_API_KEY` | `google-api-key` |
| OpenAI | `OPENAI_API_KEY` | `openai-api-key` |

## Models

**Google GenAI** (Gemini `generateContent` API):
- `gemini-3.1-flash-image-preview` — Nano Banana 2 (GenAI default)
- `gemini-3-pro-image-preview` — Nano Banana Pro (highest fidelity)
- `gemini-2.5-flash-image` — Nano Banana (cheapest)

**OpenAI** (default service):
- `gpt-image-2.5-sunburst` — default; most polished output, quality up to `max`, native transparency
- `gpt-image-2.5-flare` — same price and features, faster; good for iteration
- `gpt-image-2` — prior default; no native transparency
- `gpt-image-1.5` / `gpt-image-1` — legacy; native transparency

Native transparent backgrounds are auto-selected when the `.prompts` header
sets `remove_background: true` on a model that supports them — the rembg
postprocess is skipped in that case. On `gpt-image-2` the same flag routes
through the local chroma-key/rembg pipeline instead.

## Commands

```bash
art-generator --collection <slug>                    # generate all in a collection
art-generator --collection <slug> --image-id <id>    # one image (repeatable)
art-generator --prompts-file <path.prompts>          # one .prompts file
art-generator --prompts-dir   <path>                 # one collection directory
art-generator --list-collections                     # discover collections
art-generator --test-connection [--service openai]   # probe API
art-generator --dry-run ...                          # preview, don't generate
```

Per-batch overrides: `--aspect-ratio` (GenAI), `--size` / `--quality` /
`--style` (OpenAI), `--service`, `--model`, `--images-per-prompt`,
`--output-dir`. CLI flags win over `.prompts` per-entry overrides, which
win over `.prompts` file headers, which win over `config.yaml` defaults.

## Configuration

Copy `config.yaml.example` to `config.yaml` and customize. Key settings:

- `paths.prompts_dir` — where collections live (default: `prompts`)
- `paths.output_template` — output base, with `{collection}` placeholder
  (default: `art/{collection}`)
- `genai.model` / `openai.model` — default model per service
- `genai.defaults.aspect_ratio`, `openai.defaults.{size,quality,style}` —
  per-service defaults (used when nothing else supplies a value)
- `postprocess.remove_background.*` — bg-removal tunables (no CLI flags
  by design; see [CLAUDE.md](CLAUDE.md#post-processing))

## See also

- [CLAUDE.md](CLAUDE.md) — full reference for working on or with the tool
- [`prompts/parser.py`](prompts/parser.py) — `.prompts` file format spec
