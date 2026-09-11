# sandbox — local testing project root

Gitignored scratch project for exercising the tool from this repo. Layout
matches what the CLI expects of any consuming project:

    sandbox/
    ├── prompts/<collection>/*.prompts
    └── art/<collection>/            # outputs land here

Run from the repo root:

    uv run art-generator --project-root sandbox --collection scratch --dry-run
    uv run art-generator --project-root sandbox --collection scratch
    uv run art-generator --project-root sandbox --collection scratch --image-id fox-cutout

API keys resolve automatically (env → /run/secrets → ~/.secrets-enc gpg).
