"""Configuration management for the AI Art Generator."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Mapping: service slug → (env var, secret stem, config.yaml key).
# The stem names both `/run/secrets/<stem>` and `<secrets_enc_dir>/<stem>.gpg`.
_API_KEY_SOURCES = {
    "genai": ("GOOGLE_AI_API_KEY", "google-api-key", "api.google_ai_key"),
    "openai": ("OPENAI_API_KEY", "openai-api-key", "api.openai_key"),
}

# Where gpg-encrypted keys live when no plaintext source is available.
# Override with ART_SECRETS_ENC_DIR or `api.secrets_enc_dir` in config.yaml.
_DEFAULT_SECRETS_ENC_DIR = "~/.secrets-enc"

# Environment variables that override config.yaml (dotted key → env var).
_ENV_OVERRIDES = {
    "paths.output_template": "ART_OUTPUT_TEMPLATE",
    "paths.output_root": "ART_OUTPUT_ROOT",
}

# `paths.output_root` values. Anything else is a folder path.
OUTPUT_ROOT_PROJECT = "project"
OUTPUT_ROOT_MAIN_CHECKOUT = "main-checkout"


def main_checkout(path: Path) -> Optional[Path]:
    """The main working tree of the git repo that holds `path`.

    For a linked worktree (`git worktree add`) this is the checkout the
    worktree was made from; for the main checkout it is that checkout's
    root. None when `path` isn't in a git repo, git is missing, or the
    repo is bare (no main working tree).
    """
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute",
             "--git-common-dir"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    common = Path(out.stdout.strip())
    # A non-bare repo keeps its shared git dir at <main checkout>/.git.
    if common.name != ".git":
        return None
    return common.parent

# In Claude Code cloud sessions the outbound proxy adds the real key to each
# request, so the container never holds it. The SDKs still refuse to start
# without one, so they get this stand-in, which the proxy replaces.
_INJECTED_KEY_PLACEHOLDER = "injected-by-proxy"


def _decrypt_gpg_secret(path: Path) -> Optional[str]:
    """Decrypt `path` with gpg, in memory, never prompting.

    Returns None if gpg is missing, the file is absent, or decryption
    fails (no agent, wrong key, ...). Nothing is written to disk.
    """
    if not path.is_file() or shutil.which("gpg") is None:
        return None
    try:
        out = subprocess.run(
            ["gpg", "--batch", "--quiet", "--decrypt", str(path)],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None



class Config:
    """Loads YAML config and answers dotted-key lookups."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_dir = Path(__file__).parent
        self.config_path = (
            Path(config_path) if config_path else self.config_dir / "config.yaml"
        )
        self.config = self._load_config()
        # The template before any env override: `input_images` paths in
        # .prompts files are written against it.
        self.file_output_template: str = self.get(
            "paths.output_template", "art/{collection}"
        )
        self._apply_env_overrides()
        # Set by `set_roots` once the CLI knows them; until then both are
        # the CWD, which is the CLI's default project root.
        self.project_root: Optional[Path] = None
        self.output_root: Optional[Path] = None

    def resolve_output_root(self, project_root: Path) -> Tuple[Path, str]:
        """Where relative output paths are anchored, and why.

        `paths.output_root` (env ART_OUTPUT_ROOT):
        - `project` (default): the project root, as before.
        - `main-checkout`: the main checkout of the project root's git
          repo, so renders and their `drafts/` backups made from a linked
          worktree outlive it. Falls back to the project root, with a
          note, outside a git repo.
        - any other value: that folder (relative → under the project root).

        Returns (root, label) where label says how it was chosen.
        """
        setting = str(self.get("paths.output_root") or OUTPUT_ROOT_PROJECT)
        source = "ART_OUTPUT_ROOT" if os.getenv("ART_OUTPUT_ROOT") else "paths.output_root"
        if setting == OUTPUT_ROOT_PROJECT:
            return project_root, "project root"
        if setting == OUTPUT_ROOT_MAIN_CHECKOUT:
            main = main_checkout(project_root)
            if main is None:
                return project_root, (
                    f"{source}: {setting} → not a git checkout, "
                    f"using the project root"
                )
            return main, f"{source}: {setting} → {main}"
        root = Path(setting).expanduser()
        if not root.is_absolute():
            root = project_root / root
        return root, f"{source}: {root}"

    def set_roots(self, project_root: Path, output_root: Path) -> None:
        """Record the run's project and output roots for `resolve_reference`."""
        self.project_root = Path(project_root)
        self.output_root = Path(output_root)

    def _apply_env_overrides(self) -> None:
        for key, env_var in _ENV_OVERRIDES.items():
            value = os.getenv(env_var)
            if not value:
                continue
            *parents, leaf = key.split(".")
            node = self.config
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value

    def resolve_reference(self, raw: str) -> Path:
        """Resolve an `input_images` path.

        Absolute paths are used verbatim. A relative one is looked up, in
        order, under:
        1. the project root (so a worktree's own copy of a file wins);
        2. the output root, when `paths.output_root` moved it (so renders
           kept in the main checkout are found from a worktree);
        3. ART_OUTPUT_TEMPLATE's base, when the path starts with the
           config.yaml template's base (`art/` by default), so prompts that
           reference earlier renders keep working after a move.
        When none exists, the project-root path is returned for the error.
        """
        path = Path(raw)
        if path.is_absolute():
            return path
        project_root = self.project_root or Path.cwd()
        output_root = self.output_root or project_root
        local = project_root / path
        if local.exists():
            return local
        if output_root != project_root and (output_root / path).exists():
            return output_root / path
        file_base = self.file_output_template.split("{collection}")[0]
        live_base = self.get("paths.output_template", "").split("{collection}")[0]
        if file_base and live_base != file_base and raw.startswith(file_base):
            moved = Path(live_base) / raw[len(file_base):]
            if not moved.is_absolute():
                moved = output_root / moved
            if moved.exists():
                return moved
        return local

    def _load_config(self) -> Dict[str, Any]:
        defaults = self._get_default_config()
        if not self.config_path.exists():
            print(
                f"Config file not found: {self.config_path}\n"
                f"Using defaults. Copy {self.config_dir}/config.yaml.example "
                f"to {self.config_path} to customize."
            )
            return defaults
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            return self._merge(defaults, user)
        except Exception as e:
            print(f"Error loading config file {self.config_path}: {e}")
            print("Using default configuration.")
            return defaults

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        return {
            "api": {"google_ai_key": None, "openai_key": None},
            "paths": {
                "prompts_dir": "prompts",
                "output_template": "art/{collection}",
                "output_root": OUTPUT_ROOT_PROJECT,
            },
            "generation": {
                "images_per_prompt": 1,
                "default_service": "openai",
                "image_format": "png",
                "optimize_images": True,
                "embed_metadata": True,
            },
            "output": {"create_backups": True},
            "genai": {
                "model": "gemini-3.1-flash-image-preview",
                "defaults": {"aspect_ratio": "1:1"},
                "models": {
                    "gemini-3.1-flash-image-preview": {
                        "max_images": 1,
                        "rpm_limit": 50,
                    },
                    "gemini-3-pro-image-preview": {
                        "max_images": 1,
                        "rpm_limit": 50,
                    },
                    "gemini-2.5-flash-image": {
                        "max_images": 1,
                        "rpm_limit": 50,
                    },
                },
            },
            "openai": {
                "model": "gpt-image-2.5-sunburst",
                "defaults": {
                    "size": "1024x1024",
                    "quality": "auto",
                    "style": "natural",
                },
            },
        }

    def _merge(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        result = default.copy()
        for key, value in user.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a value by dot-notation key, e.g. `genai.defaults.aspect_ratio`."""
        value: Any = self.config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_api_key(self, service: str) -> Optional[str]:
        """Resolve the API key for a service.

        Order: env var → /run/secrets/<stem> (devcontainer bind)
        → gpg-decrypt <secrets_enc_dir>/<stem>.gpg → config.yaml
        → a placeholder in Claude Code cloud sessions, whose proxy
        injects the real key.
        """
        if service not in _API_KEY_SOURCES:
            return None
        env_var, secret_file, config_key = _API_KEY_SOURCES[service]

        env_value = os.getenv(env_var)
        if env_value:
            return env_value

        secret_path = Path("/run/secrets") / secret_file
        if secret_path.exists():
            try:
                value = secret_path.read_text().strip()
                if value:
                    return value
            except OSError:
                pass

        enc_dir = (
            os.getenv("ART_SECRETS_ENC_DIR")
            or self.get("api.secrets_enc_dir")
            or _DEFAULT_SECRETS_ENC_DIR
        )
        value = _decrypt_gpg_secret(
            Path(enc_dir).expanduser() / f"{secret_file}.gpg"
        )
        if value:
            return value

        value = self.get(config_key)
        if value:
            return value

        if os.getenv("CLAUDE_CODE_REMOTE") == "true":
            return _INJECTED_KEY_PLACEHOLDER
        return None


# Singleton. Re-instantiated only when the caller passes an explicit
# config_path different from the cached one — protects scripted callers
# that intentionally swap configs without leaking state across runs.
_config: Optional[Config] = None
_config_path: Optional[str] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Return the global Config instance (cached)."""
    global _config, _config_path
    if _config is None or (config_path and config_path != _config_path):
        _config = Config(config_path)
        _config_path = config_path
    return _config


def main():
    config = get_config()
    print("Configuration loaded:")
    print(f"  Default service: {config.get('generation.default_service')}")
    print(f"  Images per prompt: {config.get('generation.images_per_prompt')}")
    print(f"  Google AI key configured: {'Yes' if config.get_api_key('genai') else 'No'}")
    print(f"  OpenAI key configured: {'Yes' if config.get_api_key('openai') else 'No'}")
    print(f"  GenAI model: {config.get('genai.model')}")
    print(f"  OpenAI model: {config.get('openai.model')}")


if __name__ == "__main__":
    main()
