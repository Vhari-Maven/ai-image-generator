"""Configuration management for the AI Art Generator."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

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
        → gpg-decrypt <secrets_enc_dir>/<stem>.gpg → config.yaml.
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

        return self.get(config_key)


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
