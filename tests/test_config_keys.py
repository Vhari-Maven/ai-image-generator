"""API-key resolution chain, including the gpg-encrypted fallback."""
from __future__ import annotations

import subprocess

import pytest

import config as config_mod
from config import Config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for v in ("OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "ART_SECRETS_ENC_DIR"):
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "config.yaml").write_text("api: {}\n")
    return Config(str(tmp_path / "config.yaml"))


def test_env_wins(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert cfg.get_api_key("openai") == "from-env"


def test_gpg_fallback_used_when_no_env_or_run_secrets(cfg, tmp_path, monkeypatch):
    enc = tmp_path / "enc"
    enc.mkdir()
    (enc / "openai-api-key.gpg").write_bytes(b"ciphertext")
    monkeypatch.setenv("ART_SECRETS_ENC_DIR", str(enc))
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="sk-decrypted\n", stderr="")

    monkeypatch.setattr(config_mod.shutil, "which", lambda _: "/usr/bin/gpg")
    monkeypatch.setattr(config_mod.subprocess, "run", fake_run)
    assert cfg.get_api_key("openai") == "sk-decrypted"
    assert calls and calls[0][:4] == ["gpg", "--batch", "--quiet", "--decrypt"]
    assert calls[0][-1].endswith("openai-api-key.gpg")


def test_gpg_failure_falls_through_to_config(cfg, tmp_path, monkeypatch):
    enc = tmp_path / "enc"
    enc.mkdir()
    (enc / "openai-api-key.gpg").write_bytes(b"ciphertext")
    monkeypatch.setenv("ART_SECRETS_ENC_DIR", str(enc))
    monkeypatch.setattr(config_mod.shutil, "which", lambda _: "/usr/bin/gpg")
    monkeypatch.setattr(
        config_mod.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 2, stdout="", stderr="no agent"),
    )
    cfg.config["api"] = {"openai_key": "from-yaml"}
    assert cfg.get_api_key("openai") == "from-yaml"


def test_missing_gpg_file_is_silent(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ART_SECRETS_ENC_DIR", str(tmp_path / "nope"))
    assert cfg.get_api_key("openai") is None


def test_default_service_is_openai(tmp_path):
    (tmp_path / "config.yaml").write_text("{}\n")
    assert Config(str(tmp_path / "config.yaml")).get("generation.default_service") == "openai"
