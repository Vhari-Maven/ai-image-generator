"""ART_OUTPUT_TEMPLATE moves renders, and references follow them."""
from __future__ import annotations

import pytest

from config import Config


@pytest.fixture
def make_cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("ART_OUTPUT_TEMPLATE", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    def make() -> Config:
        (tmp_path / "config.yaml").write_text("{}\n")
        return Config(str(tmp_path / "config.yaml"))

    return make


def test_env_overrides_output_template(make_cfg, monkeypatch):
    monkeypatch.setenv("ART_OUTPUT_TEMPLATE", "/shared/art/{collection}")
    cfg = make_cfg()
    assert cfg.get("paths.output_template") == "/shared/art/{collection}"
    assert cfg.file_output_template == "art/{collection}"


def test_no_env_keeps_yaml_template(make_cfg):
    assert make_cfg().get("paths.output_template") == "art/{collection}"


def test_reference_found_under_moved_output(make_cfg, tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    (shared / "art" / "succubus").mkdir(parents=True)
    ref = shared / "art" / "succubus" / "queen.png"
    ref.write_bytes(b"png")
    monkeypatch.setenv("ART_OUTPUT_TEMPLATE", f"{shared}/art/{{collection}}")
    assert make_cfg().resolve_reference("art/succubus/queen.png") == ref


def test_local_reference_wins(make_cfg, tmp_path, monkeypatch):
    local = tmp_path / "project" / "art" / "succubus"
    local.mkdir(parents=True)
    (local / "queen.png").write_bytes(b"png")
    monkeypatch.setenv("ART_OUTPUT_TEMPLATE", f"{tmp_path}/shared/art/{{collection}}")
    found = make_cfg().resolve_reference("art/succubus/queen.png")
    assert found == local / "queen.png"


def test_missing_reference_reports_local_path(make_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ART_OUTPUT_TEMPLATE", f"{tmp_path}/shared/art/{{collection}}")
    found = make_cfg().resolve_reference("art/succubus/missing.png")
    assert found == tmp_path / "project" / "art" / "succubus" / "missing.png"
