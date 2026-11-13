"""paths.output_root / ART_OUTPUT_ROOT: renders (and their drafts/
backups) made from a git worktree can land in the main checkout, and
references are found there too."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from art_generator import resolve_output_dir
from config import Config, main_checkout

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A main checkout with one commit, and a linked worktree of it."""
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main)
    (main / "prompts").mkdir()
    (main / "prompts" / "keep.txt").write_text("x")
    _git("add", ".", cwd=main)
    _git("commit", "-qm", "init", cwd=main)
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", str(wt), cwd=main)
    return main.resolve(), wt.resolve()


@pytest.fixture
def make_cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("ART_OUTPUT_TEMPLATE", raising=False)
    monkeypatch.delenv("ART_OUTPUT_ROOT", raising=False)

    def make(yaml: str = "{}\n") -> Config:
        path = tmp_path / "config.yaml"
        path.write_text(yaml)
        return Config(str(path))

    return make


@needs_git
def test_main_checkout_from_worktree_and_main(repo):
    main, wt = repo
    assert main_checkout(wt) == main
    assert main_checkout(main) == main
    assert main_checkout(wt / "prompts") == main


def test_main_checkout_outside_git(tmp_path):
    assert main_checkout(tmp_path) is None


def test_default_is_project_root(make_cfg, tmp_path):
    root, label = make_cfg().resolve_output_root(tmp_path)
    assert root == tmp_path
    assert label == "project root"


@needs_git
def test_main_checkout_setting(make_cfg, repo):
    main, wt = repo
    root, label = make_cfg("paths:\n  output_root: main-checkout\n").resolve_output_root(wt)
    assert root == main
    assert "main-checkout" in label


@needs_git
def test_env_overrides_setting(make_cfg, repo, monkeypatch):
    main, wt = repo
    monkeypatch.setenv("ART_OUTPUT_ROOT", "main-checkout")
    root, label = make_cfg().resolve_output_root(wt)
    assert root == main
    assert label.startswith("ART_OUTPUT_ROOT")


def test_main_checkout_outside_git_falls_back(make_cfg, tmp_path):
    root, label = make_cfg("paths:\n  output_root: main-checkout\n").resolve_output_root(tmp_path)
    assert root == tmp_path
    assert "not a git checkout" in label


def test_explicit_folder(make_cfg, tmp_path):
    cfg = make_cfg(f"paths:\n  output_root: {tmp_path / 'shared'}\n")
    assert cfg.resolve_output_root(tmp_path / "p")[0] == tmp_path / "shared"
    rel = make_cfg("paths:\n  output_root: out\n")
    assert rel.resolve_output_root(tmp_path)[0] == tmp_path / "out"


def test_output_dir_anchored_at_output_root(tmp_path):
    main, wt = tmp_path / "main", tmp_path / "wt"
    kw = dict(project_root=wt, output_template="art/{collection}",
              collection_name="c", output_root=main)
    assert resolve_output_dir(**kw) == main / "art" / "c"
    assert resolve_output_dir(**kw, header_output_dir="hdr") == main / "hdr"
    assert resolve_output_dir(**kw, cli_output_dir="cli") == main / "cli"
    assert resolve_output_dir(**kw, cli_output_dir="/abs") == Path("/abs")


@needs_git
def test_backup_lands_in_main_checkout(make_cfg, repo):
    """drafts/ follow the render, so they're in the main checkout too."""
    from backup import backup_existing

    main, wt = repo
    root, _ = make_cfg("paths:\n  output_root: main-checkout\n").resolve_output_root(wt)
    out = resolve_output_dir(project_root=wt, output_template="art/{collection}",
                             collection_name="c", output_root=root)
    out.mkdir(parents=True)
    (out / "a.png").write_bytes(b"png")
    backup = backup_existing(out / "a.png")
    assert backup.parent == main / "art" / "c" / "drafts"


def test_reference_prefers_project_then_output_root(make_cfg, tmp_path, monkeypatch):
    wt, main = tmp_path / "wt", tmp_path / "main"
    for root in (wt, main):
        (root / "art" / "c").mkdir(parents=True)
    (main / "art" / "c" / "render.png").write_bytes(b"main")
    (main / "art" / "c" / "both.png").write_bytes(b"main")
    (wt / "art" / "c" / "both.png").write_bytes(b"wt")
    monkeypatch.chdir(tmp_path)  # CWD is neither root
    cfg = make_cfg()
    cfg.set_roots(wt, main)
    assert cfg.resolve_reference("art/c/render.png") == main / "art" / "c" / "render.png"
    assert cfg.resolve_reference("art/c/both.png") == wt / "art" / "c" / "both.png"
    assert cfg.resolve_reference("art/c/none.png") == wt / "art" / "c" / "none.png"


def test_reference_uses_project_root_not_cwd(make_cfg, tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "ref").mkdir(parents=True)
    (root / "ref" / "a.png").write_bytes(b"png")
    monkeypatch.chdir(tmp_path)
    cfg = make_cfg()
    cfg.set_roots(root, root)
    assert cfg.resolve_reference("ref/a.png") == root / "ref" / "a.png"
