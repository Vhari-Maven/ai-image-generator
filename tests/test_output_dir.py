"""Output directory resolution is anchored at the project root for every
source mode (--collection / --prompts-file / --prompts-dir) and every
override (CLI --output-dir, .prompts header output_dir)."""
from __future__ import annotations

from pathlib import Path

from art_generator import _resolve_input_path, resolve_output_dir


def test_template_under_project_root(tmp_path: Path):
    out = resolve_output_dir(
        project_root=tmp_path, output_template="art/{collection}",
        collection_name="joyco",
    )
    assert out == tmp_path / "art" / "joyco"


def test_cli_override_relative_is_under_project_root(tmp_path: Path):
    out = resolve_output_dir(
        project_root=tmp_path, output_template="art/{collection}",
        collection_name="joyco", cli_output_dir="renders/x",
    )
    assert out == tmp_path / "renders" / "x"


def test_cli_override_absolute_is_verbatim(tmp_path: Path):
    out = resolve_output_dir(
        project_root=tmp_path, output_template="art/{collection}",
        collection_name="joyco", cli_output_dir="/abs/dir",
    )
    assert out == Path("/abs/dir")


def test_cli_wins_over_header(tmp_path: Path):
    out = resolve_output_dir(
        project_root=tmp_path, output_template="art/{collection}",
        collection_name="joyco", cli_output_dir="cli", header_output_dir="hdr",
    )
    assert out == tmp_path / "cli"


def test_header_wins_over_template(tmp_path: Path):
    out = resolve_output_dir(
        project_root=tmp_path, output_template="art/{collection}",
        collection_name="joyco", header_output_dir="hdr",
    )
    assert out == tmp_path / "hdr"


def test_input_path_prefers_project_root_then_cwd(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"; (root / "prompts" / "c").mkdir(parents=True)
    (root / "prompts" / "c" / "a.prompts").write_text("")
    # relative path that exists under the project root
    assert _resolve_input_path("prompts/c/a.prompts", root) == root / "prompts" / "c" / "a.prompts"
    # relative path that does not exist under root falls back to CWD-relative
    monkeypatch.chdir(tmp_path)
    assert _resolve_input_path("root/prompts/c/a.prompts", root) == Path("root/prompts/c/a.prompts")
    # absolute verbatim
    assert _resolve_input_path(str(root / "x"), root) == root / "x"
