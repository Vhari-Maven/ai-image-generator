"""Backups land in a sibling drafts/ dir, never at a repo root."""
from __future__ import annotations

from backup import backup_existing


def test_backup_goes_to_sibling_drafts(tmp_path):
    (tmp_path / ".git").mkdir()          # a repo root above the output
    out = tmp_path / "art" / "coll"
    out.mkdir(parents=True)
    img = out / "fox.png"
    img.write_bytes(b"v1")
    dest = backup_existing(img)
    assert dest is not None
    assert dest.parent == out / "drafts"
    assert dest.name.startswith("fox_") and dest.suffix == ".png"
    assert dest.read_bytes() == b"v1"
    assert img.exists()                  # original untouched
    assert not (tmp_path / "assets").exists()


def test_missing_file_returns_none(tmp_path):
    assert backup_existing(tmp_path / "nope.png") is None


def test_same_second_collision_gets_suffix(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    a = backup_existing(img)
    b = backup_existing(img)
    assert a != b and a.exists() and b.exists()


def test_default_config_enables_backups(tmp_path, monkeypatch):
    from config import Config
    (tmp_path / "config.yaml").write_text("{}\n")
    assert Config(str(tmp_path / "config.yaml")).get("output.create_backups") is True
