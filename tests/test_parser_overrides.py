"""Per-entry overrides of header-level post-process switches and engine."""
from __future__ import annotations

from pathlib import Path

from prompts.parser import PromptsFileParser

FILE = """\
---
service: openai
model: gpt-image-2.5-flare
quality: medium
remove_background: true
slice_grid: 4x2
---

=== cutout ===
filename: cutout.png
title: Cutout
description: inherits header
tags: a

A fox.

=== scene ===
filename: scene.png
title: Scene
description: opts out of header flags
tags: b
remove_background: false
slice_grid: none
model: gpt-image-2
quality: high

A tavern.
"""


def _entries(tmp_path: Path):
    root = tmp_path
    coll = root / "prompts" / "c"
    coll.mkdir(parents=True)
    (coll / "x.prompts").write_text(FILE)
    parser = PromptsFileParser(str(root), "prompts", "art/{collection}")
    _, entries = parser.parse_prompts_file(coll / "x.prompts")
    return {e.id: e.to_art_prompt("c") for e in entries}


def test_header_values_inherited_when_entry_silent(tmp_path):
    p = _entries(tmp_path)["cutout"]
    assert p.remove_background == "true"
    assert p.slice_grid == "4x2"
    assert p.model == "gpt-image-2.5-flare"
    assert p.overrides["quality"] == "medium"


def test_entry_overrides_header_flags_and_engine(tmp_path):
    p = _entries(tmp_path)["scene"]
    assert p.remove_background == "false"
    assert p.slice_grid == "none"
    assert p.model == "gpt-image-2"
    assert p.service == "openai"
    assert p.overrides["quality"] == "high"
