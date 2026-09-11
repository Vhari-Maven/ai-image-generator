"""Timestamped backups of outputs about to be overwritten.

Backups live in a `drafts/` directory next to the file being replaced,
so they stay with the collection they belong to and never land in the
consuming project's repo root. Batch tooling ignores `drafts/` because
it only scans a collection directory one level deep.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

BACKUP_SUBDIR = "drafts"


def backup_existing(output_path: Path | str) -> Optional[Path]:
    """Copy `output_path` to `<its dir>/drafts/<stem>_<timestamp><suffix>`.

    Returns the backup path, or None when there is nothing to back up.
    The original stays in place; the caller overwrites it next.
    """
    src = Path(output_path)
    if not src.is_file():
        return None
    backup_dir = src.parent / BACKUP_SUBDIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{src.stem}_{stamp}{src.suffix}"
    n = 1
    while dest.exists():  # two saves in the same second
        dest = backup_dir / f"{src.stem}_{stamp}_{n}{src.suffix}"
        n += 1
    shutil.copy2(src, dest)
    return dest
