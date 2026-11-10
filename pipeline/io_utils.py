"""Atomic file writes.

Each task writes to a temporary file in the same folder and then renames it
into place with os.replace(), which is atomic on POSIX filesystems. A task that
crashes half-way therefore never leaves a partial file for downstream tasks.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_write(path: Path, write_fn) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def write_csv_atomic(df: pd.DataFrame, path: Path) -> Path:
    return _atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


def write_json_atomic(obj: Any, path: Path) -> Path:
    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)

    return _atomic_write(path, _write)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
