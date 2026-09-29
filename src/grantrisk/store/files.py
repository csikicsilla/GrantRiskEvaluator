"""The file store under the data root (DEC-22). Stored paths are relative and use '/'."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data_root: Path, relative_path: str, obj: Any) -> str:
    """Write ``obj`` as JSON and return the relative path. Never overwrites a file (ARC-02)."""
    path = data_root / relative_path
    if path.exists():
        raise FileExistsError(f"{relative_path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return relative_path


def write_text(data_root: Path, relative_path: str, text: str, encoding: str = "utf-8") -> str:
    """Write a text file with '\\n' line ends and return the relative path. Never overwrites a file (ARC-02)."""
    path = data_root / relative_path
    if path.exists():
        raise FileExistsError(f"{relative_path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding, newline="\n")
    tmp.replace(path)
    return relative_path


def remove(data_root: Path, relative_path: str) -> None:
    """Remove a file written by a run that then failed."""
    (data_root / relative_path).unlink(missing_ok=True)
