"""Loading the configuration (ARC-03).

One YAML file holds every path, threshold and parameter. Relative paths are
resolved against the repository root. Secrets are read from environment
variables, never from this file (ARC-08).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Config:
    values: dict[str, Any]
    path: Path

    @property
    def data_root(self) -> Path:
        return self.resolve(self.values["data_root"])

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    def source(self, name: str) -> Path:
        """Path of a read-only input of the old code line (``sources`` section)."""
        return self.resolve(self.values["sources"][name])


def load(path: str | os.PathLike[str] | None = None) -> Config:
    """Load the configuration from ``path``, ``$GRANTRISK_CONFIG`` or the default file."""
    path = Path(path or os.environ.get("GRANTRISK_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if "data_root" not in values:
        raise ValueError(f"{path}: 'data_root' is missing")
    return Config(values=values, path=path)
