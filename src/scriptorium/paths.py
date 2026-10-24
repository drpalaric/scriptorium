"""Locations for user-local data (SCF-derived datasets, models). Never inside the repo by default."""

from __future__ import annotations

import os
from pathlib import Path


def scriptorium_home() -> Path:
    """`$SCRIPTORIUM_HOME` if set, else `~/.scriptorium`."""
    return Path(os.environ.get("SCRIPTORIUM_HOME", "~/.scriptorium")).expanduser()


def scf_dir() -> Path:
    return scriptorium_home() / "scf"
