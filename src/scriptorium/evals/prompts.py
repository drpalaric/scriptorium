"""Versioned prompt files under `prompts/<task>/<version>.txt`."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from scriptorium import ingest

PROMPTS_DIR = ingest.ROOT / "prompts"
_NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


@dataclass(frozen=True)
class Prompt:
    task: str
    version: str
    template: str
    sha256: str

    def render(self, **values: str) -> str:
        return self.template.format(**values)


def load_prompt(task: str, version: str, root: Path | None = None) -> Prompt:
    """Load a prompt by task and version. Names are restricted so they cannot name a path."""
    if not _NAME_RE.match(task) or not _NAME_RE.match(version):
        raise ValueError("task and prompt version must be 1-32 characters: lowercase letters, digits, dash, underscore")
    path = (root or PROMPTS_DIR) / task / f"{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt {task}/{version} (looked for {path.name} under prompts/{task}/)")
    raw = path.read_bytes()
    return Prompt(task, version, raw.decode("utf-8"), hashlib.sha256(raw).hexdigest())
