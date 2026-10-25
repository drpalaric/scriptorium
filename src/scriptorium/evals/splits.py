"""Named data splits for the prompt experiments.

`dev-a` and `dev-b` partition the TRAINING split's base controls 50/50 within each family, grouped by
base control (an enhancement stays with its parent). `test` is the frozen held-out split and must not be
used while developing a prompt (see docs/experiments/prompt-v2.md).
"""

from __future__ import annotations

import random
from typing import Any

from scriptorium import baseline, ingest

DEV_SEED = 20261025
SPLITS = ("test", "train", "dev-a", "dev-b")


def _base_key(base_id: str) -> tuple[int, str]:
    return int(base_id.split("-")[1]), base_id


def dev_bases(train_rows: list[dict[str, Any]], seed: int = DEV_SEED) -> dict[str, list[str]]:
    by_family: dict[str, set[str]] = {}
    for r in train_rows:
        by_family.setdefault(r["family"], set()).add(r["base_id"])
    a: list[str] = []
    b: list[str] = []
    for family in sorted(by_family):
        bases = sorted(by_family[family], key=_base_key)
        random.Random(f"{seed}:dev:{family}").shuffle(bases)
        half = len(bases) // 2
        a.extend(bases[:half])
        b.extend(bases[half:])
    return {"dev-a": sorted(a, key=_base_key), "dev-b": sorted(b, key=_base_key)}


def load_split(name: str) -> list[dict[str, Any]]:
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if name == "test":
        return baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")
    train = baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl")
    if name == "train":
        return train
    keep = set(dev_bases(train)[name])
    return [r for r in train if r["base_id"] in keep]


def pool_for(split: str) -> list[dict[str, Any]]:
    """The records an example-based prompt may draw from for items of `split`. Never overlaps `split`."""
    if split == "dev-b":
        return load_split("dev-a")
    if split == "dev-a":
        return load_split("dev-b")
    if split == "test":
        return load_split("train")
    raise ValueError("example-based prompts cannot be built for the whole training split (no disjoint pool)")
