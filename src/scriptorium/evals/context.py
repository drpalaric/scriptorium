"""Per-item context for the prompt variants: parent control, retrieved examples, example guidance.

Conditions (prompt version -> input layout), registered in docs/experiments/prompt-v2.md:
    v1   statement only                                       (reference)
    v2   statement, plus the parent's statement for enhancements
    v2p  the PARENT's statement in place of the item's own     (diagnostic; enhancements only)
    v3   statement plus the 8 most similar pool records        (primary)
    v4   as v3, each example also showing guidance             (exploratory)

Everything placed in a prompt is untrusted data: tag-like text is neutralised and nothing is executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scriptorium import ingest

CONDITIONS = {"v1": "none", "v2": "parent", "v2p": "parent_only", "v3": "examples", "v4": "examples_guidance"}
EXAMPLES_K = 8
GUIDANCE_CHARS = 600
_TAGS = ("requirement", "parent_requirement", "examples", "example", "family", "guidance")
_TAG_RE = re.compile(r"<(/?)(%s)" % "|".join(_TAGS), re.IGNORECASE)


class LeakageError(RuntimeError):
    """An evaluated item's base control also appears in the pool or lookup it would draw context from."""


def neutralise(text: str) -> str:
    """Make tag-like sequences from our own vocabulary inert so quoted text cannot close or open a section."""
    return _TAG_RE.sub(lambda m: "&lt;" + m.group(1) + m.group(2), text)


def _block(tag: str, text: str) -> str:
    return f"<{tag}>\n{neutralise(text)}\n</{tag}>"


def assert_disjoint(items: list[dict[str, Any]], pool: list[dict[str, Any]]) -> None:
    shared = {r["base_id"] for r in items} & {r["base_id"] for r in pool}
    if shared:
        raise LeakageError(f"{len(shared)} base control(s) appear in both the evaluated items and the context pool, e.g. {sorted(shared)[:3]}")


class ExamplePool:
    """TF-IDF nearest neighbours over a pool of labelled records. The vectorizer sees only the pool text."""

    def __init__(self, rows: list[dict[str, Any]]):
        if not rows:
            raise ValueError("empty example pool")
        self.rows = rows
        self.vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words="english").fit([r["text"] for r in rows])
        self.mat = self.vec.transform([r["text"] for r in rows])

    def neighbors(self, text: str, k: int = EXAMPLES_K) -> list[dict[str, Any]]:
        """The k most similar records, ordered so the most similar comes last (closest to the question)."""
        sims = (self.vec.transform([text]) @ self.mat.T).toarray()[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.rows[i] for i in order][::-1]


def guidance_by_id() -> dict[str, str]:
    """Guidance text per active control ID, from the pinned raw catalog, control IDs and parameters resolved."""
    path = ingest.RAW_PATH
    if not path.exists():
        raise FileNotFoundError("raw catalog not found; run `python -m scriptorium.ingest fetch` (needed for prompt v4)")
    if ingest.sha256_file(path) != ingest.SOURCE_SHA256:
        raise RuntimeError("raw catalog hash does not match the pinned value")
    cat = json.loads(path.read_text(encoding="utf-8"))["catalog"]
    params: dict[str, str] = {}
    ingest._collect_params(cat, params)
    out: dict[str, str] = {}
    for group in cat["groups"]:
        for c in ingest._walk(group["controls"]):
            if ingest._is_withdrawn(c):
                continue
            parts = [p for p in c.get("parts", []) if p["name"] == "guidance"]
            if not parts:
                continue
            text = re.sub(r"\s+", " ", " ".join(ingest._statement_text(parts[0], params)))
            text = ingest.strip_control_ids(text)
            out[ingest._label(c)] = re.sub(r"\s+([;:,.])", r"\1", text).strip()
    return out


def _example(row: dict[str, Any], guidance: dict[str, str] | None) -> str:
    inner = _block("requirement", row["text"]) + f"\n<family>{row['family']}</family>"
    if guidance is not None:
        g = guidance.get(row["id"], "")[:GUIDANCE_CHARS]
        if g:
            inner += "\n" + _block("guidance", g)
    return f"<example>\n{inner}\n</example>"


def build_user_messages(condition: str, items: list[dict[str, Any]], *, lookup: list[dict[str, Any]] | None = None,
                        pool: list[dict[str, Any]] | None = None, k: int = EXAMPLES_K,
                        guidance: dict[str, str] | None = None) -> list[str]:
    """One user message per item, for the given condition.

    `lookup` is every record of the items' own split (used to find parents); `pool` is the disjoint set that
    examples are drawn from.
    """
    if condition == "none":
        return [_block("requirement", r["text"]) for r in items]
    if condition in ("parent", "parent_only"):
        by_id = {r["id"]: r for r in (lookup if lookup is not None else items)}
        out = []
        for r in items:
            parent = by_id.get(r["base_id"]) if r["is_enhancement"] else None
            if condition == "parent_only":
                if parent is None:
                    raise ValueError(f"{r['id']}: the parent-only diagnostic needs an enhancement with a parent in the split")
                out.append(_block("requirement", parent["text"]))
            elif parent is None:
                out.append(_block("requirement", r["text"]))
            else:
                out.append(_block("parent_requirement", parent["text"]) + "\n" + _block("requirement", r["text"]))
        return out
    if condition in ("examples", "examples_guidance"):
        if pool is None:
            raise ValueError("example conditions need a pool")
        assert_disjoint(items, pool)
        if condition == "examples_guidance" and guidance is None:
            raise ValueError("prompt v4 needs guidance text")
        ex_pool = ExamplePool(pool)
        g = guidance if condition == "examples_guidance" else None
        return ["<examples>\n" + "\n".join(_example(e, g) for e in ex_pool.neighbors(r["text"], k)) + "\n</examples>\n" + _block("requirement", r["text"])
                for r in items]
    raise ValueError(f"unknown condition {condition!r}")


def fingerprint(messages: list[str]) -> str:
    """Short hash of the exact inputs, recorded so a report shows what context was supplied."""
    h = hashlib.sha256()
    for m in messages:
        h.update(m.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
