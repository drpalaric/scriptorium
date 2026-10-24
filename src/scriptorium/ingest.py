"""Ingest the NIST SP 800-53 rev5 OSCAL catalog into a labelled, split dataset.

Each active control or enhancement becomes one record: requirement text (control
IDs removed) labelled with its control family. The train/test split is grouped by
base control, so `AC-2` and `AC-2(1)` always land on the same side.

    python -m scriptorium.ingest fetch   # download the pinned catalog to data/raw/
    python -m scriptorium.ingest build   # write data/processed/* and data/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_REPO = "https://github.com/usnistgov/oscal-content"
SOURCE_COMMIT = "f0de6d8d8c059d374f15e0ae07508d82e370fc1d"
SOURCE_PATH = "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
SOURCE_SHA256 = "64ca6b4161de51ab6af098a7d101d326f14c95e1abf721396cd8a049e8115eeb"
SOURCE_URL = f"https://raw.githubusercontent.com/usnistgov/oscal-content/{SOURCE_COMMIT}/{SOURCE_PATH}"

SEED = 20260424
TEST_FRACTION = 0.2

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / Path(SOURCE_PATH).name
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

FAMILY_CODES = ("AC AT AU CA CM CP IA IR MA MP PE PL PM PS PT RA SA SC SI SR").split()
_ID_RE = re.compile(
    r"\b(?:%s)-\d{1,2}(?:\(\d{1,2}\))?(?:\([a-z]\))?" % "|".join(FAMILY_CODES), re.IGNORECASE
)
_LINK_TARGET_RE = re.compile(r"\]\(#[^)]*\)")
_PARAM_RE = re.compile(r"\{\{\s*insert:\s*param,\s*([^\s}]+)\s*\}\}")
_ENHANCEMENT_ID_RE = re.compile(r"^([A-Z]{2}-\d+)\(\d+\)$")
CONTROL_REF = "[control]"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dest: Path = RAW_PATH) -> Path:
    """Download the pinned catalog and verify its hash."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:  # noqa: S310 - pinned https URL
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"catalog hash mismatch: got {digest}, expected {SOURCE_SHA256}")
    dest.write_bytes(data)
    return dest


def _collect_params(node: Any, out: dict[str, str]) -> None:
    """Map every param id in the catalog to the text used in place of its placeholder."""
    if isinstance(node, dict):
        for p in node.get("params", []):
            if "label" in p:
                out[p["id"]] = p["label"]
            elif "select" in p:
                choices = p["select"].get("choice", [])
                out[p["id"]] = "[" + " | ".join(choices) + "]"
        for v in node.values():
            _collect_params(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_params(v, out)


def strip_control_ids(text: str) -> str:
    text = _LINK_TARGET_RE.sub("]", text)
    text = _ID_RE.sub(CONTROL_REF, text)
    return text


def _statement_text(part: dict[str, Any], params: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if "prose" in part:
        prose = _PARAM_RE.sub(lambda m: params[m.group(1)], part["prose"])
        lines.append(prose.strip())
    for sub in part.get("parts", []):
        lines.extend(_statement_text(sub, params))
    return lines


def _label(control: dict[str, Any]) -> str:
    for p in control.get("props", []):
        if p["name"] == "label" and "class" not in p:
            return p["value"]
    raise ValueError(f"control {control['id']} has no plain label")


def _is_withdrawn(control: dict[str, Any]) -> bool:
    return any(p["name"] == "status" and p["value"] == "withdrawn" for p in control.get("props", []))


def _walk(controls: list[dict[str, Any]]):
    for c in controls:
        yield c
        yield from _walk(c.get("controls", []))


def parse_catalog(catalog: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (records, stats). Withdrawn controls are dropped and counted."""
    cat = catalog["catalog"]
    params: dict[str, str] = {}
    _collect_params(cat, params)

    records: list[dict[str, Any]] = []
    withdrawn = 0
    for group in cat["groups"]:
        family = group["id"].upper()
        for c in _walk(group["controls"]):
            if _is_withdrawn(c):
                withdrawn += 1
                continue
            label = _label(c)
            m = _ENHANCEMENT_ID_RE.match(label)
            base_id = m.group(1) if m else label
            statement = [p for p in c.get("parts", []) if p["name"] == "statement"]
            if not statement:
                raise ValueError(f"active control {label} has no statement")
            text = " ".join(_statement_text(statement[0], params))
            text = strip_control_ids(re.sub(r"\s+", " ", text))
            text = re.sub(r"\s+([;:,.])", r"\1", text).strip()
            if not text:
                raise ValueError(f"active control {label} has empty statement text")
            if not base_id.startswith(family + "-"):
                raise ValueError(f"{label} is not in group {family}")
            records.append(
                {
                    "id": label,
                    "base_id": base_id,
                    "family": family,
                    "family_title": group["title"],
                    "title": c["title"],
                    "is_enhancement": bool(m),
                    "text": text,
                }
            )
    return records, {"withdrawn_dropped": withdrawn, "oscal_version": cat["metadata"].get("oscal-version"),
                     "catalog_version": cat["metadata"].get("version")}


def split_bases(records: list[dict[str, Any]], seed: int = SEED, test_fraction: float = TEST_FRACTION) -> dict[str, list[str]]:
    """Stratified-by-family split over base controls. Returns {"train": [...], "test": [...]}."""
    by_family: dict[str, set[str]] = defaultdict(set)
    for r in records:
        by_family[r["family"]].add(r["base_id"])
    train: list[str] = []
    test: list[str] = []
    for family in sorted(by_family):
        bases = sorted(by_family[family], key=lambda b: (int(b.split("-")[1]), b))
        random.Random(f"{seed}:{family}").shuffle(bases)
        n_test = max(1, round(test_fraction * len(bases)))
        test.extend(bases[:n_test])
        train.extend(bases[n_test:])
    return {"train": sorted(train), "test": sorted(test)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build(raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    raw_hash = sha256_file(raw_path)
    if raw_hash != SOURCE_SHA256:
        raise RuntimeError(f"raw catalog hash {raw_hash} != pinned {SOURCE_SHA256}; run `fetch`")
    records, stats = parse_catalog(json.loads(raw_path.read_text(encoding="utf-8")))
    split = split_bases(records)
    test_set = set(split["test"])
    train = [r for r in records if r["base_id"] not in test_set]
    test = [r for r in records if r["base_id"] in test_set]

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"train.jsonl": train, "test.jsonl": test}
    for name, rows in files.items():
        _write_jsonl(out_dir / name, rows)
    (out_dir / "split.json").write_text(
        json.dumps({"seed": SEED, "test_fraction": TEST_FRACTION, **split}, indent=1) + "\n", encoding="utf-8"
    )

    def per_family(rows):
        return dict(sorted(Counter(r["family"] for r in rows).items()))

    manifest = {
        "source": {
            "repo": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "path": SOURCE_PATH,
            "sha256": raw_hash,
            "oscal_version": stats["oscal_version"],
            "catalog_version": stats["catalog_version"],
        },
        "split": {"seed": SEED, "test_fraction": TEST_FRACTION, "grouped_by": "base_id", "stratified_by": "family"},
        "counts": {
            "records": len(records),
            "withdrawn_dropped": stats["withdrawn_dropped"],
            "train": len(train),
            "test": len(test),
            "base_controls_train": len(split["train"]),
            "base_controls_test": len(split["test"]),
            "train_by_family": per_family(train),
            "test_by_family": per_family(test),
        },
        "outputs": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "test.jsonl", "split.json")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "build"])
    args = ap.parse_args(argv)
    if args.command == "fetch":
        print(f"wrote {fetch()}")
    else:
        m = build()
        print(json.dumps(m["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
