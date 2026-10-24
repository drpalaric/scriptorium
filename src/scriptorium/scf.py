"""Load a user-supplied Secure Controls Framework (SCF) OSCAL catalog into the same
labelled, split dataset format as the NIST ingestion.

The SCF file is NOT part of this repository and is never fetched. You download it
from the SCF Council and pass its path in. Outputs go to `~/.scriptorium/scf/`
(override with `$SCRIPTORIUM_HOME`), outside the repo. See the README for SCF's terms.

    python -m scriptorium.scf build --catalog /path/to/oscal-catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scriptorium import ingest, paths

NIST_MAPPING_CLASS = "general-nist-800-53-r5-2"
CONTROL_REF = ingest.CONTROL_REF


def _prop(node: dict[str, Any], name: str) -> str | None:
    for p in node.get("props", []):
        if p["name"] == name:
            return p["value"]
    return None


def _statement_text(part: dict[str, Any]) -> list[str]:
    lines = [part["prose"].strip()] if "prose" in part else []
    for sub in part.get("parts", []):
        lines.extend(_statement_text(sub))
    return lines


def _id_regex(domains: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:%s)-\d{1,3}(?:\.\d+)?" % "|".join(sorted(domains)), re.IGNORECASE)


def parse_scf_catalog(catalog: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    """Return (records, nist_mappings, stats). Orphaned controls are dropped and counted.

    `nist_mappings` maps SCF control ID to its raw NIST 800-53 R5.2 IDs (padded, e.g. `PM-01`).
    """
    cat = catalog["catalog"]
    domains = [g["id"].upper() for g in cat["groups"]]
    id_re = _id_regex(domains)

    records: list[dict[str, Any]] = []
    mappings: dict[str, list[str]] = {}
    orphaned = 0
    for group in cat["groups"]:
        domain = group["id"].upper()
        for c in group["controls"]:
            label = _prop(c, "label")
            if label is None:
                raise ValueError(f"control {c['id']} has no label prop")
            if _prop(c, "orphaned-control"):
                orphaned += 1
                continue
            if not label.startswith(domain + "-"):
                raise ValueError(f"{label} is not in group {domain}")
            statement = [p for p in c.get("parts", []) if p["name"] == "statement"]
            if not statement:
                raise ValueError(f"control {label} has no statement")
            text = re.sub(r"\s+", " ", " ".join(_statement_text(statement[0])))
            text = id_re.sub(CONTROL_REF, text)
            text = re.sub(r"\s+([;:,.])", r"\1", text).strip()
            if not text:
                raise ValueError(f"control {label} has empty statement text")
            records.append(
                {
                    "id": label,
                    "base_id": label.split(".")[0],
                    "family": domain,
                    "family_title": group["title"],
                    "title": c["title"],
                    "is_enhancement": "." in label,
                    "text": text,
                }
            )
            mappings[label] = sorted(p["value"] for p in c["props"] if p["name"] == "mapping" and p.get("class") == NIST_MAPPING_CLASS)
    stats = {
        "orphaned_dropped": orphaned,
        "scf_version": cat["metadata"].get("version"),
        "oscal_version": cat["metadata"].get("oscal-version"),
    }
    return records, mappings, stats


def build(catalog_path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or paths.scf_dir()
    records, mappings, stats = parse_scf_catalog(json.loads(catalog_path.read_text(encoding="utf-8")))
    split = ingest.split_bases(records)
    test_set = set(split["test"])
    train = [r for r in records if r["base_id"] not in test_set]
    test = [r for r in records if r["base_id"] in test_set]

    out_dir.mkdir(parents=True, exist_ok=True)
    ingest.write_jsonl(out_dir / "train.jsonl", train)
    ingest.write_jsonl(out_dir / "test.jsonl", test)
    (out_dir / "split.json").write_text(
        json.dumps({"seed": ingest.SEED, "test_fraction": ingest.TEST_FRACTION, **split}, indent=1) + "\n", encoding="utf-8"
    )
    (out_dir / "nist_r5_2_mappings.json").write_text(json.dumps(mappings, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    per_family = lambda rows: dict(sorted(Counter(r["family"] for r in rows).items()))
    manifest = {
        "note": "User-supplied SCF input; derived files are local-only and must not be committed or shared.",
        "source": {
            "file_name": catalog_path.name,
            "sha256": ingest.sha256_file(catalog_path),
            "version": stats["scf_version"],
            "oscal_version": stats["oscal_version"],
        },
        "split": {"seed": ingest.SEED, "test_fraction": ingest.TEST_FRACTION, "grouped_by": "base_id", "stratified_by": "domain"},
        "counts": {
            "records": len(records),
            "orphaned_dropped": stats["orphaned_dropped"],
            "train": len(train),
            "test": len(test),
            "base_controls_train": len(split["train"]),
            "base_controls_test": len(split["test"]),
            "with_nist_r5_2_mapping": sum(1 for v in mappings.values() if v),
            "train_by_family": per_family(train),
            "test_by_family": per_family(test),
        },
        "outputs": {
            name: ingest.sha256_file(out_dir / name)
            for name in ("train.jsonl", "test.jsonl", "split.json", "nist_r5_2_mappings.json")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the local SCF dataset from a downloaded OSCAL catalog")
    b.add_argument("--catalog", type=Path, required=True)
    b.add_argument("--out", type=Path, default=None, help="default: ~/.scriptorium/scf")
    args = ap.parse_args(argv)
    m = build(args.catalog, args.out)
    print(json.dumps({k: v for k, v in m["counts"].items() if not k.endswith("_by_family")}, indent=1))
    print(f"wrote dataset to {args.out or paths.scf_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
