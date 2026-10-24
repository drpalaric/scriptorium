"""Cross-framework retrieval: map a control or requirement in one catalog to similar
controls in another, and score it against a published crosswalk.

Method: TF-IDF cosine similarity over control text. It is unsupervised: the
vectorizer sees the text of both catalogs but never the mappings. The configuration
is fixed in advance (not tuned on the crosswalk). SCF input and results stay local.

    python -m scriptorium.crosswalk      # needs `python -m scriptorium.scf build` first
"""

from __future__ import annotations

import json
import re
import sys
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scriptorium import baseline, ingest, paths

KS = (1, 5, 10)
_NIST_ID_RE = re.compile(r"^([A-Z]{2})-0*(\d+)(?:\(0*(\d+)\))?$")


def normalize_nist_id(raw: str) -> str | None:
    """`PM-01` -> `PM-1`, `AC-02(01)` -> `AC-2(1)`. None if the ID is not in that form."""
    m = _NIST_ID_RE.match(raw.strip())
    if not m:
        return None
    return f"{m[1]}-{m[2]}" + (f"({m[3]})" if m[3] else "")


def normalized_mapping(raw: dict[str, list[str]], valid_targets: set[str]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Normalize SCF->NIST mappings and report what could not be joined."""
    out: dict[str, set[str]] = {}
    stats = {"pairs": 0, "unparseable": 0, "not_in_target_catalog": 0}
    for scf_id, ids in raw.items():
        keep = set()
        for i in ids:
            stats["pairs"] += 1
            n = normalize_nist_id(i)
            if n is None:
                stats["unparseable"] += 1
            elif n not in valid_targets:
                stats["not_in_target_catalog"] += 1
            else:
                keep.add(n)
        if keep:
            out[scf_id] = keep
    return out, stats


def invert(mapping: dict[str, set[str]]) -> dict[str, set[str]]:
    inv: dict[str, set[str]] = {}
    for a, bs in mapping.items():
        for b in bs:
            inv.setdefault(b, set()).add(a)
    return inv


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words="english")


def rank_targets(query_texts: list[str], target_texts: list[str]) -> np.ndarray:
    """Return an (n_queries, n_targets) matrix of target indices, best match first."""
    vec = make_vectorizer().fit(list(query_texts) + list(target_texts))
    sims = (vec.transform(query_texts) @ vec.transform(target_texts).T).toarray()
    return np.argsort(-sims, axis=1, kind="stable")


def random_hit_at_k(n_targets: int, n_correct: int, k: int) -> float:
    """P(at least one correct target in a random top-k)."""
    if n_correct >= n_targets or k >= n_targets:
        return 1.0
    return 1.0 - comb(n_targets - n_correct, k) / comb(n_targets, k)


def score(queries: list[dict[str, Any]], targets: list[dict[str, Any]], truth: dict[str, set[str]]) -> dict[str, Any]:
    """Score retrieval of `targets` for each query that has a ground-truth mapping."""
    qs = [q for q in queries if truth.get(q["id"])]
    order = rank_targets([q["text"] for q in qs], [t["text"] for t in targets])
    target_ids = [t["id"] for t in targets]
    first_ranks = []
    for q, row in zip(qs, order):
        correct = truth[q["id"]]
        ranks = [i for i, j in enumerate(row, start=1) if target_ids[j] in correct]
        first_ranks.append(ranks[0])
    first = np.array(first_ranks)
    n_t = len(targets)
    return {
        "n_queries": len(qs),
        "n_targets": n_t,
        "hit_at_k": {str(k): float(np.mean(first <= k)) for k in KS},
        "random_hit_at_k": {
            str(k): float(np.mean([random_hit_at_k(n_t, len(truth[q["id"]]), k) for q in qs])) for k in KS
        },
        "mrr": float(np.mean(1.0 / first)),
        "median_first_correct_rank": float(np.median(first)),
    }


def run(scf_dir: Path | None = None) -> dict[str, Any]:
    scf_dir = scf_dir or paths.scf_dir()
    nist = baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")
    scf = baseline.load_jsonl(scf_dir / "train.jsonl") + baseline.load_jsonl(scf_dir / "test.jsonl")
    raw = json.loads((scf_dir / "nist_r5_2_mappings.json").read_text(encoding="utf-8"))
    truth, join_stats = normalized_mapping(raw, {r["id"] for r in nist})
    manifest = json.loads((scf_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "method": "TF-IDF cosine similarity (sublinear tf, English stop words), vectorizer fit on both catalogs' text; unsupervised, config fixed in advance",
        "ground_truth": "SCF OSCAL catalog `general-nist-800-53-r5-2` mappings, IDs normalized to NIST SP 800-53 Rev 5 labels",
        "join": {**join_stats, "scf_controls_with_mapping": len(truth), "distinct_nist_targets": len(invert(truth))},
        "scf_to_nist": score(scf, nist, truth),
        "nist_to_scf": score(nist, scf, invert(truth)),
        "provenance": {
            "scf_version": manifest["source"]["version"],
            "scf_file_sha256": manifest["source"]["sha256"],
            "nist_source_commit": ingest.SOURCE_COMMIT,
        },
    }


def main(argv: list[str] | None = None) -> int:
    out = run()
    path = paths.scf_dir() / "results" / "crosswalk_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for d in ("scf_to_nist", "nist_to_scf"):
        s = out[d]
        print(d, f"n={s['n_queries']} targets={s['n_targets']} hit@k={s['hit_at_k']} random={ {k: round(v,3) for k,v in s['random_hit_at_k'].items()} } mrr={s['mrr']:.3f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
