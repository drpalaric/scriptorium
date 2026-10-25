"""Find statements that say the same thing: near-duplicate pairs and clusters. Offline, free, deterministic.

This is the part scikit-learn is good at: cheap similarity over many statements. It does not judge whether a
policy satisfies a control, and it misses paraphrase that shares few words. Pair it with a reader (a person or
an LLM) for the pairs it surfaces.

    python -m scriptorium.overlap duplicates --file my_policy.txt --threshold 0.6
    python -m scriptorium.overlap duplicates --catalog nist --threshold 0.8 --variant char

A file holds statements separated by blank lines, or one per line if it has no blank lines. Nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from scriptorium import baseline, ingest

VARIANTS = ("word", "char", "lsa")
LSA_DIMS = 150
MAX_STATEMENTS = 5000  # the pairwise matrix is dense
MAX_FILE_BYTES = 2_000_000


class Scorer:
    """Cosine similarity between statements. Vectorizer (and SVD for `lsa`) is fitted on `corpus`, with no labels."""

    def __init__(self, variant: str, corpus: list[str]):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if len(corpus) < 2:
            raise ValueError("the corpus needs at least two texts")
        self.variant = variant
        if variant == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        else:
            self.vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words="english")
        X = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(min(LSA_DIMS, X.shape[1] - 1, X.shape[0] - 1), random_state=0).fit(X) if variant == "lsa" else None

    def embed(self, texts: list[str]):
        X = self.vec.transform(texts)
        return normalize(self.svd.transform(X)) if self.svd is not None else X

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        M = self.embed(a) @ self.embed(b).T
        return M.toarray() if hasattr(M, "toarray") else np.asarray(M)

    def pair_scores(self, a: list[str], b: list[str]) -> np.ndarray:
        """Cosine similarity of a[i] with b[i], elementwise."""
        A, B = self.embed(a), self.embed(b)
        s = A.multiply(B).sum(axis=1) if hasattr(A, "multiply") else (A * B).sum(axis=1)
        return np.asarray(s).ravel()


def nist_corpus() -> list[str]:
    return [r["text"] for r in baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")]


def find_duplicates(texts: list[str], variant: str = "char", threshold: float = 0.6, corpus: list[str] | None = None) -> list[tuple[int, int, float]]:
    """All pairs (i, j, similarity) with i < j at or above `threshold`, most similar first."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if len(texts) > MAX_STATEMENTS:
        raise ValueError(f"too many statements ({len(texts)}); the limit is {MAX_STATEMENTS}")
    if len(texts) < 2:
        return []
    S = Scorer(variant, (corpus if corpus is not None else texts) + (texts if corpus is not None else [])).similarity(texts, texts)
    i, j = np.triu_indices(len(texts), 1)
    keep = S[i, j] >= threshold
    pairs = sorted(zip(i[keep].tolist(), j[keep].tolist(), S[i, j][keep].tolist()), key=lambda p: -p[2])
    return [(a, b, round(s, 4)) for a, b, s in pairs]


def clusters(texts: list[str], variant: str = "char", threshold: float = 0.6, corpus: list[str] | None = None) -> list[list[int]]:
    """Groups of statements that are all, on average, at least `threshold` similar (average linkage). Singletons omitted."""
    if len(texts) < 2:
        return []
    if len(texts) > MAX_STATEMENTS:
        raise ValueError(f"too many statements ({len(texts)}); the limit is {MAX_STATEMENTS}")
    S = Scorer(variant, (corpus if corpus is not None else texts) + (texts if corpus is not None else [])).similarity(texts, texts)
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1.0 - threshold).fit_predict(np.clip(1.0 - S, 0, None))
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g[0]))


def read_statements(path: Path) -> list[str]:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("file is too large")
    text = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in text.split("\n\n")] if "\n\n" in text else [ln.strip() for ln in text.splitlines()]
    return [" ".join(p.split()) for p in parts if p.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("duplicates")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path)
    src.add_argument("--catalog", choices=["nist"])
    d.add_argument("--variant", choices=VARIANTS, default="char")
    d.add_argument("--threshold", type=float, required=True, help="similarity from 0 to 1; choose it from the calibration in docs/experiments/overlap-results.md")
    d.add_argument("--clusters", action="store_true", help="group statements instead of listing pairs")
    d.add_argument("--limit", type=int, default=50)
    args = ap.parse_args(argv)
    try:
        if args.file:
            texts, labels = read_statements(args.file), None
        else:
            rows = baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")
            texts, labels = [r["text"] for r in rows], [r["id"] for r in rows]
        name = lambda i: labels[i] if labels else f"#{i + 1}"
        if args.clusters:
            for g in clusters(texts, args.variant, args.threshold, corpus=nist_corpus() if args.file else None)[: args.limit]:
                print(f"cluster of {len(g)}: " + ", ".join(name(i) for i in g[:12]) + (" ..." if len(g) > 12 else ""))
        else:
            pairs = find_duplicates(texts, args.variant, args.threshold, corpus=nist_corpus() if args.file else None)
            print(f"{len(pairs)} pair(s) at or above {args.threshold} ({args.variant}); showing up to {args.limit}")
            for a, b, s in pairs[: args.limit]:
                print(f"  {s:.2f}  {name(a)}  ~  {name(b)}   | {texts[a][:60]!r} | {texts[b][:60]!r}")
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
