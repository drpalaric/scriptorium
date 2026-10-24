"""TF-IDF + linear baseline: requirement text to control family.

Tuning uses grouped cross-validation on the training split only. The held-out test
split is read once, after the configuration is fixed.

    python -m scriptorium.baseline
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from scriptorium import ingest

SEED = ingest.SEED
N_FOLDS = 5
N_BOOTSTRAP = 1000
LOW_SUPPORT = 5

ROOT = ingest.ROOT
RESULTS_PATH = ROOT / "results" / "baseline_metrics.json"

GRID = {
    "clf": ["logreg", "linear_svm"],
    "ngram_max": [1, 2],
    "C": [0.3, 1.0, 3.0, 10.0],
    "class_weight": [None, "balanced"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def configs() -> list[dict[str, Any]]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(1, cfg["ngram_max"]), sublinear_tf=True, lowercase=True)
    if cfg["clf"] == "logreg":
        clf = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=2000, random_state=SEED)
    else:
        clf = LinearSVC(C=cfg["C"], class_weight=cfg["class_weight"], random_state=SEED)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def grouped_folds(rows: list[dict[str, Any]], n_folds: int = N_FOLDS, seed: int = SEED):
    """Yield (train_idx, val_idx); base controls never straddle a fold boundary."""
    y = [r["family"] for r in rows]
    groups = [r["base_id"] for r in rows]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(rows)), y, groups)


def tune(train: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the config with the best mean macro-F1 across grouped folds. Train data only."""
    X = np.array([r["text"] for r in train], dtype=object)
    y = np.array([r["family"] for r in train])
    folds = list(grouped_folds(train))
    scored = []
    for cfg in configs():
        f1s = []
        for tr, va in folds:
            model = build_pipeline(cfg).fit(X[tr], y[tr])
            f1s.append(f1_score(y[va], model.predict(X[va]), average="macro", zero_division=0))
        scored.append({"config": cfg, "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s))})
    best = max(scored, key=lambda s: s["cv_macro_f1_mean"])  # first wins ties, in grid order
    return best["config"], sorted(scored, key=lambda s: -s["cv_macro_f1_mean"])


def bootstrap_ci(y_true, y_pred, bases, n: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, list[float]]:
    """95% percentile intervals, resampling whole base controls with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    by_base: dict[str, list[int]] = {}
    for i, b in enumerate(bases):
        by_base.setdefault(b, []).append(i)
    groups = [np.array(v) for v in by_base.values()]
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo_hi = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy": lo_hi(accs), "macro_f1": lo_hi(f1s)}


def evaluate(model, test: list[dict[str, Any]]) -> dict[str, Any]:
    X = [r["text"] for r in test]
    y = [r["family"] for r in test]
    pred = model.predict(X)
    labels = sorted(set(y) | set(pred))
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    per_family = {
        f: {
            "precision": report[f]["precision"],
            "recall": report[f]["recall"],
            "f1": report[f]["f1-score"],
            "support": int(report[f]["support"]),
            "low_support": bool(report[f]["support"] < LOW_SUPPORT),
        }
        for f in labels
    }
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "ci95": bootstrap_ci(y, pred, [r["base_id"] for r in test]),
        "per_family": per_family,
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }


def run(processed_dir: Path = ingest.PROCESSED_DIR, manifest_path: Path = ingest.MANIFEST_PATH) -> dict[str, Any]:
    train = load_jsonl(processed_dir / "train.jsonl")
    cfg, cv_table = tune(train)

    model = build_pipeline(cfg).fit([r["text"] for r in train], [r["family"] for r in train])
    # First and only read of the test split.
    test = load_jsonl(processed_dir / "test.jsonl")
    result = evaluate(model, test)

    majority = DummyClassifier(strategy="most_frequent").fit([r["text"] for r in train], [r["family"] for r in train])
    result["majority_class_baseline"] = {
        "accuracy": float(accuracy_score([r["family"] for r in test], majority.predict([r["text"] for r in test]))),
        "predicts": str(majority.predict(["x"])[0]),
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "model": "TF-IDF + " + ("logistic regression" if cfg["clf"] == "logreg" else "linear SVM"),
        "selected_config": cfg,
        "selection": {"method": f"StratifiedGroupKFold({N_FOLDS}) by base_id on train only", "metric": "macro_f1", "seed": SEED},
        "cv_top5": cv_table[:5],
        "n_configs_tried": len(cv_table),
        "n_train": len(train),
        "n_test": len(test),
        "test": result,
        "provenance": {
            "source_commit": manifest["source"]["commit"],
            "train_sha256": manifest["outputs"]["train.jsonl"],
            "test_sha256": manifest["outputs"]["test.jsonl"],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "_model": model,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    out = run()
    model = out.pop("_model")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    t = out["test"]
    print(f"{out['model']}  config={out['selected_config']}")
    print(f"test accuracy {t['accuracy']:.3f} {t['ci95']['accuracy']}  macro-F1 {t['macro_f1']:.3f} {t['ci95']['macro_f1']}")
    print(f"majority-class accuracy {t['majority_class_baseline']['accuracy']:.3f}")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
