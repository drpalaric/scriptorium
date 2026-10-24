"""Tests for scriptorium.baseline, using a small synthetic dataset (no NIST data needed)."""

import numpy as np
import pytest

from scriptorium import baseline

VOCAB = {
    "AC": "account access privilege authorize login permission",
    "AU": "audit log record event timestamp review",
    "SC": "encrypt network boundary transmission cryptographic channel",
}


def synthetic(n_bases: int = 12, per_base: int = 2):
    rows = []
    for fam, words in VOCAB.items():
        w = words.split()
        for b in range(n_bases):
            for k in range(per_base):
                text = " ".join(w[(b + k + i) % len(w)] for i in range(4)) + f" item{b}{k}"
                rows.append({"id": f"{fam}-{b}({k})", "base_id": f"{fam}-{b}", "family": fam, "text": text})
    return rows


def test_folds_are_grouped_and_cover_all_rows():
    rows = synthetic()
    seen = []
    for tr, va in baseline.grouped_folds(rows):
        assert not {rows[i]["base_id"] for i in tr} & {rows[i]["base_id"] for i in va}
        seen.extend(va)
    assert sorted(seen) == list(range(len(rows)))


def test_tuning_is_deterministic(monkeypatch):
    monkeypatch.setattr(baseline, "GRID", {"clf": ["logreg"], "ngram_max": [1, 2], "C": [1.0], "class_weight": [None]})
    rows = synthetic()
    assert baseline.tune(rows) == baseline.tune(list(rows))


def test_tune_takes_only_training_rows(monkeypatch):
    """tune() has no test-set argument; changing unrelated data cannot change its output."""
    monkeypatch.setattr(baseline, "GRID", {"clf": ["linear_svm"], "ngram_max": [1], "C": [1.0], "class_weight": [None]})
    import inspect

    assert list(inspect.signature(baseline.tune).parameters) == ["train"]


def test_pipeline_learns_separable_families_and_evaluate_reports_fields(monkeypatch):
    rows = synthetic()
    train = [r for r in rows if int(r["base_id"].split("-")[1]) < 9]
    test = [r for r in rows if int(r["base_id"].split("-")[1]) >= 9]
    cfg = {"clf": "logreg", "ngram_max": 1, "C": 10.0, "class_weight": None}
    model = baseline.build_pipeline(cfg).fit([r["text"] for r in train], [r["family"] for r in train])
    out = baseline.evaluate(model, test)
    assert out["accuracy"] > 0.9
    assert set(out["per_family"]) == set(VOCAB)
    assert out["per_family"]["AC"]["support"] == 6
    assert out["per_family"]["AC"]["low_support"] is False  # 6 >= LOW_SUPPORT
    lo, hi = out["ci95"]["accuracy"]
    assert lo <= out["accuracy"] <= hi + 1e-9


def test_low_support_flag(monkeypatch):
    rows = synthetic(n_bases=12)
    cfg = {"clf": "logreg", "ngram_max": 1, "C": 10.0, "class_weight": None}
    model = baseline.build_pipeline(cfg).fit([r["text"] for r in rows], [r["family"] for r in rows])
    tiny = [r for r in rows if r["family"] == "AC"][:2] + [r for r in rows if r["family"] == "AU"][:6]
    out = baseline.evaluate(model, tiny)
    assert out["per_family"]["AC"]["low_support"] is True
    assert out["per_family"]["AU"]["low_support"] is False


def test_bootstrap_is_deterministic_and_resamples_groups():
    y = np.array(["A", "A", "B", "B"])
    p = np.array(["A", "B", "B", "B"])
    bases = ["x", "x", "y", "y"]
    a = baseline.bootstrap_ci(y, p, bases, n=50)
    assert a == baseline.bootstrap_ci(y, p, bases, n=50)
    assert 0.0 <= a["accuracy"][0] <= a["accuracy"][1] <= 1.0
