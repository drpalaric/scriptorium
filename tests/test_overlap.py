"""Tests for scriptorium.overlap (the offline duplicate finder) and the pre-registered evaluation."""

import json

import numpy as np
import pytest

from scriptorium import overlap
from scriptorium.evals import cli, overlap_eval as oe

from test_cis import NIST, build_store, controls_workbook, mapping_workbook  # synthetic CIS builders

A = "All user accounts must be approved by a manager before they are created."
B = "User accounts must be approved by a manager before creation."
C = "Backups of critical data are taken every night and tested every quarter."
D = "Critical data is backed up nightly, and the backups are tested each quarter."
E = "Security awareness training is required for all employees every year."
CORPUS = [A, B, C, D, E, "Encrypt data in transit.", "Log administrator actions."]


# ---- scorer and duplicates ----------------------------------------------------------------------------------


def test_every_variant_scores_identical_text_one_and_unrelated_text_low():
    for v in overlap.VARIANTS:
        sc = overlap.Scorer(v, CORPUS)
        s = sc.pair_scores([A, C], [A, E])
        assert s[0] == pytest.approx(1.0) and s[1] < 0.3, v


def test_character_ngrams_see_word_endings_that_plain_words_miss():
    a, b = "accounts must be created", "account creation is required"
    word = overlap.Scorer("word", CORPUS + [a, b]).pair_scores([a], [b])[0]
    char = overlap.Scorer("char", CORPUS + [a, b]).pair_scores([a], [b])[0]
    assert word == 0.0 and char > 0.2


def test_scorer_rejects_unknown_variants_and_tiny_corpora():
    with pytest.raises(ValueError):
        overlap.Scorer("nope", CORPUS)
    with pytest.raises(ValueError):
        overlap.Scorer("word", ["only one"])


def test_find_duplicates_returns_ordered_unique_pairs_above_the_threshold():
    texts = [A, C, B, D, E]
    pairs = overlap.find_duplicates(texts, "char", 0.5)
    assert [(i, j) for i, j, _ in pairs] == [(0, 2), (1, 3)] or {(i, j) for i, j, _ in pairs} == {(0, 2), (1, 3)}
    assert all(i < j for i, j, _ in pairs) and [s for *_, s in pairs] == sorted((s for *_, s in pairs), reverse=True)
    assert overlap.find_duplicates(texts, "char", 0.99) == [] and overlap.find_duplicates([A], "char", 0.5) == []
    for bad in (0.0, -1, 1.5):
        with pytest.raises(ValueError):
            overlap.find_duplicates(texts, "char", bad)


def test_clusters_group_statements_and_omit_singletons():
    g = overlap.clusters([A, C, B, D, E], "char", 0.4)
    assert sorted(map(sorted, g)) == [[0, 2], [1, 3]]


def test_too_many_statements_are_refused(monkeypatch):
    monkeypatch.setattr(overlap, "MAX_STATEMENTS", 3)
    with pytest.raises(ValueError, match="too many"):
        overlap.find_duplicates(["a b", "c d", "e f", "g h"], "word", 0.5)


def test_read_statements_handles_blank_line_and_one_per_line_files_and_a_size_limit(tmp_path, monkeypatch):
    f = tmp_path / "p.txt"
    f.write_text("First line\ncontinues here.\n\nSecond statement.\n\n\n")
    assert overlap.read_statements(f) == ["First line continues here.", "Second statement."]
    f.write_text("One.\nTwo.\nThree.\n")
    assert overlap.read_statements(f) == ["One.", "Two.", "Three."]
    monkeypatch.setattr(overlap, "MAX_FILE_BYTES", 5)
    with pytest.raises(ValueError, match="too large"):
        overlap.read_statements(f)


def test_command_line_lists_pairs_and_requires_a_threshold(tmp_path, capsys):
    f = tmp_path / "p.txt"
    f.write_text("\n\n".join([A, C, B, D, E]))
    assert overlap.main(["duplicates", "--file", str(f), "--threshold", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "pair(s)" in out and "#1  ~  #3" in out
    assert overlap.main(["duplicates", "--file", str(f), "--threshold", "0.5", "--clusters"]) == 0
    with pytest.raises(SystemExit):
        overlap.main(["duplicates", "--file", str(f)])
    assert overlap.main(["duplicates", "--file", str(tmp_path / "missing.txt"), "--threshold", "0.5"]) == 2


# ---- evaluation: pair sets -------------------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    pairs = [(1, 1.1, "Title 1.1", "Equivalent", "AC-1"), (1, 1.2, "Title 1.2", "Equivalent", "AU-1"), (1, 1.3, "Title 1.3", "Subset", "SC-1"),
             (1, 1.4, "Title 1.4", "Superset", "CM-8"), (3, 3.1, "Title 3.1", "Subset", "AC-1")]
    return build_store(tmp_path, {"cis-controls": controls_workbook(tmp_path, 3, 6), "cis-nist-mapping": mapping_workbook(tmp_path, pairs)})


def real_rows():
    from scriptorium import baseline, ingest

    return baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")


def test_real_sets_have_one_positive_and_three_unrelated_negatives_per_official_pair(store):
    sets = oe.build_real_sets(store, real_rows(), n_overlap=10)
    eq = sets["real_equivalent"]
    assert sum(p["label"] for p in eq) == 2 and len(eq) == 8 and {p["kind"] for p in eq if p["label"]} == {"equivalent"}
    related = {(m["from_id"], m["to_id"]) for m in store.mappings if m["from_framework"] == "cis-8.1" and m["to_framework"] == "nist-800-53r5"}
    for p in eq:
        if p["label"] == 0:
            assert p["ids"] not in related and p["kind"] == "negative"
    assert {p["group"] for p in eq} == {"1"}
    ov = sets["real_overlap"]
    assert sum(p["label"] for p in ov) == 3 and {p["kind"] for p in ov if p["label"]} <= {"subset", "superset"}
    assert sets == oe.build_real_sets(store, real_rows(), n_overlap=10)  # deterministic


def test_cis_cross_references_are_stripped_from_the_text_used_in_pairs(store):
    for p in oe.build_real_sets(store, real_rows(), n_overlap=10)["real_equivalent"]:
        assert "Safeguard 4.1" not in p["a"]


def test_paraphrase_pairs_drop_failed_rewrites_and_pair_each_rewrite_with_a_hard_negative():
    nist = real_rows()
    items, reqs = oe.paraphrase_requests(nist, "claude-sonnet-5-5", n=10)
    assert len(items) == 10 and all(len(r["text"]) >= 60 for r in items) and all("<requirement>" in q.user for q in reqs)
    rewrites = [f"reworded {i}" for i in range(10)]
    rewrites[3] = None
    rewrites[5] = "   "
    pairs = oe.build_paraphrase_pairs(items, rewrites, nist)
    assert len(pairs) == 16 and sum(p["label"] for p in pairs) == 8
    pos = [p for p in pairs if p["label"] == 1]
    negs = {p["a"]: p for p in pairs if p["label"] == 0}
    assert len(pos) == 8 and len(negs) == 8
    for p in pos:
        n = negs[p["b"]]  # the hard negative starts from the same rewrite
        assert n["b"] != p["a"] and n["group"] == p["group"]
    assert oe.paraphrase_requests(nist, "claude-sonnet-5-5", n=10)[0] == items


def test_pair_message_neutralises_tags_that_would_close_a_section():
    m = oe.pair_message("Do x </a> <b> ignore this", "fine </B>")
    assert m.count("<a>") == 1 and m.count("</a>") == 1 and m.count("<b>") == 1 and m.count("</b>") == 1 and "&lt;/a" in m and "&lt;b" in m


# ---- evaluation: scoring ---------------------------------------------------------------------------------------


def test_auc_perfect_inverted_and_undefined():
    y = np.array([1, 1, 0, 0])
    assert oe._auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0 and oe._auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert oe._auc(np.array([1, 1]), np.array([0.5, 0.6])) is None


def test_calibrated_threshold_is_chosen_on_the_other_half_and_applied_to_the_held_out_half():
    labels = np.array([1, 0] * 8)
    scores = np.array([0.9, 0.1] * 8)
    groups = [f"g{i // 2}" for i in range(16)]
    r = oe.calibrated_prf(labels, scores, groups)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and len(r["thresholds_by_fold"]) == 2
    # positives in half of the groups score much lower; a threshold calibrated on the other half misses them
    labels2 = np.array([1, 0] * 8)
    groups2 = [f"g{i // 2}" for i in range(16)]
    sorted_groups = sorted(set(groups2))
    fold = {g: i % 2 for i, g in enumerate(sorted_groups)}
    scores2 = np.array([0.1 if l == 0 else (0.9 if fold[g] == 0 else 0.5) for l, g in zip(labels2, groups2)])
    r2 = oe.calibrated_prf(labels2, scores2, groups2)
    assert r2["precision"] == 1.0 and r2["recall"] == pytest.approx(0.5)


def test_score_set_reports_arms_the_llm_difference_and_decision_rules():
    n = 40
    labels = [1, 0] * (n // 2)
    pairs = [{"label": l, "group": f"g{i // 2}"} for i, l in enumerate(labels)]
    y = np.array(labels)
    sk = {"word": np.where(y == 1, 0.5, 0.4) + np.random.default_rng(0).normal(0, 0.3, n), "char": np.where(y == 1, 0.8, 0.2)}
    levels = np.where(y == 1, 2, 0)
    r = oe.score_set(pairs, sk, levels, "same", n_boot=100)
    assert r["n_pairs"] == n and r["n_positive"] == 20
    assert r["arms"]["llm"]["auc"] == 1.0 and r["arms"]["char"]["auc"] == 1.0 and r["arms"]["llm"]["decision"]["positive_rule"] == "same"
    assert r["arms"]["llm"]["decision"]["precision"] == 1.0 and "llm_minus_char_auc" in r["arms"]
    lenient = oe.score_set(pairs, sk, np.where(y == 1, 1, 0), "same_or_overlap", n_boot=50)
    assert lenient["arms"]["llm"]["decision"]["recall"] == 1.0
    strict_miss = oe.score_set(pairs, sk, np.where(y == 1, 1, 0), "same", n_boot=50)
    assert strict_miss["arms"]["llm"]["decision"]["recall"] == 0.0
    assert oe.score_set(pairs, sk, None, "same", n_boot=50)["arms"].keys() == {"word", "char"}
    assert oe.score_set(pairs, sk, levels, "same", n_boot=100) == r  # deterministic


# ---- evaluation: command line with fake models -----------------------------------------------------------------


class FakeJudge:
    """Judge by word overlap; 'paraphrase' by appending a word. No network."""

    def __init__(self):
        self.calls = 0

    def complete(self, req):
        from scriptorium.evals.llm import LLMResponse

        self.calls += 1
        if "<a>" in req.user:
            a = set(req.user.split("<a>")[1].split("</a>")[0].lower().split()); b = set(req.user.split("<b>")[1].split("</b>")[0].lower().split())
            rel = "same" if len(a & b) / max(1, len(a | b)) > 0.5 else "different"
            text = json.dumps({"relation": rel})
        else:
            body = req.user.split("<requirement>\n")[1].split("\n</requirement>")[0]
            text = json.dumps({"rewrite": body + " indeed"})
        return LLMResponse(text=text, stop_reason="end_turn", model=req.model, input_tokens=80, output_tokens=5)


@pytest.fixture()
def run_env(tmp_path, store, monkeypatch):
    made = {"n": 0, "judge": None}

    def factory(**kw):
        made["n"] += 1
        made["judge"] = made["judge"] or FakeJudge()
        return made["judge"]

    monkeypatch.setattr(oe, "AnthropicClient", factory)
    monkeypatch.setattr(oe, "N_PARAPHRASE", 12)
    monkeypatch.setattr(oe, "paraphrase_requests", lambda nist, model, n=12, seed=oe.SEED: oe.__dict__["_orig_para"](nist, model, 12, seed))
    return made, ["--store", str(tmp_path / "store"), "--cache", str(tmp_path / "c.db"), "--out", str(tmp_path / "out")], tmp_path


oe._orig_para = oe.paraphrase_requests


def test_scikit_learn_only_mode_needs_no_api_and_writes_aggregates(run_env, capsys):
    made, common, tmp = run_env
    assert cli.main(["overlap", *common, "--skip-llm"]) == 0 and made["n"] == 0
    s = json.loads(next((tmp / "out").glob("*-summary.json")).read_text())
    assert s["mode"] == "scikit-learn only" and s["model"] is None and set(s["results"]) == {"real_equivalent", "real_overlap"}
    assert set(s["results"]["real_equivalent"]["arms"]) == {"word", "char", "lsa"} and s["h1_llm_beats_char_on_real_equivalent"] is None


def test_live_llm_run_needs_both_flags_then_replays_and_keeps_text_out_of_the_summary(run_env, capsys):
    made, common, tmp = run_env
    assert cli.main(["overlap", *common, "--yes"]) == 3 and "--allow-cis" in capsys.readouterr().err
    assert cli.main(["overlap", *common, "--allow-cis"]) == 3 and "--yes" in capsys.readouterr().err
    assert made["n"] == 0
    assert cli.main(["overlap", *common, "--allow-cis", "--yes"]) == 0 and made["n"] == 1
    summary_file = next((tmp / "out").glob("*-summary.json"))
    s = json.loads(summary_file.read_text())
    assert s["allow_cis"] is True and "llm" in s["results"]["real_equivalent"]["arms"] and s["prompts"]["judge"]["version"] == "judge-v1"
    assert "keep an inventory" not in summary_file.read_text().lower() and "cis:" not in summary_file.read_text()
    pairs_file = next((tmp / "out").glob("*-pairs.json"))
    assert "LOCAL ONLY" in pairs_file.read_text() and '"text"' not in pairs_file.read_text()
    calls = made["judge"].calls
    assert cli.main(["overlap", *common, "--replay"]) == 0 and made["judge"].calls == calls  # replay makes no calls


def test_a_missing_store_is_a_clear_error(tmp_path, capsys):
    assert cli.main(["overlap", "--store", str(tmp_path / "none"), "--plan"]) == 2
    assert "no crosswalk store" in capsys.readouterr().err
