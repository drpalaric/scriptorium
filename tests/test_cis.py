"""Tests for the CIS experiments (E1, E2, E3) with synthetic CIS data. No real CIS content is used."""

import json
import stat

import numpy as np
import pytest

from conftest import crosswalk_files, xlsx
from scriptorium import crosswalks
from scriptorium.crosswalks import MappingStore
from scriptorium.evals import cis, cis_cli, cli, mapping
from scriptorium.evals.llm import FatalLLMError, LLMResponse

pytest.importorskip("openpyxl")


def controls_workbook(tmp_path, controls=3, per=6, name="c.xlsx"):
    rows = [["CIS Control", "CIS Safeguard", "Asset Class", "Security Function", "Title", "Description"]]
    for c in range(1, controls + 1):
        rows.append([f"{c}\xa0", "", "", "", f"Control {c} title", f"Control {c} text"])
        for k in range(1, per + 1):
            rows.append([str(c), float(f"{c}.{k}"), "Devices", "Protect", f"Title {c}.{k}", f"word{c} word{c}x topic{c} safeguard text number {k}"])
    return xlsx(tmp_path / name, {"Controls v8.1.2": rows})


def mapping_workbook(tmp_path, pairs, name="m.xlsx"):
    rows = [["", "CIS Control", "CIS Safeguard", "Asset Type", "Security Function", "Title", "Description", "IG1", "IG2", "IG3", "Relationship", "Control Identifier", "Control or Control Enhancement Name", "Control Text"]]
    for ctrl, sg_num, title, rel, nid in pairs:
        rows.append(["", str(ctrl), float(sg_num), "Devices", "Protect", title, "d", "x", "x", "x", rel, nid, "name", "t"])
    return xlsx(tmp_path / name, {"All CIS Controls & Safeguards": rows})


def build_store(tmp_path, files):
    crosswalks.build(files, out_dir=tmp_path / "store")
    return MappingStore.load(tmp_path / "store")


NIST = [{"id": "AC-1", "family": "AC", "text": "account access privilege authorize login"}, {"id": "AU-1", "family": "AU", "text": "audit log record event timestamp"},
        {"id": "SC-1", "family": "SC", "text": "encrypt network boundary transmission channel"}, {"id": "CM-8", "family": "CM", "text": "inventory component hardware asset"}]


# ---- preprocessing ----------------------------------------------------------------------------------------------


def test_cross_references_are_replaced_so_a_number_cannot_leak_the_label():
    t, changed = cis.strip_cis_refs("Do this per Safeguard 4.1 and Controls 12 and safeguards 3.10, not other things.")
    assert changed and "4.1" not in t and "12" not in t and "3.10" not in t and t.count("[safeguard]") == 2 and "[control]" in t
    assert cis.strip_cis_refs("Nothing to replace here.") == ("Nothing to replace here.", False)


def test_load_cis_requires_text_and_control_titles(tmp_path):
    with pytest.raises(cis.CisDataError, match="no crosswalk store"):
        cis.load_cis(None)
    store = build_store(tmp_path, {"cis-attack": crosswalk_files(tmp_path)["cis-attack"]})  # titles only, no descriptions
    with pytest.raises(cis.CisDataError, match="descriptions are missing"):
        cis.load_cis(store)


# ---- E2 -------------------------------------------------------------------------------------------------------------


@pytest.fixture()
def e2_store(tmp_path):
    pairs = [(1, 1.1, "Title 1.1", "Subset", "AC-1"), (1, 1.1, "Title 1.1", "Superset", "AU-1"), (1, 1.2, "Title 1.2", "Equivalent", "SC-1"),
             (3, 3.1, "Title 3.1", "Subset", "AC-1"), (3, 3.1, "Title 3.1", "Subset", "ZZ-99"),  # ZZ-99 is not in the dataset
             (3, 3.2, "Title 3.2", "Subset", "ZZ-98")]  # only unknown targets: excluded
    return build_store(tmp_path, {"cis-controls": controls_workbook(tmp_path, 3, 6), "cis-nist-mapping": mapping_workbook(tmp_path, pairs)})


def test_e2_items_use_description_only_with_acceptable_families_and_drop_unknown_targets(e2_store):
    d = cis.build_e2(e2_store, NIST)
    ids = [i["id"] for i in d["items"]]
    assert ids == ["cis:1.1", "cis:1.2", "cis:3.1"]  # 3.2 had no usable target
    assert d["acceptable"] == {"cis:1.1": {"AC", "AU"}, "cis:1.2": {"SC"}, "cis:3.1": {"AC"}}
    assert [i["base_id"] for i in d["items"]] == ["1", "1", "3"] and d["n_single_family"] == 2
    msgs = cis.user_messages(d["items"])
    assert all("Title" not in m and "cis:" not in m and "Control" not in m for m in msgs) and "word1" in msgs[0]


def test_e2_reference_baselines(e2_store):
    d = cis.build_e2(e2_store, NIST)  # families in NIST: AC, AU, SC, CM  -> 4
    assert d["random_lenient"] == pytest.approx(np.mean([2 / 4, 1 / 4, 1 / 4]))
    assert d["most_frequent_family"] in {"AC"} and d["most_frequent_lenient"] == pytest.approx(2 / 3)


def test_numeric_ids_sort_numerically_and_ten_is_kept(tmp_path):
    store = build_store(tmp_path, {"cis-controls": controls_workbook(tmp_path, 1, 11), "cis-nist-mapping": mapping_workbook(
        tmp_path, [(1, float(f"1.{k}"), f"Title 1.{k}", "Subset", "AC-1") for k in range(1, 12)])})
    d = cis.build_e2(store, NIST)
    assert [i["id"] for i in d["items"]] == [f"cis:1.{k}" for k in range(1, 12)]


def out(i, base, pred, true="x", status="ok"):
    return mapping.Outcome(f"cis:{i}", base, true, pred, status)


def test_score_e2_lenient_strict_invalid_and_the_registered_rule():
    acceptable = {f"cis:{i}": ({"AC"} if i % 3 else {"AC", "AU"}) for i in range(30)}
    clusters = {f"cis:{i}": str(i // 3) for i in range(30)}
    llm = [out(i, clusters[f"cis:{i}"], "AC") for i in range(30)]                       # always right (AC is in every set)
    sk = {f"cis:{i}": ("AU" if i % 3 else "AC") for i in range(30)}                       # wrong on every single-family item
    r = cis.score_e2(llm, sk, acceptable, clusters, n_boot=300)
    assert r["lenient"]["llm"] == 1.0 and r["lenient"]["sklearn"] == pytest.approx(10 / 30)
    assert r["lenient"]["only_llm"] == 20 and r["lenient"]["both"] == 10 and r["lenient"]["only_sklearn"] == 0 and r["lenient"]["neither"] == 0
    assert r["strict_single_family"]["n"] == 20 and r["strict_single_family"]["llm"] == 1.0 and r["strict_single_family"]["sklearn"] == 0.0
    assert r["lenient"]["llm_minus_sklearn_ci95"][0] > 0 and r["h1_supported"] is True and r["n_clusters"] == 10
    same = cis.score_e2(llm, {k: "AC" for k in sk}, acceptable, clusters, n_boot=300)
    assert same["lenient"]["llm_minus_sklearn"] == 0 and same["h1_supported"] is False
    bad = [out(i, clusters[f"cis:{i}"], None, status="invalid_schema") for i in range(30)]
    assert cis.score_e2(bad, sk, acceptable, clusters, n_boot=100)["lenient"]["llm"] == 0.0  # invalid answers are wrong
    assert cis.score_e2([out(0, "0", None, status="not_run")], sk, acceptable, clusters) == {"n_scored": 0}
    assert cis.score_e2(llm, sk, acceptable, clusters, n_boot=300) == r  # deterministic


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = ["a", "a", "b", "b", "c", "c"]
    seen = []
    cis._boot_clusters(clusters, lambda idx: seen.append(sorted(set(idx))) or {"x": 0.0}, n=50)
    assert all(len(s) % 2 == 0 for s in seen)  # indices always come in whole clusters (pairs)


# ---- E1 -------------------------------------------------------------------------------------------------------------


@pytest.fixture()
def e1_store(tmp_path):
    return build_store(tmp_path, {"cis-controls": controls_workbook(tmp_path, 12, 6)})


def test_e1_items_labels_and_numeric_control_order(e1_store):
    d = cis.build_e1(e1_store)
    assert len(d["items"]) == 72 and list(d["families"]) == [str(i) for i in range(1, 13)]  # numeric, not '1','10','11','12','2'
    assert d["items"][0]["family"] == "1" and d["items"][-1]["id"] == "cis:12.6" and all(i["base_id"] == i["id"] for i in d["items"])
    from scriptorium.evals.prompts import load_prompt

    sp = cis.e1_system_prompt(load_prompt("cis_control", "v1"), d["families"])
    assert sp.index("2: Control 2") < sp.index("10: Control 10") and '{"family": "<number of the control>"}' in sp


def test_cv_predictions_are_out_of_fold_complete_and_deterministic(e1_store, monkeypatch):
    items = cis.build_e1(e1_store)["items"]
    fits = []
    real = cis.baseline.build_pipeline

    def spy(cfg):
        pipe = real(cfg)
        orig = pipe.fit

        def fit(X, y, **kw):
            fits.append(set(X))
            return orig(X, y, **kw)

        pipe.fit = fit
        return pipe

    monkeypatch.setattr(cis.baseline, "build_pipeline", spy)
    p1 = cis.sklearn_cv_predictions(items)
    monkeypatch.setattr(cis.baseline, "build_pipeline", real)
    assert set(p1) == {i["id"] for i in items} and p1 == cis.sklearn_cv_predictions(items)
    assert len(fits) == 5
    seen = set()
    for train_texts, _ in zip(fits, range(5)):
        assert len(train_texts) < len(items)
        seen |= {i["id"] for i in items if i["text"] not in train_texts}
    assert seen == {i["id"] for i in items}  # every item was held out of exactly the fold that predicted it
    assert np.mean([p1[i["id"]] == i["family"] for i in items]) > 0.8  # separable synthetic vocabulary


# ---- E3 -------------------------------------------------------------------------------------------------------------


def test_e3_retrieves_the_matching_text_in_both_directions_and_reports_the_equivalent_subset(tmp_path):
    nist = [{"id": f"AC-{k}", "family": "AC", "text": f"uniqueword{k} alpha{k} beta{k} discussion about topic {k}"} for k in range(1, 7)]
    rows = [["CIS Control", "CIS Safeguard", "Asset Class", "Security Function", "Title", "Description"], ["1\xa0", "", "", "", "T", "x"]]
    rows += [["1", float(f"1.{k}"), "D", "P", f"Title {k}", f"uniqueword{k} alpha{k} beta{k} safeguard {k}"] for k in range(1, 7)]
    pairs = [(1, float(f"1.{k}"), f"Title {k}", "Equivalent" if k <= 2 else "Subset", f"AC-{k}") for k in range(1, 7)]
    store = build_store(tmp_path, {"cis-controls": xlsx(tmp_path / "e3.xlsx", {"Controls v8.1": rows}), "cis-nist-mapping": mapping_workbook(tmp_path, pairs)})
    r = cis.run_e3(store, nist)
    a = r["all_pairs"]
    assert a["pairs"] == 6 and a["cis_to_nist"]["n_queries"] == 6 and a["nist_to_cis"]["n_queries"] == 6
    assert a["cis_to_nist"]["hit_at_k"]["1"] == 1.0 and a["nist_to_cis"]["hit_at_k"]["1"] == 1.0 and a["cis_to_nist"]["mrr"] == 1.0
    assert a["cis_to_nist"]["random_hit_at_k"]["1"] < 0.5
    eq = r["equivalent_only"]
    assert eq["pairs"] == 2 and eq["cis_to_nist"]["n_queries"] == 2 and eq["nist_to_cis"]["n_queries"] == 2


# ---- command line --------------------------------------------------------------------------------------------------


class Spy:
    def __init__(self, family="AC"):
        self.sent, self.family = [], family

    def complete(self, req):
        self.sent.append(req)
        return LLMResponse(text=json.dumps({"family": self.family}), stop_reason="end_turn", model=req.model, input_tokens=100, output_tokens=5)


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    made = {"n": 0, "spy": None}

    def factory(**kw):
        made["n"] += 1
        made["spy"] = made["spy"] or Spy()
        return made["spy"]

    monkeypatch.setattr(cis_cli, "AnthropicClient", factory)
    common = ["--store", str(tmp_path / "store"), "--cache", str(tmp_path / "c.db"), "--out", str(tmp_path / "out")]
    return made, common, tmp_path


def test_e3_needs_no_api_and_writes_an_aggregate_summary(tmp_path, cli_env, e1_store):
    made, common, tmp = cli_env
    # reuse the store built by e1_store (same tmp_path) – E3 over it has no pairs, which must still run cleanly
    assert cli.main(["cis", "e3", *common]) == 0 and made["n"] == 0


def test_live_runs_need_both_flags_and_refuse_before_any_call(cli_env, e1_store, capsys):
    made, common, _ = cli_env
    assert cli.main(["cis", "e1", *common, "--yes"]) == 3 and "--allow-cis" in capsys.readouterr().err
    assert cli.main(["cis", "e1", *common, "--allow-cis"]) == 3 and "--yes" in capsys.readouterr().err
    assert cli.main(["cis", "e1", *common, "--allow-cis", "--yes", "--budget-usd", "0.0001"]) == 3
    assert cli.main(["cis", "e1", *common, "--plan"]) == 0 and made["n"] == 0


def test_e1_run_sends_descriptions_only_keeps_summaries_aggregate_and_replays(cli_env, e1_store, capsys):
    made, common, tmp = cli_env
    assert cli.main(["cis", "e1", *common, "--allow-cis", "--yes"]) == 0
    sent = made["spy"].sent
    assert len(sent) == 72 and made["n"] == 1
    for r in sent:
        assert "Title " not in r.user and "cis:" not in r.user and r.user.startswith("<requirement>\n")
        assert "Control 1 title" in r.system and "topic1" not in r.system  # titles of the Controls are the label list; safeguard text is not
    summaries = sorted((tmp / "out").glob("*-summary.json"))
    items = sorted((tmp / "out").glob("*-items.json"))
    assert len(summaries) == 1 and len(items) == 1
    s = json.loads(summaries[0].read_text())
    assert s["allow_cis"] is True and s["prompt"]["version"] == "v1" and s["model"] == "claude-sonnet-5-5" and "8651103" in s["preregistration"]
    text = summaries[0].read_text()
    assert "topic1" not in text and "cis:" not in text and "per_family" not in text  # aggregates only
    assert "LOCAL ONLY" in items[0].read_text() and "cis:1.1" in items[0].read_text()
    for p in (*summaries, *items):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert cli.main(["cis", "e1", *common, "--replay"]) == 0 and made["n"] == 1  # replay: no new client, no new call
    assert len(made["spy"].sent) == 72


def test_fatal_api_error_exits_cleanly(cli_env, e1_store, monkeypatch, capsys):
    made, common, _ = cli_env

    class Bad:
        def complete(self, r):
            raise FatalLLMError("no credentials found")

    monkeypatch.setattr(cis_cli, "AnthropicClient", lambda **kw: Bad())
    assert cli.main(["cis", "e1", *common, "--allow-cis", "--yes"]) == 4
    assert "no credentials found" in capsys.readouterr().err


def test_missing_store_is_a_clear_error(tmp_path, capsys):
    assert cli.main(["cis", "e1", "--store", str(tmp_path / "none"), "--plan"]) == 2
    assert "no crosswalk store" in capsys.readouterr().err
