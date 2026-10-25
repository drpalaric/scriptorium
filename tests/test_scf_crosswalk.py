"""Tests for scriptorium.scf and scriptorium.crosswalk.

All fixtures are synthetic (made-up domains and IDs). No real SCF content is used or
required, and the guardrail test checks none is tracked in git.
"""

import json
import subprocess

import pytest

from scriptorium import crosswalk, ingest, paths, scf


def control(domain, label, text, nist=(), orphaned=False, title="T"):
    props = [{"name": "label", "value": label}]
    props += [{"name": "mapping", "value": v, "class": scf.NIST_MAPPING_CLASS} for v in nist]
    props += [{"name": "mapping", "value": "X-1", "class": "some-other-framework"}]
    if orphaned:
        props.append({"name": "orphaned-control", "value": "ORPHANED CONTROL"})
    return {"id": label.lower(), "title": title, "props": props, "parts": [{"name": "statement", "prose": text}]}


FIXTURE = {
    "catalog": {
        "metadata": {"version": "9.9", "oscal-version": "1.0.0"},
        "groups": [
            {
                "id": "abc",
                "title": "Alpha Domain",
                "controls": [
                    control("ABC", "ABC-01", "Maintain the inventory.\nSee ABC-02 and XYZ-01.2 for details .", nist=["PM-01", "AC-02(01)"]),
                    control("ABC", "ABC-01.1", "Automate the inventory.", nist=["PM-01"]),
                    control("ABC", "ABC-02", "Gone control.", orphaned=True),
                ],
            },
            {"id": "xyz", "title": "Zed Domain", "controls": [control("XYZ", "XYZ-01", "Review logs.")]},
        ],
    }
}


def test_parse_drops_orphans_groups_subcontrols_strips_ids_and_extracts_mappings():
    records, mappings, stats = scf.parse_scf_catalog(FIXTURE)
    assert [r["id"] for r in records] == ["ABC-01", "ABC-01.1", "XYZ-01"]
    assert stats["orphaned_dropped"] == 1 and stats["scf_version"] == "9.9"
    sub = records[1]
    assert sub["base_id"] == "ABC-01" and sub["is_enhancement"] and sub["family"] == "ABC"
    assert records[0]["text"] == "Maintain the inventory. See [control] and [control] for details."
    assert mappings["ABC-01"] == ["AC-02(01)", "PM-01"]  # only the NIST class, other frameworks ignored
    assert mappings["XYZ-01"] == []


def test_parse_rejects_label_outside_its_group():
    bad = json.loads(json.dumps(FIXTURE))
    bad["catalog"]["groups"][1]["controls"][0]["props"][0]["value"] = "ABC-09"
    with pytest.raises(ValueError, match="not in group"):
        scf.parse_scf_catalog(bad)


def test_build_writes_only_to_given_dir_and_keeps_base_controls_together(tmp_path):
    catalog = tmp_path / "in" / "catalog.json"
    catalog.parent.mkdir()
    catalog.write_text(json.dumps(FIXTURE))
    out = tmp_path / "out"
    m = scf.build(catalog, out)
    assert {p.name for p in out.iterdir()} == {"train.jsonl", "test.jsonl", "split.json", "nist_r5_2_mappings.json", "manifest.json"}
    assert m["counts"]["records"] == 3 and m["source"]["file_name"] == "catalog.json"
    assert "catalog.json" in m["source"]["file_name"] and str(tmp_path) not in json.dumps(m)  # no absolute paths recorded
    rows = [json.loads(l) for f in ("train", "test") for l in (out / f"{f}.jsonl").read_text().splitlines()]
    sides = {}
    for f in ("train", "test"):
        for l in (out / f"{f}.jsonl").read_text().splitlines():
            sides.setdefault(json.loads(l)["base_id"], set()).add(f)
    assert all(len(v) == 1 for v in sides.values()) and len(rows) == 3


def test_scriptorium_home_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIPTORIUM_HOME", str(tmp_path))
    assert paths.scf_dir() == tmp_path / "scf"


def test_normalize_nist_ids():
    n = crosswalk.normalize_nist_id
    assert n("PM-01") == "PM-1" and n("AC-02(01)") == "AC-2(1)" and n("SC-12") == "SC-12" and n("AC-2(10)") == "AC-2(10)"
    assert n("AC-2.a") is None and n("garbage") is None


def test_normalized_mapping_reports_join_losses():
    raw = {"S-1": ["PM-01", "AC-02(01)", "ZZ-99", "bad"], "S-2": []}
    truth, stats = crosswalk.normalized_mapping(raw, {"PM-1", "AC-2(1)"})
    assert truth == {"S-1": {"PM-1", "AC-2(1)"}}
    assert stats == {"pairs": 4, "unparseable": 1, "not_in_target_catalog": 1}
    assert crosswalk.invert(truth) == {"PM-1": {"S-1"}, "AC-2(1)": {"S-1"}}


def test_random_baseline_matches_hand_calculation():
    assert crosswalk.random_hit_at_k(10, 1, 1) == pytest.approx(0.1)
    assert crosswalk.random_hit_at_k(10, 2, 1) == pytest.approx(0.2)
    assert crosswalk.random_hit_at_k(4, 1, 4) == 1.0


def test_score_ranks_matching_text_first_and_shuffled_truth_scores_low():
    topics = ["encrypt network traffic tunnel", "review audit log events", "backup restore recovery plan", "badge physical door access"]
    targets = [{"id": f"N-{i}", "text": t + " target"} for i, t in enumerate(topics)]
    queries = [{"id": f"Q-{i}", "text": t + " query"} for i, t in enumerate(topics)]
    truth = {f"Q-{i}": {f"N-{i}"} for i in range(4)}
    good = crosswalk.score(queries, targets, truth)
    assert good["hit_at_k"]["1"] == 1.0 and good["mrr"] == 1.0
    wrong = {f"Q-{i}": {f"N-{(i + 1) % 4}"} for i in range(4)}
    assert crosswalk.score(queries, targets, wrong)["hit_at_k"]["1"] == 0.0


def test_guardrail_no_scf_data_is_tracked_in_git():
    root = ingest.ROOT
    try:
        files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert not [f for f in files if f.startswith("data/scf/")]
    assert not [f for f in files if "oscal-catalog-20" in f or f.startswith("scf-full")]
    assert not [f for f in files if f.endswith((".xlsx", ".xls"))], "framework workbooks must never be tracked"
    markers = ["User-supplied SCF input", "Built from user-supplied files; local-only", "local-only: never commit this file or content derived from it"]
    for f in files:
        if f.endswith((".json", ".jsonl")) and (root / f).exists():
            text = (root / f).read_text(encoding="utf-8", errors="ignore")
            for marker in markers:
                assert marker not in text, (f, marker)
