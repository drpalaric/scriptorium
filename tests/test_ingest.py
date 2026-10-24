"""Tests for scriptorium.ingest.

Fixture tests exercise the parser on a tiny synthetic OSCAL catalog. Data tests check
the committed outputs against data/manifest.json. The rebuild test needs the raw
catalog (`python -m scriptorium.ingest fetch`) and is skipped without it.
"""

import json
import re

import pytest

from scriptorium import ingest

FIXTURE = {
    "catalog": {
        "metadata": {"version": "x", "oscal-version": "1.0.0"},
        "groups": [
            {
                "id": "ac",
                "title": "Access Control",
                "controls": [
                    {
                        "id": "ac-1",
                        "title": "Policy",
                        "props": [{"name": "label", "value": "AC-1"}],
                        "params": [{"id": "ac-01_odp", "label": "personnel"}],
                        "parts": [
                            {
                                "name": "statement",
                                "prose": "Send policy to {{ insert: param, ac-01_odp }} and review per AC-2(1) and [AC-3](#ac-3) .",
                            }
                        ],
                        "controls": [
                            {
                                "id": "ac-1.1",
                                "title": "Enh",
                                "props": [{"name": "label", "value": "AC-1(1)"}],
                                "parts": [{"name": "statement", "prose": "Do the enhanced thing."}],
                            },
                            {
                                "id": "ac-1.2",
                                "title": "Gone",
                                "props": [
                                    {"name": "label", "value": "AC-1(2)"},
                                    {"name": "status", "value": "withdrawn"},
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }
}


def _make_records(n_bases: int, n_families: int = 2):
    fams = ["AC", "AU", "SC"][:n_families]
    return [
        {"id": f"{f}-{b}{suffix}", "base_id": f"{f}-{b}", "family": f, "text": "t"}
        for f in fams
        for b in range(1, n_bases + 1)
        for suffix in ("", "(1)", "(2)")
    ]


def test_parse_fixture_resolves_params_strips_ids_and_drops_withdrawn():
    records, stats = ingest.parse_catalog(FIXTURE)
    assert [r["id"] for r in records] == ["AC-1", "AC-1(1)"]
    assert stats["withdrawn_dropped"] == 1
    assert records[0]["text"] == "Send policy to personnel and review per [control] and [[control]]."
    assert not ingest._ID_RE.search(records[0]["text"])
    assert records[1]["base_id"] == "AC-1" and records[1]["is_enhancement"]


def test_split_is_deterministic_grouped_and_covers_everything():
    records = _make_records(10)
    a = ingest.split_bases(records)
    b = ingest.split_bases(list(reversed(records)))
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert set(a["train"]) | set(a["test"]) == {r["base_id"] for r in records}
    for fam in ("AC", "AU"):
        assert any(x.startswith(fam) for x in a["test"])


def test_split_changes_with_seed():
    records = _make_records(20)
    assert ingest.split_bases(records, seed=1) != ingest.split_bases(records, seed=2)


@pytest.fixture(scope="module")
def data():
    manifest = json.loads(ingest.MANIFEST_PATH.read_text())
    load = lambda n: [json.loads(l) for l in (ingest.PROCESSED_DIR / n).read_text().splitlines()]
    return manifest, load("train.jsonl"), load("test.jsonl")


def test_manifest_pins_source_and_hashes_outputs(data):
    manifest, _, _ = data
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source"]["commit"])
    assert manifest["source"]["commit"] == ingest.SOURCE_COMMIT
    assert manifest["source"]["sha256"] == ingest.SOURCE_SHA256
    for name, digest in manifest["outputs"].items():
        assert ingest.sha256_file(ingest.PROCESSED_DIR / name) == digest, name


def test_counts_match_manifest(data):
    manifest, train, test = data
    c = manifest["counts"]
    assert len(train) == c["train"] and len(test) == c["test"]
    assert len(train) + len(test) == c["records"]
    assert len({r["id"] for r in train + test}) == c["records"]
    assert {r["family"] for r in train + test} == set(ingest.FAMILY_CODES)
    assert sum(c["train_by_family"].values()) == c["train"]
    assert sum(c["test_by_family"].values()) == c["test"]


def test_no_base_control_leaks_across_split(data):
    _, train, test = data
    assert not {r["base_id"] for r in train} & {r["base_id"] for r in test}


def test_every_family_present_in_both_splits(data):
    _, train, test = data
    assert {r["family"] for r in train} == {r["family"] for r in test} == set(ingest.FAMILY_CODES)


def test_text_has_no_control_ids_and_labels_are_consistent(data):
    _, train, test = data
    for r in train + test:
        assert not ingest._ID_RE.search(r["text"]), r["id"]
        assert r["text"].strip()
        assert r["id"].startswith(r["family"] + "-")


def test_rebuild_reproduces_committed_outputs(data, tmp_path):
    if not ingest.RAW_PATH.exists():
        pytest.skip("raw catalog not fetched; run `python -m scriptorium.ingest fetch`")
    manifest, _, _ = data
    rebuilt = ingest.build(out_dir=tmp_path, manifest_path=tmp_path / "manifest.json")
    assert rebuilt["outputs"] == manifest["outputs"]
    assert rebuilt["counts"] == manifest["counts"]
