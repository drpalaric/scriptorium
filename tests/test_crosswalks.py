"""Tests for scriptorium.crosswalks using small synthetic workbooks and JSON (no real framework content)."""

import json
import stat

import pytest

from scriptorium import crosswalks, sources
from scriptorium.crosswalks import CrosswalkError, MappingStore

openpyxl = pytest.importorskip("openpyxl")


from conftest import crosswalk_files, xlsx  # noqa: E402


@pytest.fixture()
def files(tmp_path):
    return crosswalk_files(tmp_path)


@pytest.fixture()
def store(files, tmp_path):
    crosswalks.build(files, out_dir=tmp_path / "store")
    return MappingStore.load(tmp_path / "store")


def test_cis_controls_adapter_reads_safeguards_and_control_headers(files):
    els, maps = crosswalks.read_cis_controls(files["cis-controls"])
    assert maps == [] and {e["id"] for e in els} == {"1", "1.1", "1.2"}
    e = next(e for e in els if e["id"] == "1.1")
    assert e["title"] == "Inventory" and e["text"].startswith("Keep an inventory") and e["meta"]["control"] == "1"


def test_cis_mapping_adapter_keeps_relationships_and_skips_header_rows(files):
    els, maps = crosswalks.read_cis_mapping(files["cis-nist-mapping"])
    assert [(m["from_id"], m["to_id"], m["relationship"]) for m in maps] == [("1.1", "CM-8", "subset"), ("1.1", "CM-8(1)", "superset"), ("1.2", "CM-8(3)", "equivalent")]
    assert {e["id"] for e in els} == {"CM-8", "CM-8(1)", "CM-8(3)"}


def test_olir_adapter_normalises_nist_ids_and_handles_both_template_layouts(files):
    _, iso = crosswalks.read_olir(files["nist-iso27001"], "nist-800-53r5", "iso-27001-2022", "nist-iso27001")
    assert [(m["from_id"], m["to_id"]) for m in iso] == [("CM-8", "A.5.9"), ("CM-8(1)", "A.5.9"), ("CM-8", "7.5.1")] and all(m["relationship"] is None for m in iso)
    els, ccm = crosswalks.read_olir(files["ccm-csf"], "csf-2.0", "ccm-4.0", "ccm-csf")
    assert [(m["from_id"], m["to_id"]) for m in ccm] == [("GV.OC-01", "BCR-01"), ("GV.OC-01", "BCR-07")]
    assert [e["id"] for e in els] == ["GV.OC-01"] and els[0]["text"] == "The mission is understood"


def test_ctid_adapter_drops_non_mappable_and_keeps_rationale(files):
    els, maps = crosswalks.read_ctid(files["cis-attack"])
    assert [(m["from_id"], m["to_id"], m["relationship"]) for m in maps] == [("1.1", "T1200", "mitigates"), ("1.2", "T1200", "mitigates"), ("1.2", "T1078", "mitigates")]
    assert maps[0]["rationale"] == "why" and maps[2]["rationale"] is None
    assert {e["id"] for e in els if e["framework"] == "attack-19.1"} == {"T1200", "T1078"}


def test_wrong_files_fail_loudly_with_a_useful_message(tmp_path):
    bad = xlsx(tmp_path / "bad.xlsx", {"Sheet1": [["a", "b"]]})
    for fn in (crosswalks.read_cis_controls, crosswalks.read_cis_mapping):
        with pytest.raises(CrosswalkError, match="no sheet named"):
            fn(bad)
    with pytest.raises(CrosswalkError, match="OLIR"):
        crosswalks.read_olir(bad, "a", "b", "x")
    wrong_cols = xlsx(tmp_path / "w.xlsx", {"Controls v8.1": [["CIS Control", "Foo"]]})
    with pytest.raises(CrosswalkError, match="missing"):
        crosswalks.read_cis_controls(wrong_cols)
    (tmp_path / "x.json").write_text("[1,2]")
    with pytest.raises(CrosswalkError, match="mapping_objects"):
        crosswalks.read_ctid(tmp_path / "x.json")
    notxlsx = tmp_path / "n.xlsx"
    notxlsx.write_bytes(b"nope")
    with pytest.raises(CrosswalkError, match="could not be read"):
        crosswalks.read_cis_controls(notxlsx)


def test_build_registers_sources_writes_private_files_and_a_manifest_without_content(files, tmp_path):
    m = crosswalks.build(files, out_dir=tmp_path / "store")
    assert set(m["sources"]) == set(files) and m["totals"]["mappings"] == 3 + 3 + 2 + 3
    d = tmp_path / "store"
    for f in ("elements.jsonl", "mappings.jsonl", "manifest.json"):
        assert stat.S_IMODE((d / f).stat().st_mode) == 0o600
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    manifest = (d / "manifest.json").read_text()
    assert "Keep an inventory" not in manifest and str(tmp_path) not in manifest and "local-only" in manifest
    assert set(sources.registered()) == set(files)


def test_build_refuses_a_file_that_changed_after_registration(files, tmp_path):
    crosswalks.build(files, out_dir=tmp_path / "store")
    files["cis-attack"].write_text(json.dumps({"mapping_objects": []}))
    with pytest.raises(sources.SourceError, match="hash differs"):
        crosswalks.build(files, out_dir=tmp_path / "store2")
    crosswalks.build(files, out_dir=tmp_path / "store2", accept_change=True)


def test_published_lookups_work_in_both_directions_with_oriented_relationships(store):
    cis = store.published("cis-8.1", "1.1")
    assert [(x["framework"], x["id"], x["relationship_stated"]) for x in cis] == [("attack-19.1", "T1200", "mitigates"), ("nist-800-53r5", "CM-8", "subset"), ("nist-800-53r5", "CM-8(1)", "superset")]
    assert all(x["kind"] == "published" for x in cis)
    nist = {x["id"]: x for x in store.published("nist-800-53r5", "cm-8(1)")}  # case and padding are normalised
    assert nist["1.1"]["relationship_stated"] == "superset" and nist["1.1"]["relationship_for_this_element"] == "subset"
    assert {x["id"] for x in store.published("nist-800-53r5", "CM-08")} == {"1.1", "A.5.9", "7.5.1"}
    assert nist["1.1"]["title"] == "Inventory"
    assert {x["id"] for x in store.published("csf-2.0", "gv.oc-01")} == {"BCR-01", "BCR-07"}
    assert [x["id"] for x in store.published("ccm-4.0", "bcr-01")] == ["GV.OC-01"]
    assert store.published("nist-800-53r5", "AC-1") == []


def test_indirect_attack_is_labelled_and_composed_from_published_links(store):
    r = store.indirect_attack("nist-800-53r5", "CM-8")
    assert [t["id"] for t in r["techniques"]] == ["T1200"] and r["total"] == 1 and "not an official" in r["note"]
    t = r["techniques"][0]
    assert t["kind"] == "indirect" and t["via"] == [{"cis_safeguard": "1.1", "cis_to_nist_relationship": "subset", "cis_to_attack": "mitigates"}]
    two = store.indirect_attack("nist-800-53r5", "CM-8(3)")  # 1.2 maps to two techniques
    assert {t["id"] for t in two["techniques"]} == {"T1200", "T1078"}
    assert store.indirect_attack("cis-8.1", "1.1") == {"techniques": [], "total": 0}


@pytest.mark.parametrize("fw,eid", [("nope", "AC-1"), ("nist-800-53r5", "AC-1; DROP"), ("nist-800-53r5", "x" * 40), ("nist-800-53r5", ""), ("nist-800-53r5", None), (None, "AC-1")])
def test_lookup_validates_input(store, fw, eid):
    with pytest.raises(CrosswalkError):
        store.published(fw, eid)


def test_store_loads_none_when_nothing_was_built(tmp_path):
    assert MappingStore.load(tmp_path / "empty") is None
