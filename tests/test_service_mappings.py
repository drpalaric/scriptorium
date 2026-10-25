"""Published-mapping lookups through the service and the MCP server, using synthetic files only."""

import asyncio
import json

import pytest

from conftest import crosswalk_files, write
from scriptorium import crosswalks
from scriptorium import service as svc_mod
from scriptorium.crosswalks import MappingStore
from scriptorium.service import CatalogSource, Service, ServiceError

mcp = pytest.importorskip("mcp")


def nist_rows():
    rows = []
    for i, (cid, fam) in enumerate([("CM-8", "CM"), ("CM-8(1)", "CM"), ("CM-8(3)", "CM"), ("AC-1", "AC"), ("AU-1", "AU"), ("SC-1", "SC")] * 4):
        n = f"{cid}-{i}" if i >= 6 else cid
        base = cid.split("(")[0] if i < 6 else f"{fam}-{20 + i}"
        rows.append({"id": n, "base_id": base, "family": fam, "family_title": fam, "title": f"t {n}", "is_enhancement": "(" in cid, "text": f"{fam} text {i} audit account encrypt"})
    return rows


@pytest.fixture()
def env(tmp_path):
    rows = nist_rows()
    d = tmp_path / "nist"
    d.mkdir()
    (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:18]))
    (d / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[18:]))
    crosswalks.build(crosswalk_files(tmp_path), out_dir=tmp_path / "store")
    store = MappingStore.load(tmp_path / "store")
    return Service({"nist": CatalogSource("nist", d)}, None, store), Service({"nist": CatalogSource("nist", d)}, None, None)


def test_nist_control_gets_published_pairs_with_oriented_relationships_and_labelled_indirect_techniques(env):
    svc, _ = env
    r = svc.published_mappings("nist-800-53r5", "cm-8")
    assert r["framework"] == "nist-800-53r5" and r["element"]["id"] == "CM-8" and r["element"]["title"] == "System Component Inventory"
    assert {p["framework"] for p in r["published"]} == {"cis-8.1", "iso-27001-2022"} and all(p["kind"] == "published" for p in r["published"])
    cis = next(p for p in r["published"] if p["framework"] == "cis-8.1")
    assert cis["id"] == "1.1" and cis["relationship_stated"] == "subset" and cis["relationship_for_this_element"] == "superset"
    assert r["indirect_attack"]["techniques"][0]["kind"] == "indirect" and "not an official" in r["indirect_attack"]["note"]
    assert r["sources"] == ["cis-nist-mapping", "nist-iso27001"] and "Indirect = composed" in r["disclaimer"]


def test_a_cis_safeguard_maps_to_nist_and_attack_directly_with_no_indirect_section(env):
    svc, _ = env
    r = svc.published_mappings("cis-8.1", "1.2")
    assert {(p["framework"], p["id"]) for p in r["published"]} == {("nist-800-53r5", "CM-8(3)"), ("attack-19.1", "T1200"), ("attack-19.1", "T1078")}
    assert r["indirect_attack"] is None and r["element"]["title"] == "Unauthorized"
    assert [p["id"] for p in svc.published_mappings("csf-2.0", "GV.OC-01")["published"]] == ["BCR-01", "BCR-07"]


def test_without_a_built_store_the_error_says_how_to_build_one(env):
    _, bare = env
    with pytest.raises(ServiceError, match="crosswalks build"):
        bare.published_mappings("nist-800-53r5", "CM-8")
    assert bare.list_catalogs()["published_mapping_frameworks"] == []
    assert bare.map_control("CM-8", "nist", "nist", 2)["published_cross_framework"] is None


@pytest.mark.parametrize("fw,eid", [("nope", "AC-1"), ("nist-800-53r5", "AC-1; DROP TABLE"), ("nist-800-53r5", "x" * 50), ("nist-800-53r5", 5), (None, "AC-1"), ("../etc", "AC-1")])
def test_published_mapping_inputs_are_validated_and_not_echoed(env, fw, eid):
    svc, _ = env
    with pytest.raises(ServiceError) as e:
        svc.published_mappings(fw, eid)
    assert "DROP" not in str(e.value) and "etc" not in str(e.value)


def test_results_are_capped_but_the_true_total_is_reported(env, monkeypatch):
    svc, _ = env
    monkeypatch.setattr(svc_mod, "MAX_PUBLISHED", 1)
    r = svc.published_mappings("nist-800-53r5", "CM-8")
    assert len(r["published"]) == 1 and r["published_total"] == 3


def test_map_control_includes_cross_framework_pairs_for_nist_controls(env):
    svc, _ = env
    r = svc.map_control("CM-8", "nist", "nist", 2)
    cf = r["published_cross_framework"]
    assert cf["published_total"] == 3 and cf["indirect_attack"]["total"] == 1 and "Published =" in cf["disclaimer"]
    assert "CM-8" not in [m["id"] for m in r["similarity_matches"]]  # unchanged behaviour
    assert svc.list_catalogs()["published_mapping_frameworks"] == ["attack-19.1", "ccm-4.0", "cis-8.1", "csf-2.0", "iso-27001-2022", "nist-800-53r5"]


def test_mcp_tool_returns_labelled_results_and_safe_errors(env):
    from scriptorium.mcp_server import build_server

    svc, _ = env

    async def go(name, args):
        async with mcp.Client(build_server(svc)) as c:
            return await c.call_tool(name, args)

    ok = asyncio.run(go("get_published_mappings", {"framework": "nist-800-53r5", "element_id": "CM-8"}))
    assert not ok.is_error and ok.structured_content["published_total"] == 3
    bad = asyncio.run(go("get_published_mappings", {"framework": "nist-800-53r5", "element_id": "AC-1; DROP"}))
    assert bad.is_error and "DROP" not in json.dumps([c.model_dump() for c in bad.content])
