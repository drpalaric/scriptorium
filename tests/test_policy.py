"""Tests for scriptorium.policy: the OPA runner, rule metadata validation, evidence verification, audit export."""

import hashlib
import json
import shutil
import sqlite3
import subprocess

import pytest

from scriptorium import ingest, policy
from scriptorium.db import Database, append_event
from scriptorium.policy import PolicyError

import os


def test_opa_is_present_whenever_ci_requires_it():
    """CI sets REQUIRE_OPA=1: a missing binary must fail the run instead of silently skipping the policy tests."""
    if os.environ.get("REQUIRE_OPA") == "1":
        assert shutil.which("opa"), "REQUIRE_OPA is set but opa is not on PATH"


needs_opa = pytest.mark.skipif(shutil.which("opa") is None, reason="opa is not installed")


# ---- metadata validation (pure Python) -----------------------------------------------------------------------


def good_rule(**kw):
    return {"package": "p", "id": "X-1", "title": "t", "controls": ["AC-2"], "mapping_basis": "author's judgment", **kw}


KNOWN = {"AC-2", "AU-3", "CM-8"}


def test_valid_rules_pass_and_each_defect_is_reported():
    assert policy.validate_controls([good_rule()], KNOWN) == []
    assert any("no custom.id" in p for p in policy.validate_controls([good_rule(id=None)], KNOWN))
    assert any("mapping_basis" in p for p in policy.validate_controls([good_rule(mapping_basis=None)], KNOWN))
    assert any("no controls" in p for p in policy.validate_controls([good_rule(controls=[])], KNOWN))
    assert any("no controls" in p for p in policy.validate_controls([good_rule(controls="AC-2")], KNOWN))
    assert any("not a NIST control ID" in p for p in policy.validate_controls([good_rule(controls=["AC2"])], KNOWN))
    assert any("not an active control" in p for p in policy.validate_controls([good_rule(controls=["ZZ-99"])], KNOWN))
    assert any("duplicate" in p for p in policy.validate_controls([good_rule(), good_rule()], KNOWN))
    assert policy.validate_controls([good_rule(controls=["AC-2", "AU-3"])], KNOWN) == []
    assert any("not a NIST control ID" in p for p in policy.validate_controls([good_rule(controls=[7])], KNOWN))


@needs_opa
def test_the_repositorys_own_policies_have_valid_metadata_and_real_nist_controls():
    rules = policy.rule_controls()
    assert len(rules) >= 16 and {r["package"] for r in rules} == {"scriptorium.change_control", "scriptorium.ai_system"}
    assert policy.validate_controls(rules, policy.known_nist_ids()) == []
    assert all(r["mapping_basis"] and "judgment" in r["mapping_basis"] for r in rules)


@needs_opa
def test_the_rego_files_are_formatted_type_checked_and_their_own_tests_pass_with_full_coverage():
    d = str(policy.POLICY_DIR)
    assert subprocess.run(["opa", "fmt", "--fail", "-l", d], capture_output=True).returncode == 0
    assert subprocess.run(["opa", "check", "--strict", d], capture_output=True).returncode == 0
    r = subprocess.run(["opa", "test", d, "--coverage", "--threshold", "100", "-f", "json"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:]


# ---- evidence verification ------------------------------------------------------------------------------------


def test_evidence_is_verified_only_for_files_inside_the_root_whose_hash_matches(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "r.json").write_text("{}")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    good = hashlib.sha256(b"{}").hexdigest()
    systems = [{"evaluation_evidence": [
        {"report": "r.json", "sha256": good},
        {"report": "r.json", "sha256": "0" * 64},
        {"report": "missing.json", "sha256": good},
        {"report": "../secret.txt", "sha256": hashlib.sha256(b"secret").hexdigest()},
        {"report": str(outside), "sha256": hashlib.sha256(b"secret").hexdigest()},
        {"report": None, "sha256": good},
        {"report": "r.json", "sha256": 5},
        "not a dict",
    ]}, {"evaluation_evidence": "not a list"}, {}]
    opened = []
    real = policy.sources.sha256_file
    monkeypatch.setattr(policy.sources, "sha256_file", lambda p: (opened.append(p), real(p))[1])
    policy.verify_evidence(systems, root)
    flags = [e.get("verified") for e in systems[0]["evaluation_evidence"][:7]]
    assert flags == [True, False, False, False, False, False, False]
    assert all(str(p).startswith(str(root.resolve())) for p in opened)  # nothing outside the root was ever hashed


# ---- inputs ---------------------------------------------------------------------------------------------------


def test_system_files_are_validated_before_anything_runs(tmp_path):
    f = tmp_path / "s.json"
    for body, msg in [("not json", "valid JSON"), ("[]", '"systems" list'), ('{"systems": "x"}', '"systems" list'), ('{"systems": [1]}', "JSON object")]:
        f.write_text(body)
        with pytest.raises(PolicyError, match=msg):
            policy.check_systems(f)
    f.write_text(json.dumps({"systems": [{}] * 5001}))
    with pytest.raises(PolicyError, match="too many"):
        policy.check_systems(f)
    f.write_text("x" * (policy.MAX_INPUT_BYTES + 1))
    with pytest.raises(PolicyError, match="too large"):
        policy.check_systems(f)


def test_a_missing_opa_binary_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(policy.shutil, "which", lambda name: None)
    with pytest.raises(PolicyError, match="not found on PATH"):
        policy.evaluate("data.x", {})


@needs_opa
def test_opa_errors_become_safe_messages_and_a_wrong_query_is_reported():
    with pytest.raises(PolicyError, match="opa failed"):
        policy.evaluate("data.scriptorium.change_control.report[", {})
    with pytest.raises(PolicyError, match="no result"):
        policy.evaluate("data.scriptorium.nope.report", {})


# ---- the audit log ----------------------------------------------------------------------------------------------


def feedback_event(fid, event, actor, **extra):
    return {"event": event, "id": fid, "ts": "2026-10-25T00:00:00+00:00", "actor": actor, **extra}


def write_events(db, events):
    with db.transaction() as conn:
        for e in events:
            append_event(conn, "feedback", e["event"], e, catalog="nist", entity_id=e["id"], actor=e.get("actor") or e.get("submitter"), ts=e["ts"])


CLEAN = [
    {"event": "submitted", "id": "fb_000000000001", "ts": "2026-10-25T00:00:00+00:00", "submitter": "user:alice", "text": "t", "label": "AC"},
    feedback_event("fb_000000000001", "approved", "admin:phil", reason="ok"),
    feedback_event("fb_000000000001", "applied", "admin:phil", gate={"passed": True}, forced=False),
]


def test_export_has_parsed_payloads_the_chain_result_and_the_option(tmp_path):
    db = Database(tmp_path / "x.db")
    write_events(db, CLEAN)
    d = policy.export_audit(db, single_operator=True)
    assert [e["event"] for e in d["events"]] == ["submitted", "approved", "applied"] and d["events"][2]["payload"]["gate"] == {"passed": True}
    assert d["events"][0]["actor"] == "user:alice" and d["chain"]["ok"] is True and d["options"] == {"single_operator": True}
    assert policy.export_audit(Database(tmp_path / "none.db"))["events"] == []


@needs_opa
def test_a_clean_log_passes_and_separation_of_duties_is_found_then_excepted(tmp_path):
    db = Database(tmp_path / "a.db")
    write_events(db, CLEAN)
    assert policy.check_audit(db)["violations"] == []
    same = [dict(CLEAN[0], submitter="cli:alger"), dict(CLEAN[1], actor="cli:alger"), dict(CLEAN[2], actor="cli:alger")]
    db2 = Database(tmp_path / "b.db")
    write_events(db2, same)
    r = policy.check_audit(db2)
    assert {v["rule"] for v in r["violations"]} == {"CC-SOD"} and r["violations"][0]["controls"] == ["AC-5"]
    r2 = policy.check_audit(db2, single_operator=True)
    assert r2["violations"] == [] and {e["rule"] for e in r2["exceptions"]} == {"CC-SOD"}


@needs_opa
def test_a_tampered_audit_log_is_reported_as_a_chain_violation(tmp_path):
    db = Database(tmp_path / "t.db")
    write_events(db, CLEAN)
    raw = sqlite3.connect(db.path)
    for t in ("events_no_update", "events_no_delete"):
        raw.execute(f"DROP TRIGGER IF EXISTS {t}")
    raw.execute("UPDATE events SET actor='someone-else' WHERE seq=1")
    raw.commit()
    raw.close()
    assert "CC-CHAIN" in {v["rule"] for v in policy.check_audit(db)["violations"]}


@needs_opa
def test_the_command_line_exit_codes_and_output(tmp_path, capsys):
    db = Database(policy.paths.db_path())  # SCRIPTORIUM_HOME is isolated per test
    write_events(db, CLEAN)
    assert policy.main(["check-audit"]) == 0 and "no findings" in capsys.readouterr().out
    assert policy.main(["controls", "--no-mappings"]) == 0 and "control validation: ok" in capsys.readouterr().out
    assert policy.main(["check-systems", str(ingest.ROOT / "examples" / "ai_systems.json"), "--today", "2026-10-25"]) == 0
    assert policy.main(["check-systems", str(ingest.ROOT / "examples" / "ai_systems_noncompliant.json"), "--today", "2026-10-25"]) == 1
    out = capsys.readouterr().out
    assert "FINDING AI-" in out and "controls" in out
    assert policy.main(["check-systems", str(tmp_path / "missing.json")]) == 2


# ---- the examples behave as documented ---------------------------------------------------------------------------


@needs_opa
def test_this_projects_own_system_records_have_no_findings_and_their_evidence_verifies():
    r = policy.check_systems(ingest.ROOT / "examples" / "ai_systems.json", today="2026-10-25")
    assert r["systems_checked"] == 2 and r["findings"] == []


@needs_opa
def test_the_fictional_non_compliant_systems_trip_every_system_rule():
    r = policy.check_systems(ingest.ROOT / "examples" / "ai_systems_noncompliant.json", today="2026-10-25")
    assert {f["rule"] for f in r["findings"]} == {"AI-EVIDENCE", "AI-EXTERNAL-APPROVED", "AI-INVENTORY", "AI-LOGGING", "AI-OVERSIGHT", "AI-PERSONAL-DATA", "AI-RESTRICTED-EXTERNAL", "AI-REVIEW"}
    assert all(f["controls"] for f in r["findings"])
