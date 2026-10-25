"""Tests for scriptorium.feedback: lifecycle, the held-out gate, poisoning, leakage, integrity."""

import json
import stat

import pytest

from conftest import rows_for, write
from scriptorium import feedback as fb
from scriptorium.feedback import FeedbackError, FeedbackStore
from scriptorium.service import CatalogSource, Service, ServiceError


@pytest.fixture()
def env(tmp_path):
    write(tmp_path / "nist", rows_for({"AC": "AC", "AU": "AU", "SC": "SC"}))
    store = FeedbackStore(tmp_path / "home" / "feedback")
    svc = Service({"nist": CatalogSource("nist", tmp_path / "nist")}, store)
    return svc, store, tmp_path


def approve_all(store, ids):
    for i in ids:
        fb.transition(store, "nist", i, "approved", "admin")


def test_submit_returns_receipt_echoing_what_was_stored(env):
    svc, store, _ = env
    r = fb.submit(svc, store, "nist", "  badge door camera visitor  ", "sc", submitter="alice")
    assert r["status"] == "pending" and r["label"] == "SC" and r["text"] == "badge door camera visitor"
    assert r["catalog"] == "nist" and r["submitter"] == "alice" and r["predicted"] in {"AC", "AU", "SC"}
    assert r["id"].startswith("fb_") and len(r["id"]) == 15 and r["duplicate"] is False
    assert store.entries("nist")[r["id"]]["status"] == "pending"


def test_submit_validation(env):
    svc, store, _ = env
    with pytest.raises(FeedbackError, match="label must be one of"):
        fb.submit(svc, store, "nist", "some text", "ZZ")
    with pytest.raises(FeedbackError):
        fb.submit(svc, store, "nist", "some text", None)
    with pytest.raises(ServiceError):
        fb.submit(svc, store, "../nist", "some text", "AC")
    with pytest.raises(ServiceError):
        fb.submit(svc, store, "nist", "x" * 5000, "AC")


def test_duplicate_pending_submission_returns_existing_receipt(env):
    svc, store, _ = env
    a = fb.submit(svc, store, "nist", "Badge door camera", "SC")
    b = fb.submit(svc, store, "nist", "badge   DOOR camera", "SC")
    assert b["id"] == a["id"] and b["duplicate"] is True
    assert len(store.entries("nist")) == 1


def test_pending_cap(env, monkeypatch):
    svc, store, _ = env
    monkeypatch.setattr(fb, "MAX_PENDING", 2)
    fb.submit(svc, store, "nist", "one", "AC")
    fb.submit(svc, store, "nist", "two", "AC")
    with pytest.raises(FeedbackError, match="too many pending"):
        fb.submit(svc, store, "nist", "three", "AC")


def test_pending_and_approved_entries_never_change_the_model(env):
    svc, store, _ = env
    v0 = svc._model("nist").version
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "SC")
    assert svc._model("nist").version == v0
    approve_all(store, [r["id"]])
    assert svc._model("nist").version == v0


def test_withdraw_pending_is_never_applied(env):
    svc, store, _ = env
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "SC")
    fb.transition(store, "nist", r["id"], "withdrawn", "alice")
    out = fb.apply(svc, store, "nist")
    assert out["applied"] == [] and store.entries("nist")[r["id"]]["status"] == "withdrawn"


@pytest.mark.parametrize("event,setup", [("approved", "approved"), ("withdrawn", "approved"), ("approved", "withdrawn")])
def test_illegal_transitions_are_refused(env, event, setup):
    svc, store, _ = env
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "SC")
    fb.transition(store, "nist", r["id"], setup, "admin")
    with pytest.raises(FeedbackError, match="cannot mark"):
        fb.transition(store, "nist", r["id"], event, "admin")


def test_bad_ids_and_unknown_ids(env):
    _, store, _ = env
    for bad in ["../x", "fb_XYZ", "", None, "fb_" + "0" * 12 + "; rm"]:
        with pytest.raises(FeedbackError):
            fb.transition(store, "nist", bad, "approved", "admin")
    with pytest.raises(FeedbackError, match="no feedback entry"):
        fb.transition(store, "nist", "fb_" + "0" * 12, "approved", "admin")


def test_apply_changes_version_reports_gate_and_surfaces_exact_match(env):
    svc, store, _ = env
    v0 = svc._model("nist").version
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "SC")
    approve_all(store, [r["id"]])
    out = fb.apply(svc, store, "nist", actor="admin")
    assert out["gate"]["passed"] is True and set(out["gate"]["before"]) == {"accuracy", "macro_f1"}
    assert [a["id"] for a in out["applied"]] == [r["id"]] and "now_correct" in out["applied"][0]
    assert svc._model("nist").version != v0
    c = svc.classify("Badge door CAMERA visitor")
    assert c["applied_correction"] == {"id": r["id"], "label": "SC"}
    assert svc.classify("account access")["applied_correction"] is None


def test_rollback_restores_original_model_exactly(env):
    svc, store, _ = env
    probe = ["account access privilege", "audit log record", "badge door camera visitor"]
    v0 = svc._model("nist").version
    p0 = [svc.classify(t)["predictions"] for t in probe]
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "AU")
    approve_all(store, [r["id"]])
    fb.apply(svc, store, "nist")
    assert svc._model("nist").version != v0
    fb.rollback(store, "nist", [r["id"]], reason="wrong label")
    assert svc._model("nist").version == v0
    assert [svc.classify(t)["predictions"] for t in probe] == p0
    assert store.entries("nist")[r["id"]]["status"] == "rolled_back"
    with pytest.raises(FeedbackError, match="cannot roll back"):
        fb.rollback(store, "nist", [r["id"]])


def test_reset_rolls_back_everything_and_keeps_history(env):
    svc, store, _ = env
    v0 = svc._model("nist").version
    ids = [fb.submit(svc, store, "nist", f"badge door visitor {i}", "SC")["id"] for i in range(3)]
    approve_all(store, ids)
    fb.apply(svc, store, "nist")
    out = fb.reset(store, "nist")
    assert sorted(out["rolled_back"]) == sorted(ids) and svc._model("nist").version == v0
    assert all(len(store.entries("nist")[i]["history"]) == 4 for i in ids)  # submitted, approved, applied, rolled_back


def poison(svc, store, n=30):
    """Broad poisoning: ordinary Access Control wording (no trigger token) relabelled as Audit."""
    from itertools import permutations

    words = "account access privilege authorize login permission".split()
    texts = [" ".join(p) for p in permutations(words, 4)][:n]
    ids = [fb.submit(svc, store, "nist", t, "AU")["id"] for t in texts]
    approve_all(store, ids)
    return ids


def test_broad_poisoning_is_blocked_by_the_held_out_gate(env):
    svc, store, _ = env
    v0 = svc._model("nist").version
    ids = poison(svc, store)
    out = fb.apply(svc, store, "nist")
    assert out["gate"]["passed"] is False and out["applied"] == []
    assert out["gate"]["macro_f1_drop"] > fb.GATE_TOLERANCE
    assert any("relabel AC as AU" in w for w in out["warnings"])
    assert svc._model("nist").version == v0
    assert all(store.entries("nist")[i]["status"] == "approved" for i in ids)


def test_known_limitation_trigger_word_poisoning_passes_the_gate_but_is_flagged(env):
    """A targeted attack that leaves held-out records alone is invisible to the gate.
    This test documents that limit: review is the control, and the warnings point at it."""
    svc, store, _ = env
    ids = [fb.submit(svc, store, "nist", f"account access privilege authorize login permission zqtrigger {i}", "AU")["id"] for i in range(30)]
    approve_all(store, ids)
    out = fb.apply(svc, store, "nist")
    assert out["gate"]["passed"] is True  # the gate cannot see it
    assert any("relabel AC as AU" in w for w in out["warnings"])
    assert any("contradict a confident model prediction" in w for w in out["warnings"])
    assert svc.classify("account access privilege zqtrigger")["predictions"][0]["label"] == "AU"  # the backdoor works


def test_force_override_requires_a_reason_and_is_recorded(env):
    svc, store, _ = env
    poison(svc, store, n=10)
    with pytest.raises(FeedbackError, match="requires a reason"):
        fb.apply(svc, store, "nist", force=True)
    out = fb.apply(svc, store, "nist", force=True, reason="testing override")
    if out["gate"]["passed"]:
        pytest.skip("10 poisoned corrections did not trip the gate on this data")
    assert out["forced"] is True and out["applied"]
    ev = [h for h in store.entries("nist")[out["applied"][0]["id"]]["history"] if h["event"] == "applied"][0]
    assert ev["forced"] is True and ev["reason"] == "testing override"


def test_correction_matching_a_held_out_record_is_rejected_not_trained_on(env):
    svc, store, tmp = env
    test_text = json.loads((tmp / "nist" / "test.jsonl").read_text().splitlines()[0])["text"]
    v0 = svc._model("nist").version
    r = fb.submit(svc, store, "nist", test_text.upper(), "SC")
    approve_all(store, [r["id"]])
    out = fb.apply(svc, store, "nist")
    assert out["rejected_as_test_leak"] == [r["id"]] and out["applied"] == []
    assert store.entries("nist")[r["id"]]["status"] == "rejected"
    assert svc._model("nist").version == v0


def test_test_split_file_is_never_modified(env):
    svc, store, tmp = env
    before = (tmp / "nist" / "test.jsonl").read_bytes()
    ids = [fb.submit(svc, store, "nist", f"badge door visitor {i}", "SC")["id"] for i in range(3)]
    approve_all(store, ids)
    fb.apply(svc, store, "nist")
    assert (tmp / "nist" / "test.jsonl").read_bytes() == before


def test_log_is_append_only_and_private(env):
    svc, store, _ = env
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "SC")
    path = store.root / "nist.jsonl"
    first = path.read_text()
    fb.transition(store, "nist", r["id"], "approved", "admin")
    assert path.read_text().startswith(first) and len(path.read_text().splitlines()) == 2
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "line",
    ["not json", json.dumps({"event": "approved", "id": "fb_000000000000"}), json.dumps({"event": "hack", "id": "fb_000000000000"}), json.dumps({"event": "submitted", "id": "../x"})],
)
def test_corrupt_or_tampered_log_is_refused(env, line):
    svc, store, _ = env
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "nist.jsonl").write_text(line + "\n")
    with pytest.raises(FeedbackError):
        store.entries("nist")


def test_reason_is_length_limited(env):
    svc, store, _ = env
    r = fb.submit(svc, store, "nist", "badge door camera visitor", "SC")
    with pytest.raises(FeedbackError, match="reason"):
        fb.transition(store, "nist", r["id"], "approved", "admin", reason="x" * 600)


def test_apply_weight_bounds(env):
    svc, store, _ = env
    for bad in (0, -1, 101):
        with pytest.raises(FeedbackError, match="weight"):
            fb.apply(svc, store, "nist", weight=bad)
