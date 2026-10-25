"""Tests for scriptorium.auth (tokens), scriptorium.db import and backup."""

import json
import sqlite3
import stat

import pytest

from scriptorium import auth, db as dbmod
from scriptorium.db import Database, DatabaseError


@pytest.fixture()
def database(tmp_path):
    return Database(tmp_path / "home" / "scriptorium.db")


# ---- tokens ------------------------------------------------------------------------------


def test_create_authenticate_and_plaintext_is_never_stored(database):
    rec, token = auth.create_token(database, "user", "alice", "cli:test")
    assert token.startswith("sct_") and rec["role"] == "user" and rec["id"].startswith("tok_")
    p = auth.authenticate(database, token)
    assert (p.name, p.role, p.actor, p.is_admin) == ("alice", "user", "user:alice", False)
    raw = database.path.read_bytes()
    for suffix in ("-wal",):
        side = database.path.with_name(database.path.name + suffix)
        if side.exists():
            raw += side.read_bytes()
    assert token.encode() not in raw  # only the hash is on disk
    assert auth.hash_token(token).encode() in raw


def test_admin_role(database):
    _, token = auth.create_token(database, "admin", "phil", "cli:test")
    assert auth.authenticate(database, token).is_admin is True


@pytest.mark.parametrize("bad", [None, 5, "", "sct_", "sct_short", "x" * 200, "sct_" + "a" * 40 + "' OR 1=1 --", "Bearer sct_" + "a" * 40, b"sct_" + b"a" * 40, "sct_" + "é" * 30])
def test_malformed_or_unknown_tokens_authenticate_to_none(database, bad):
    auth.create_token(database, "user", "alice", "cli:test")
    assert auth.authenticate(database, bad) is None


def test_unknown_but_well_formed_token_is_rejected(database):
    auth.create_token(database, "user", "alice", "cli:test")
    assert auth.authenticate(database, "sct_" + "A" * 43) is None


def test_authenticate_with_no_database_does_not_create_one(tmp_path):
    d = Database(tmp_path / "none" / "scriptorium.db")
    assert auth.authenticate(d, "sct_" + "A" * 43) is None and not (tmp_path / "none").exists()


def test_revocation_takes_effect_immediately_and_is_audited(database):
    rec, token = auth.create_token(database, "user", "alice", "cli:test")
    assert auth.authenticate(database, token)
    auth.revoke_token(database, rec["id"], "cli:test")
    assert auth.authenticate(database, token) is None
    with pytest.raises(auth.AuthError, match="already revoked"):
        auth.revoke_token(database, rec["id"], "cli:test")
    with database.reader() as conn:
        events = [r["event"] for r in conn.execute("SELECT event FROM events WHERE kind='audit' ORDER BY seq")]
    assert events == ["token_created", "token_revoked"] and database.verify()["ok"]


def test_list_never_exposes_tokens_or_hashes(database):
    _, token = auth.create_token(database, "user", "alice", "cli:test")
    out = json.dumps(auth.list_tokens(database))
    assert token not in out and auth.hash_token(token) not in out and "token_hash" not in out


@pytest.mark.parametrize("role,name", [("root", "x"), ("user", ""), ("user", "a b"), ("user", "x" * 33), ("user", "../x"), ("user", None)])
def test_create_validation(database, role, name):
    with pytest.raises(auth.AuthError):
        auth.create_token(database, role, name, "cli:test")


@pytest.mark.parametrize("bad", ["tok_XYZ", "", None, "tok_1234567", "tok_123456789", "1; DROP TABLE tokens"])
def test_revoke_validation(database, bad):
    with pytest.raises(auth.AuthError):
        auth.revoke_token(database, bad, "cli:test")


def test_tokens_are_unique_and_high_entropy(database):
    tokens = {auth.create_token(database, "user", f"u{i}", "cli:test")[1] for i in range(20)}
    assert len(tokens) == 20 and all(len(t) >= 40 for t in tokens)


def test_sql_injection_in_name_is_stored_as_data_or_rejected(database):
    with pytest.raises(auth.AuthError):
        auth.create_token(database, "user", "x'); DROP TABLE tokens;--", "cli:test")
    auth.create_token(database, "user", "ok", "cli:test")
    assert len(auth.list_tokens(database)) == 1


def test_role_check_constraint_is_enforced_by_the_database(database):
    auth.create_token(database, "user", "alice", "cli:test")
    with database.transaction() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE tokens SET role='superuser'")


def test_database_files_are_private(database):
    auth.create_token(database, "user", "alice", "cli:test")
    assert stat.S_IMODE(database.path.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        side = database.path.with_name(database.path.name + suffix)
        if side.exists():
            assert stat.S_IMODE(side.stat().st_mode) == 0o600


# ---- legacy import ---------------------------------------------------------------------


def legacy_dir(tmp_path):
    d = tmp_path / "legacy"
    d.mkdir()
    ev1 = {"event": "submitted", "id": "fb_000000000001", "ts": "2026-01-01T00:00:00+00:00", "catalog": "nist", "text": "t", "label": "AC", "predicted": "AU", "model_version": "v", "submitter": "cli"}
    ev2 = {"event": "approved", "id": "fb_000000000001", "ts": "2026-01-01T00:01:00+00:00", "actor": "cli", "reason": "r"}
    (d / "nist.jsonl").write_text(json.dumps(ev1) + "\n" + json.dumps(ev2) + "\n")
    return d


def test_import_legacy_preserves_events_is_idempotent_and_leaves_files(database, tmp_path):
    d = legacy_dir(tmp_path)
    before = (d / "nist.jsonl").read_bytes()
    out = dbmod.import_legacy(database, d)
    assert out["imported"] == [{"file": "nist.jsonl", "events": 2}]
    again = dbmod.import_legacy(database, d)
    assert again["imported"] == [] and again["skipped_already_imported"] == ["nist.jsonl"]
    assert (d / "nist.jsonl").read_bytes() == before
    from scriptorium.feedback import FeedbackStore

    e = FeedbackStore(database).entries("nist")["fb_000000000001"]
    assert e["status"] == "approved" and e["ts"] == "2026-01-01T00:00:00+00:00" and e["history"][1]["reason"] == "r"
    assert database.verify()["ok"] is True


def test_import_legacy_rejects_a_bad_file_without_partial_import(database, tmp_path):
    d = legacy_dir(tmp_path)
    (d / "scf.jsonl").write_text('{"event": "submitted"}\nnot json\n')
    with pytest.raises(DatabaseError, match="scf.jsonl line"):
        dbmod.import_legacy(database, d)
    with database.reader() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM events WHERE catalog='scf'").fetchone()["n"]
    assert n == 0


# ---- backup ----------------------------------------------------------------------------


def test_backup_is_a_private_consistent_copy_and_never_overwrites(database, tmp_path):
    auth.create_token(database, "user", "alice", "cli:test")
    dest = tmp_path / "copy.db"
    out = dbmod.Database.backup(database, dest)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600 and len(out["sha256"]) == 64
    assert Database(dest).verify()["ok"] is True and Database(dest).head()["seq"] == 1
    with pytest.raises(DatabaseError, match="already exists"):
        database.backup(dest)


def test_verify_and_head_on_empty_database(tmp_path):
    d = Database(tmp_path / "x" / "scriptorium.db")
    assert d.verify() == {"ok": True, "events": 0, "head_seq": 0, "head_hash": dbmod.GENESIS, "anchors_checked": 0, "problems": []}
    assert d.head() == {"seq": 0, "hash": dbmod.GENESIS}


def test_malformed_anchor_file_is_reported(database, tmp_path):
    auth.create_token(database, "user", "alice", "cli:test")
    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text("garbage\n")
    v = database.verify(anchors)
    assert v["ok"] is False and any("malformed" in p for p in v["problems"])
