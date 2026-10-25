"""Local SQLite database: tamper-evident event log, plus the token table.

Everything auditable (feedback events, token creation and revocation, admin actions) goes
into one insert-only `events` table. Each row stores a SHA-256 hash over its own content
and the previous row's hash, so editing, deleting or reordering a row breaks every hash
after it. `verify` recomputes the chain.

The chain is tamper-EVIDENT, not tamper-proof. Someone who can rewrite the whole file
can recompute every hash, and deleting the newest rows leaves a valid shorter chain.
`anchor` records the current head hash in a separate file; keep copies of it somewhere
the database owner cannot quietly change (a git commit, an email). `verify` then checks
that the anchored rows still exist unchanged.

The file lives in `~/.scriptorium/` with mode 0600 (directory 0700). It is not encrypted;
use full-disk encryption. All SQL uses bound parameters.

    python -m scriptorium.db verify
    python -m scriptorium.db head
    python -m scriptorium.db anchor
    python -m scriptorium.db import-legacy
    python -m scriptorium.db backup --to /path/to/copy.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scriptorium import paths

GENESIS = "0" * 64
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    catalog TEXT,
    entity_id TEXT,
    event TEXT NOT NULL,
    actor TEXT,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS events_kind_catalog ON events(kind, catalog, seq);
CREATE INDEX IF NOT EXISTS events_entity ON events(entity_id, seq);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TABLE IF NOT EXISTS tokens (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    token_hash TEXT NOT NULL UNIQUE,
    created_ts TEXT NOT NULL,
    last_used_ts TEXT,
    revoked_ts TEXT
);
"""


class DatabaseError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(prev: str, seq: int, ts: str, kind: str, catalog: str | None, entity_id: str | None, event: str, actor: str | None, payload: str) -> str:
    body = json.dumps([seq, ts, kind, catalog, entity_id, event, actor, payload], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256((prev + "|" + body).encode("utf-8")).hexdigest()


def append_event(
    conn: sqlite3.Connection,
    kind: str,
    event: str,
    payload: dict[str, Any],
    *,
    catalog: str | None = None,
    entity_id: str | None = None,
    actor: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Append one chained event. Must run inside a write transaction (`Database.transaction`)."""
    if not conn.in_transaction:
        raise DatabaseError("append_event must run inside a transaction")
    last = conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    seq = (last["seq"] if last else 0) + 1
    prev = last["hash"] if last else GENESIS
    ts = ts or now()
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = _digest(prev, seq, ts, kind, catalog, entity_id, event, actor, payload_json)
    conn.execute(
        "INSERT INTO events (seq, ts, kind, catalog, entity_id, event, actor, payload, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (seq, ts, kind, catalog, entity_id, event, actor, payload_json, prev, h),
    )
    return {"seq": seq, "hash": h}


class Database:
    def __init__(self, path: Path, anchors_path: Path | None = None):
        self.path = Path(path)
        self.anchors_path = Path(anchors_path) if anchors_path else self.path.parent / "anchors.jsonl"
        self._ready = False
        self._init_lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.exists()

    def _prepare_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        if not self.path.exists():
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
        os.chmod(self.path, 0o600)

    def connect(self) -> sqlite3.Connection:
        """Open a connection, creating and migrating the file on first use."""
        self._prepare_file()
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        if not self._ready:
            with self._init_lock:  # first use only: switching to WAL needs exclusive access
                if not self._ready:
                    self._init(conn)
                    self._ready = True
        for suffix in ("-wal", "-shm"):
            side = Path(str(self.path) + suffix)
            if side.exists():
                os.chmod(side, 0o600)
        return conn

    @staticmethod
    def _init(conn: sqlite3.Connection) -> None:
        """One-time setup. WAL mode persists in the file; other processes may be doing the same, so retry if busy."""
        for attempt in range(60):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SCHEMA)
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 59:
                    raise
                time.sleep(0.1)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Exclusive write transaction (BEGIN IMMEDIATE). Commits on success, rolls back on error."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection | None]:
        """A read connection, or None if the database file does not exist yet (nothing is created)."""
        if not self.exists():
            yield None
            return
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def audit(self, event: str, actor: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.transaction() as conn:
            return append_event(conn, "audit", event, payload or {}, actor=actor)

    # ---- integrity -------------------------------------------------------------------

    def verify(self, anchors_path: Path | None = None) -> dict[str, Any]:
        """Recompute the whole chain and check any recorded anchors."""
        problems: list[str] = []
        count, prev, head_seq = 0, GENESIS, 0
        with self.reader() as conn:
            rows = [] if conn is None else conn.execute(
                "SELECT seq, ts, kind, catalog, entity_id, event, actor, payload, prev_hash, hash FROM events ORDER BY seq"
            ).fetchall()
            by_seq = {r["seq"]: r["hash"] for r in rows}
        for r in rows:
            count += 1
            if r["seq"] != head_seq + 1:
                problems.append(f"sequence gap before seq {r['seq']} (expected {head_seq + 1})")
            if r["prev_hash"] != prev:
                problems.append(f"seq {r['seq']}: prev_hash does not match the previous row")
            if r["hash"] != _digest(r["prev_hash"], r["seq"], r["ts"], r["kind"], r["catalog"], r["entity_id"], r["event"], r["actor"], r["payload"]):
                problems.append(f"seq {r['seq']}: content does not match its hash")
            prev, head_seq = r["hash"], r["seq"]
        anchors_checked = 0
        ap = anchors_path if anchors_path is not None else self.anchors_path
        if ap.exists():
            for n, line in enumerate(ap.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    a = json.loads(line)
                    seq, h = int(a["seq"]), str(a["hash"])
                except (ValueError, KeyError, TypeError):
                    problems.append(f"anchors file line {n} is malformed")
                    continue
                anchors_checked += 1
                if seq not in by_seq:
                    problems.append(f"anchor at seq {seq} is missing: rows may have been removed")
                elif by_seq[seq] != h:
                    problems.append(f"anchor at seq {seq} does not match the stored row")
        return {"ok": not problems, "events": count, "head_seq": head_seq, "head_hash": prev, "anchors_checked": anchors_checked, "problems": problems}

    def head(self) -> dict[str, Any]:
        with self.reader() as conn:
            row = None if conn is None else conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return {"seq": row["seq"], "hash": row["hash"]} if row else {"seq": 0, "hash": GENESIS}

    def anchor(self, anchors_path: Path | None = None) -> dict[str, Any]:
        """Record the current head in the anchors file (append-only, mode 0600). Refuses if the chain is broken."""
        v = self.verify(anchors_path)
        if not v["ok"]:
            raise DatabaseError("refusing to anchor a chain that fails verification: " + "; ".join(v["problems"][:3]))
        ap = anchors_path if anchors_path is not None else self.anchors_path
        ap.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        entry = {"ts": now(), "seq": v["head_seq"], "hash": v["head_hash"]}
        fd = os.open(ap, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return entry

    def backup(self, dest: Path) -> dict[str, Any]:
        """Consistent snapshot using SQLite's online backup. Refuses to overwrite."""
        dest = Path(dest)
        if dest.exists():
            raise DatabaseError(f"{dest} already exists; refusing to overwrite")
        with self.reader() as src:
            if src is None:
                raise DatabaseError("no database to back up")
            os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            out = sqlite3.connect(dest)
            try:
                src.backup(out)
            finally:
                out.close()
        os.chmod(dest, 0o600)
        return {"backup": str(dest), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}


# ---- legacy JSONL import -----------------------------------------------------------------


def import_legacy(db: Database, feedback_dir: Path, actor: str = "cli") -> dict[str, Any]:
    """Copy the old per-catalog JSONL feedback logs into the database, once per file contents.

    Original timestamps and fields are preserved. The old files are left in place."""
    imported, skipped = [], []
    for path in sorted(Path(feedback_dir).glob("*.jsonl")):
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        events = []
        for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                assert isinstance(ev, dict) and isinstance(ev["event"], str) and isinstance(ev["id"], str) and isinstance(ev["ts"], str)
            except (ValueError, KeyError, AssertionError):
                raise DatabaseError(f"{path.name} line {n} is not a valid feedback event; nothing imported from it") from None
            events.append(ev)
        with db.transaction() as conn:
            if conn.execute("SELECT 1 FROM events WHERE kind='audit' AND event='legacy_import' AND entity_id=?", (sha,)).fetchone():
                skipped.append(path.name)
                continue
            for ev in events:
                append_event(conn, "feedback", ev["event"], ev, catalog=path.stem, entity_id=ev["id"], actor=ev.get("actor") or ev.get("submitter"), ts=ev["ts"])
            append_event(conn, "audit", "legacy_import", {"file": path.name, "events": len(events)}, entity_id=sha, actor=actor)
        imported.append({"file": path.name, "events": len(events)})
    return {"imported": imported, "skipped_already_imported": skipped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("head")
    sub.add_parser("anchor")
    sub.add_parser("import-legacy")
    b = sub.add_parser("backup")
    b.add_argument("--to", type=Path, required=True)
    args = ap.parse_args(argv)
    db = Database(paths.db_path())
    try:
        if args.cmd == "verify":
            out = db.verify()
        elif args.cmd == "head":
            out = db.head()
        elif args.cmd == "anchor":
            out = db.anchor()
        elif args.cmd == "import-legacy":
            out = import_legacy(db, paths.scriptorium_home() / "feedback", f"cli:{os.environ.get('USER', 'unknown')}")
        else:
            out = db.backup(args.to)
    except DatabaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
