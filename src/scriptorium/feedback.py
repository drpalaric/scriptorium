"""Corrections ("feedback") with an approval step, a held-out gate, and rollback.

Lifecycle of one correction:

    submitted -> pending --approve--> approved --apply--> applied --rollback--> rolled_back
                   |                     |
                   +--withdraw/reject    +--reject

Events are stored in the hash-chained, insert-only `events` table of the local SQLite
database (see `scriptorium.db`). Status is derived by replaying events; nothing is edited
or deleted in place, so every change has a history. Only `applied` corrections are used for training, and only after
`apply` has checked that the frozen held-out score did not drop by more than a tolerance.

Pending entries never change a model. Approve, apply, rollback and reset are CLI (admin)
operations; a client such as the MCP server will only be able to submit, list and withdraw.

    python -m scriptorium.feedback list --catalog nist
    python -m scriptorium.feedback approve fb_ab12cd34ef56 --catalog nist
    python -m scriptorium.feedback apply --catalog nist
    python -m scriptorium.feedback rollback fb_ab12cd34ef56 --catalog nist
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import secrets
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scriptorium.db import Database, append_event

if TYPE_CHECKING:
    from scriptorium.service import Service

CORRECTION_WEIGHT = 5.0  # training weight of one correction relative to one catalog record
GATE_TOLERANCE = 0.02  # max allowed drop in held-out macro-F1 when applying
MAX_PENDING = 500  # per catalog
MAX_REASON_CHARS = 500

STATUS_AFTER = {
    "submitted": "pending",
    "withdrawn": "withdrawn",
    "approved": "approved",
    "rejected": "rejected",
    "applied": "applied",
    "rolled_back": "rolled_back",
}
ALLOWED_FROM = {
    "withdrawn": {"pending"},
    "approved": {"pending"},
    "rejected": {"pending", "approved"},
    "applied": {"approved"},
    "rolled_back": {"applied"},
}
_ID_RE = re.compile(r"^fb_[0-9a-f]{12}$")
BAD_ID = "invalid feedback id: expected fb_ followed by 12 lowercase hex characters (for example fb_5eed285fa122)"
VERB = {"withdrawn": "withdrawn", "approved": "approved", "rejected": "rejected", "applied": "applied", "rolled_back": "rolled back"}
_CATALOG_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


class FeedbackError(ValueError):
    """Invalid feedback operation. The message is safe to show to the caller."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class FeedbackStore:
    """Feedback events kept in the hash-chained `events` table of the local database."""

    def __init__(self, db: Database):
        self.db = db
        self._tx = threading.local()

    def _check_catalog(self, catalog: str) -> None:
        if not isinstance(catalog, str) or not _CATALOG_RE.match(catalog):
            raise FeedbackError("invalid catalog name")

    @contextmanager
    def locked(self, catalog: str):
        """Exclusive write transaction held across a read-check-append sequence."""
        self._check_catalog(catalog)
        with self.db.transaction() as conn:
            self._tx.conn = conn
            try:
                yield
            finally:
                self._tx.conn = None

    def _append(self, catalog: str, event: dict[str, Any]) -> None:
        """Append one event. Caller must hold `locked(catalog)`."""
        conn = getattr(self._tx, "conn", None)
        if conn is None:
            raise FeedbackError("internal error: append outside a locked transaction")
        append_event(
            conn, "feedback", event["event"], event, catalog=catalog, entity_id=event["id"],
            actor=event.get("actor") or event.get("submitter"), ts=event["ts"],
        )

    def events(self, catalog: str) -> list[dict[str, Any]]:
        self._check_catalog(catalog)
        conn = getattr(self._tx, "conn", None)
        if conn is not None:
            rows = conn.execute("SELECT seq, payload FROM events WHERE kind='feedback' AND catalog=? ORDER BY seq", (catalog,)).fetchall()
        else:
            with self.db.reader() as c:
                rows = [] if c is None else c.execute("SELECT seq, payload FROM events WHERE kind='feedback' AND catalog=? ORDER BY seq", (catalog,)).fetchall()
        out = []
        for r in rows:
            try:
                ev = json.loads(r["payload"])
            except json.JSONDecodeError:
                raise FeedbackError(f"feedback event {r['seq']} is corrupt; not using the log") from None
            if not isinstance(ev, dict) or ev.get("event") not in STATUS_AFTER or not _ID_RE.match(str(ev.get("id", ""))):
                raise FeedbackError(f"feedback event {r['seq']} is invalid; not using the log")
            out.append(ev)
        return out

    def entries(self, catalog: str) -> dict[str, dict[str, Any]]:
        """Replay the log into {id: entry}. Each entry has its current status and full history."""
        entries: dict[str, dict[str, Any]] = {}
        for ev in self.events(catalog):
            fid = ev["id"]
            if ev["event"] == "submitted":
                if fid in entries:
                    raise FeedbackError(f"feedback log has a duplicate submission for {fid}")
                entries[fid] = {k: ev[k] for k in ("id", "catalog", "text", "label", "predicted", "model_version", "submitter", "ts")}
                entries[fid]["history"] = []
            elif fid not in entries:
                raise FeedbackError(f"feedback log has an event for unknown entry {fid}")
            entry = entries[fid]
            entry["status"] = STATUS_AFTER[ev["event"]]
            entry["history"].append({k: v for k, v in ev.items() if k not in ("text", "label", "predicted", "catalog")})
            if ev["event"] == "applied":
                entry["weight"] = ev["weight"]
        return entries

    def applied(self, catalog: str) -> list[dict[str, Any]]:
        return [e for e in self.entries(catalog).values() if e["status"] == "applied"]


# ---- operations ---------------------------------------------------------------------------


def _reason(reason: Any) -> str | None:
    if reason is None:
        return None
    if not isinstance(reason, str) or len(reason) > MAX_REASON_CHARS or "\x00" in reason:
        raise FeedbackError(f"reason must be a string of at most {MAX_REASON_CHARS} characters")
    return reason.strip() or None


def _state_error(fid: str, status: str, event: str) -> FeedbackError:
    if status == STATUS_AFTER[event]:
        msg = f"{fid} is already {VERB[event]}; nothing to do"
    else:
        allowed = " or ".join(sorted(ALLOWED_FROM[event]))
        msg = f"{fid} is {VERB.get(status, status)}; only {allowed} entries can be {VERB[event]}"
    if status == "applied" and event != "rolled_back":
        msg += " (use rollback to undo it)"
    return FeedbackError(msg)


def _receipt(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: entry[k] for k in ("id", "status", "catalog", "text", "label", "predicted", "model_version", "submitter", "ts")}


def submit(svc: "Service", store: FeedbackStore, catalog: Any, text: Any, label: Any, submitter: str = "cli") -> dict[str, Any]:
    """Record a pending correction and return a receipt echoing exactly what was stored."""
    src = svc._catalog(catalog)
    text = svc._text(text)
    labels = svc.labels(src.name)
    if not isinstance(label, str) or label.upper() not in labels:
        raise FeedbackError(f"label must be one of the catalog's labels: {sorted(labels)}")
    label = label.upper()
    with store.locked(src.name):
        entries = store.entries(src.name)
        norm = normalize_text(text)
        for e in entries.values():
            if e["status"] == "pending" and e["label"] == label and normalize_text(e["text"]) == norm:
                return {**_receipt(e), "duplicate": True}
        if sum(1 for e in entries.values() if e["status"] == "pending") >= MAX_PENDING:
            raise FeedbackError(f"too many pending corrections for {src.name} (max {MAX_PENDING}); review the queue first")
        pred = svc.classify(text, src.name, 1)
        event = {
            "event": "submitted",
            "id": "fb_" + secrets.token_hex(6),
            "ts": _now(),
            "catalog": src.name,
            "text": text,
            "label": label,
            "predicted": pred["predictions"][0]["label"],
            "model_version": pred["model_version"],
            "submitter": submitter,
        }
        store._append(src.name, event)
        return {**_receipt(store.entries(src.name)[event["id"]]), "duplicate": False}


def transition(
    store: FeedbackStore, catalog: str, fid: Any, event: str, actor: str, reason: Any = None, only_submitter: str | None = None
) -> dict[str, Any]:
    """Withdraw, approve or reject one entry. Apply and rollback have their own functions.

    `only_submitter` restricts the change to entries that submitter created (used so an
    MCP client can withdraw only its own submissions)."""
    if event not in ("withdrawn", "approved", "rejected"):
        raise FeedbackError("unsupported transition")
    if not isinstance(fid, str) or not _ID_RE.match(fid):
        raise FeedbackError(BAD_ID)
    with store.locked(catalog):
        entry = store.entries(catalog).get(fid)
        if entry is None or (only_submitter is not None and entry["submitter"] != only_submitter):
            raise FeedbackError(f"no feedback entry {fid} in {catalog}")
        if entry["status"] not in ALLOWED_FROM[event]:
            raise _state_error(fid, entry["status"], event)
        store._append(catalog, {"event": event, "id": fid, "ts": _now(), "actor": actor, "reason": _reason(reason)})
        return _receipt(store.entries(catalog)[fid])


BULK_RELABEL_MIN = 5  # this many corrections moving one predicted label to one other label is flagged


def anomaly_warnings(svc: "Service", catalog: str, current_pipe, candidates: list[dict[str, Any]]) -> list[str]:
    """Signals for the human reviewer. They never block: the held-out gate cannot see
    attacks aimed away from the test set (for example a trigger word), so review is the
    real control and these point the reviewer at what to look at."""
    if not candidates:
        return []
    import numpy as np

    out = []
    threshold, _ = svc._calibrate(catalog)
    scores = current_pipe.decision_function([e["text"] for e in candidates])
    order = np.argsort(-scores, axis=1)
    rows = np.arange(len(candidates))
    margins = scores[rows, order[:, 0]] - scores[rows, order[:, 1]]
    preds = current_pipe.classes_[order[:, 0]]
    confident_overrides = [e["id"] for e, p, m in zip(candidates, preds, margins) if p != e["label"] and m >= threshold]
    if confident_overrides:
        out.append(f"{len(confident_overrides)} correction(s) contradict a confident model prediction: {confident_overrides[:10]}")
    pairs: dict[tuple[str, str], int] = {}
    for e, p in zip(candidates, preds):
        if p != e["label"]:
            pairs[(str(p), e["label"])] = pairs.get((str(p), e["label"]), 0) + 1
    for (a, b), n in sorted(pairs.items()):
        if n >= BULK_RELABEL_MIN:
            out.append(f"{n} corrections relabel {a} as {b} in one batch")
    return out


def apply(
    svc: "Service",
    store: FeedbackStore,
    catalog: str,
    actor: str = "cli",
    weight: float = CORRECTION_WEIGHT,
    tolerance: float = GATE_TOLERANCE,
    force: bool = False,
    reason: Any = None,
) -> dict[str, Any]:
    """Train on applied + approved corrections, compare held-out scores, and apply if the gate passes.

    Approved corrections whose text matches a held-out test record are rejected, so the
    test split never leaks into training. With `force`, a failed gate is overridden and the
    override is recorded with its reason.
    """
    src = svc._catalog(catalog)
    reason = _reason(reason)
    if force and not reason:
        raise FeedbackError("--force requires a reason")
    if not 0 < weight <= 100:
        raise FeedbackError("weight must be in (0, 100]")
    with store.locked(src.name):
        entries = store.entries(src.name)
        approved = [e for e in entries.values() if e["status"] == "approved"]
        if not approved:
            return {"catalog": src.name, "applied": [], "message": "nothing approved to apply"}
        test_texts = {normalize_text(r["text"]) for r in svc._load(src.name)[1]}
        leaked = [e for e in approved if normalize_text(e["text"]) in test_texts]
        for e in leaked:
            store._append(src.name, {"event": "rejected", "id": e["id"], "ts": _now(), "actor": "system", "reason": "text matches a held-out test record"})
        candidates = [e for e in approved if e not in leaked]
        if not candidates:
            return {"catalog": src.name, "applied": [], "rejected_as_test_leak": [e["id"] for e in leaked]}

        current = [e for e in entries.values() if e["status"] == "applied"]
        current_pipe = svc.fit(src.name, current)
        before = svc.heldout_scores(src.name, current_pipe)
        warnings = anomaly_warnings(svc, src.name, current_pipe, candidates)
        new = [{**e, "weight": weight} for e in candidates]
        pipe = svc.fit(src.name, current + new)
        after = svc.heldout_scores(src.name, pipe)
        drop = before["macro_f1"] - after["macro_f1"]
        passed = drop <= tolerance
        gate = {"before": before, "after": after, "macro_f1_drop": round(drop, 4), "tolerance": tolerance, "passed": passed}
        report = {
            "catalog": src.name,
            "gate": gate,
            "rejected_as_test_leak": [e["id"] for e in leaked],
            "weight": weight,
            "warnings": warnings,
        }
        if not passed and not force:
            return {**report, "applied": [], "message": "held-out score dropped more than the tolerance; nothing applied"}
        preds = pipe.predict([e["text"] for e in candidates])
        for e in candidates:
            store._append(
                src.name,
                {
                    "event": "applied",
                    "id": e["id"],
                    "ts": _now(),
                    "actor": actor,
                    "weight": weight,
                    "gate": gate,
                    "forced": bool(force and not passed),
                    "reason": reason,
                },
            )
        report["applied"] = [
            {"id": e["id"], "label": e["label"], "model_now_predicts": str(p), "now_correct": str(p) == e["label"]}
            for e, p in zip(candidates, preds)
        ]
        report["forced"] = bool(force and not passed)
        return report


def rollback(store: FeedbackStore, catalog: str, fids: list[str], actor: str = "cli", reason: Any = None) -> dict[str, Any]:
    reason = _reason(reason)
    with store.locked(catalog):
        entries = store.entries(catalog)
        for fid in fids:
            if not isinstance(fid, str) or not _ID_RE.match(fid):
                raise FeedbackError(BAD_ID)
            if fid not in entries:
                raise FeedbackError(f"no feedback entry {fid} in {catalog}")
            if entries[fid]["status"] != "applied":
                raise _state_error(fid, entries[fid]["status"], "rolled_back")
        for fid in fids:
            store._append(catalog, {"event": "rolled_back", "id": fid, "ts": _now(), "actor": actor, "reason": reason})
    return {"catalog": catalog, "rolled_back": fids}


def reset(store: FeedbackStore, catalog: str, actor: str = "cli", reason: Any = "reset") -> dict[str, Any]:
    """Roll back every applied correction for a catalog. The history is kept."""
    ids = [e["id"] for e in store.applied(catalog)]
    if not ids:
        return {"catalog": catalog, "rolled_back": []}
    return rollback(store, catalog, ids, actor, reason)


MAX_LIST = 100
LIST_TEXT_CHARS = 300


def list_entries(store: FeedbackStore, catalog: str, status: Any = None, limit: Any = 50, submitter: str | None = None) -> list[dict[str, Any]]:
    """Newest first. Text is shortened; use `show` (CLI) for the full entry."""
    if status is not None and status not in set(STATUS_AFTER.values()):
        raise FeedbackError(f"status must be one of {sorted(set(STATUS_AFTER.values()))}")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST:
        raise FeedbackError(f"limit must be an integer from 1 to {MAX_LIST}")
    rows = [
        e for e in store.entries(catalog).values()
        if (status is None or e["status"] == status) and (submitter is None or e["submitter"] == submitter)
    ]
    rows.sort(key=lambda e: e["history"][-1]["ts"], reverse=True)
    return [
        {"id": e["id"], "status": e["status"], "label": e["label"], "predicted": e["predicted"], "submitter": e["submitter"],
         "submitted": e["ts"], "updated": e["history"][-1]["ts"], "text": e["text"][:LIST_TEXT_CHARS]}
        for e in rows[:limit]
    ]


# ---- CLI ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from scriptorium.service import Service

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="nist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--text", required=True)
    s.add_argument("--label", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=sorted(set(STATUS_AFTER.values())))
    sh = sub.add_parser("show")
    sh.add_argument("id")
    for name in ("approve", "reject", "withdraw"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("--reason")
    a = sub.add_parser("apply")
    a.add_argument("--weight", type=float, default=CORRECTION_WEIGHT)
    a.add_argument("--tolerance", type=float, default=GATE_TOLERANCE)
    a.add_argument("--force", action="store_true")
    a.add_argument("--reason")
    r = sub.add_parser("rollback")
    r.add_argument("ids", nargs="+")
    r.add_argument("--reason")
    rs = sub.add_parser("reset")
    rs.add_argument("--reason", default="reset")
    args = ap.parse_args(argv)

    svc = Service.default()
    store = svc.feedback
    actor = f"cli:{getpass.getuser()}"
    try:
        if args.cmd == "submit":
            out: Any = submit(svc, store, args.catalog, args.text, args.label, submitter=actor)
        elif args.cmd == "list":
            out = list_entries(store, args.catalog, args.status, MAX_LIST)
        elif args.cmd == "show":
            entry = store.entries(args.catalog).get(args.id)
            if entry is None:
                raise FeedbackError(BAD_ID if not _ID_RE.match(args.id) else f"no feedback entry {args.id} in {args.catalog}")
            out = {"id": entry["id"], "status": entry["status"]} | entry
        elif args.cmd in ("approve", "reject", "withdraw"):
            event = {"approve": "approved", "reject": "rejected", "withdraw": "withdrawn"}[args.cmd]
            out = transition(store, args.catalog, args.id, event, actor, args.reason)
        elif args.cmd == "apply":
            out = apply(svc, store, args.catalog, actor, args.weight, args.tolerance, args.force, args.reason)
        elif args.cmd == "rollback":
            out = rollback(store, args.catalog, args.ids, actor, args.reason)
        else:
            out = reset(store, args.catalog, actor, args.reason)
    except (FeedbackError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
