"""API tokens with two roles, `user` and `admin`.

  * A token is 32 random bytes, shown once at creation. Only its SHA-256 hash is stored,
    so a copy of the database does not contain working tokens. (A fast hash is fine for
    high-entropy random tokens; it would not be for passwords.)
  * Tokens are created and revoked only from this CLI. The HTTP API has no way to mint a
    token, so a stolen admin token cannot create new ones to keep access.
  * Creation and revocation are recorded in the hash-chained event log.

    python -m scriptorium.auth create --role user --name alice
    python -m scriptorium.auth list
    python -m scriptorium.auth revoke tok_1a2b3c4d
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from scriptorium import paths
from scriptorium.db import Database, DatabaseError, append_event, now

ROLES = ("user", "admin")
TOKEN_PREFIX = "sct_"
MAX_TOKEN_CHARS = 80
LAST_USED_RESOLUTION = timedelta(minutes=1)  # limits write traffic from frequent requests
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_TOKEN_RE = re.compile(r"^sct_[A-Za-z0-9_-]{20,70}$")
_ID_RE = re.compile(r"^tok_[0-9a-f]{8}$")


class AuthError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    token_id: str
    name: str
    role: str

    @property
    def actor(self) -> str:
        return f"{self.role}:{self.name}"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_token(db: Database, role: str, name: str, actor: str) -> tuple[dict, str]:
    """Create a token. Returns (record, plaintext). The plaintext is not stored and cannot be recovered."""
    if role not in ROLES:
        raise AuthError(f"role must be one of {list(ROLES)}")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise AuthError("name must be 1-32 characters: letters, digits, dot, dash, underscore")
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    tid = "tok_" + secrets.token_hex(4)
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO tokens (id, name, role, token_hash, created_ts) VALUES (?,?,?,?,?)",
            (tid, name, role, hash_token(token), now()),
        )
        append_event(conn, "audit", "token_created", {"id": tid, "name": name, "role": role}, entity_id=tid, actor=actor)
    return {"id": tid, "name": name, "role": role}, token


def authenticate(db: Database, token: object) -> Principal | None:
    """Return the principal for a valid, unrevoked token, else None. Never raises on bad input."""
    if not isinstance(token, str) or len(token) > MAX_TOKEN_CHARS or not _TOKEN_RE.match(token):
        return None
    digest = hash_token(token)
    with db.reader() as conn:
        if conn is None:
            return None
        row = conn.execute("SELECT id, name, role, token_hash, last_used_ts FROM tokens WHERE token_hash=? AND revoked_ts IS NULL", (digest,)).fetchone()
    if row is None or not hmac.compare_digest(row["token_hash"], digest):
        return None
    last = row["last_used_ts"]
    if last is None or datetime.now(timezone.utc) - datetime.fromisoformat(last) > LAST_USED_RESOLUTION:
        with db.transaction() as conn:
            conn.execute("UPDATE tokens SET last_used_ts=? WHERE id=?", (now(), row["id"]))
    return Principal(row["id"], row["name"], row["role"])


def list_tokens(db: Database) -> list[dict]:
    with db.reader() as conn:
        if conn is None:
            return []
        rows = conn.execute("SELECT id, name, role, created_ts, last_used_ts, revoked_ts FROM tokens ORDER BY created_ts, id").fetchall()
    return [dict(r) for r in rows]  # never includes the token or its hash


def revoke_token(db: Database, token_id: str, actor: str) -> dict:
    if not isinstance(token_id, str) or not _ID_RE.match(token_id):
        raise AuthError("invalid token id: expected tok_ followed by 8 hex characters")
    with db.transaction() as conn:
        row = conn.execute("SELECT id, revoked_ts FROM tokens WHERE id=?", (token_id,)).fetchone()
        if row is None:
            raise AuthError(f"no token {token_id}")
        if row["revoked_ts"]:
            raise AuthError(f"{token_id} is already revoked")
        conn.execute("UPDATE tokens SET revoked_ts=? WHERE id=?", (now(), token_id))
        append_event(conn, "audit", "token_revoked", {"id": token_id}, entity_id=token_id, actor=actor)
    return {"id": token_id, "revoked": True}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--role", choices=ROLES, required=True)
    c.add_argument("--name", required=True)
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("id")
    args = ap.parse_args(argv)
    db = Database(paths.db_path())
    actor = f"cli:{getpass.getuser()}"
    try:
        if args.cmd == "create":
            rec, token = create_token(db, args.role, args.name, actor)
            print(json.dumps(rec, indent=1))
            print(f"\nToken (shown once, store it somewhere safe):\n{token}")
            return 0
        out = list_tokens(db) if args.cmd == "list" else revoke_token(db, args.id, actor)
    except (AuthError, DatabaseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
