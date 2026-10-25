"""Run the Rego policies in `policy/` and check that every rule's control IDs are real.

    python -m scriptorium.policy check-audit                 # evaluate the real audit log
    python -m scriptorium.policy check-audit --single-operator
    python -m scriptorium.policy check-systems examples/ai_systems.json
    python -m scriptorium.policy controls                    # every rule with its NIST controls (and published CIS/ISO equivalents)

OPA is invoked as a subprocess with fixed arguments and the input on stdin; no user text is placed on a command
line or in a shell. The control assignments in the policies are the author's judgment, not an official mapping.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from scriptorium import baseline, ingest, paths, sources
from scriptorium.crosswalks import MappingStore, store_dir
from scriptorium.db import Database

POLICY_DIR = ingest.ROOT / "policy"
OPA_TIMEOUT_SECONDS = 60
MAX_INPUT_BYTES = 1_000_000
MAX_EVENTS = 200_000
_CONTROL_RE = re.compile(r"^[A-Z]{2}-\d{1,2}(\(\d{1,2}\))?$")


class PolicyError(RuntimeError):
    """OPA is missing, failed, or the input was not acceptable. The message is safe to show."""


def opa_binary() -> str:
    exe = shutil.which("opa")
    if not exe:
        raise PolicyError("the `opa` binary was not found on PATH; install it from https://www.openpolicyagent.org/docs/#running-opa")
    return exe


def _opa(args: list[str], stdin: str | None = None) -> str:
    try:
        r = subprocess.run([opa_binary(), *args], input=stdin, capture_output=True, text=True, timeout=OPA_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        raise PolicyError(f"opa did not finish within {OPA_TIMEOUT_SECONDS} seconds") from None
    if r.returncode != 0:
        raise PolicyError("opa failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[0][:300])
    return r.stdout


def evaluate(query: str, input_obj: dict[str, Any]) -> Any:
    out = _opa(["eval", "--format", "json", "--data", str(POLICY_DIR), "--stdin-input", query], json.dumps(input_obj))
    try:
        return json.loads(out)["result"][0]["expressions"][0]["value"]
    except (KeyError, IndexError, ValueError):
        raise PolicyError("opa returned no result for the query (is the policy package present?)") from None


# ---- the audit log -----------------------------------------------------------------------------------------


def export_audit(db: Database, single_operator: bool = False) -> dict[str, Any]:
    """The audit events plus the hash-chain result, in the shape the change-control policy expects."""
    with db.reader() as conn:
        rows = [] if conn is None else conn.execute(
            "SELECT seq, ts, kind, catalog, entity_id, event, actor, payload FROM events ORDER BY seq LIMIT ?", (MAX_EVENTS,)
        ).fetchall()
    events = [{**{k: r[k] for k in ("seq", "ts", "kind", "catalog", "entity_id", "event", "actor")}, "payload": json.loads(r["payload"])} for r in rows]
    chain = db.verify()
    return {"events": events, "chain": {k: chain[k] for k in ("ok", "events", "problems")}, "options": {"single_operator": bool(single_operator)}}


def check_audit(db: Database, single_operator: bool = False) -> dict[str, Any]:
    return evaluate("data.scriptorium.change_control.report", export_audit(db, single_operator))


def verify_evidence(systems: list[dict[str, Any]], root: Path) -> None:
    """Mark each evidence item `verified` only if its report is a file inside `root` whose SHA-256 matches. Mutates `systems`.

    A path outside `root` is never opened, so a record cannot make this read arbitrary files."""
    root = root.resolve()
    for s in systems:
        ev = s.get("evaluation_evidence")
        for item in ev if isinstance(ev, list) else []:
            if not isinstance(item, dict):
                continue
            item["verified"] = False
            report, claimed = item.get("report"), item.get("sha256")
            if not isinstance(report, str) or not isinstance(claimed, str):
                continue
            try:
                path = (root / report).resolve()
                if path.is_relative_to(root) and path.is_file() and sources.sha256_file(path) == claimed:
                    item["verified"] = True
            except (OSError, ValueError):
                pass


def check_systems(file: Path, today: str | None = None, root: Path | None = None) -> dict[str, Any]:
    if file.stat().st_size > MAX_INPUT_BYTES:
        raise PolicyError("system records file is too large")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except ValueError:
        raise PolicyError("the system records file is not valid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("systems"), list):
        raise PolicyError('expected a JSON object with a "systems" list')
    if len(data["systems"]) > 5000:
        raise PolicyError("too many system records")
    if not all(isinstance(s, dict) for s in data["systems"]):
        raise PolicyError("every system record must be a JSON object")
    verify_evidence(data["systems"], root or ingest.ROOT)
    return evaluate("data.scriptorium.ai_system.report", {"systems": data["systems"], "today": today or date.today().isoformat()})


# ---- rule metadata -------------------------------------------------------------------------------------------


def rule_controls() -> list[dict[str, Any]]:
    """Every annotated rule: package, id, title, controls, mapping basis."""
    out = json.loads(_opa(["inspect", "--annotations", "--format", "json", str(POLICY_DIR)]))
    rules = []
    for a in out.get("annotations") or []:
        ann = a["annotations"]
        if ann.get("scope") != "rule":
            continue
        custom = ann.get("custom") or {}
        pkg = ".".join(p["value"] for p in a["path"][1:-1])
        rules.append({"package": pkg, "id": custom.get("id"), "title": ann.get("title"), "controls": custom.get("controls"), "mapping_basis": custom.get("mapping_basis")})
    return rules


def validate_controls(rules: list[dict[str, Any]], known_nist_ids: set[str]) -> list[str]:
    """Problems with the annotations: missing fields, duplicate ids, malformed IDs, IDs not in the NIST dataset."""
    problems, seen = [], set()
    for r in rules:
        name = r.get("id") or f"(rule titled {r.get('title')!r})"
        if not r.get("id"):
            problems.append(f"{name}: no custom.id")
        elif r["id"] in seen:
            problems.append(f"{name}: duplicate id")
        seen.add(r.get("id"))
        if not r.get("mapping_basis"):
            problems.append(f"{name}: no mapping_basis (say whose judgment the control assignment is)")
        ctrls = r.get("controls")
        if not isinstance(ctrls, list) or not ctrls:
            problems.append(f"{name}: no controls")
            continue
        for c in ctrls:
            if not isinstance(c, str) or not _CONTROL_RE.match(c):
                problems.append(f"{name}: {c!r} is not a NIST control ID")
            elif c not in known_nist_ids:
                problems.append(f"{name}: {c} is not an active control in the NIST SP 800-53 Rev 5 dataset")
    return problems


def known_nist_ids() -> set[str]:
    return {r["id"] for r in baseline.load_jsonl(ingest.PROCESSED_DIR / "train.jsonl") + baseline.load_jsonl(ingest.PROCESSED_DIR / "test.jsonl")}


def equivalents(control: str, store: MappingStore | None) -> dict[str, list[str]]:
    """Published CIS and ISO 27001 pairs for a NIST control, from the user's local store (empty if none was built)."""
    if store is None:
        return {}
    out: dict[str, list[str]] = {}
    for p in store.published("nist-800-53r5", control):
        if p["framework"] in ("cis-8.1", "iso-27001-2022"):
            out.setdefault(p["framework"], []).append(p["id"])
    return out


# ---- command line ----------------------------------------------------------------------------------------------


def _print_findings(items: list[dict[str, Any]], key: str, store: MappingStore | None, label: str) -> None:
    for v in sorted(items, key=lambda x: (x["rule"], str(x.get(key)))):
        eq = {c: equivalents(c, store) for c in v["controls"]}
        extra = "; ".join(f"{c}: " + ", ".join(f"{fw} {','.join(ids[:4])}" for fw, ids in e.items()) for c, e in eq.items() if e)
        print(f"  {label} {v['rule']:18} {v['message']}  [{key} {v.get(key)}]  controls {', '.join(v['controls'])}" + (f"  (published: {extra})" if extra else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check-audit")
    a.add_argument("--single-operator", action="store_true", help="one person holds every role: separation-of-duties findings are reported as accepted exceptions")
    a.add_argument("--json", action="store_true")
    s = sub.add_parser("check-systems")
    s.add_argument("file", type=Path)
    s.add_argument("--today", default=None, help="YYYY-MM-DD, for the review-age rule (default: today)")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("controls")
    c.add_argument("--no-mappings", action="store_true")
    args = ap.parse_args(argv)
    try:
        store = None if getattr(args, "no_mappings", False) else MappingStore.load(store_dir())
        if args.cmd == "controls":
            rules = rule_controls()
            problems = validate_controls(rules, known_nist_ids())
            for r in rules:
                eq = {c: equivalents(c, store) for c in r["controls"]}
                print(f"{r['id']:20} {', '.join(r['controls']):14} {r['title']}")
                for ctrl, e in eq.items():
                    if e:
                        print(f"{'':20} {ctrl}: " + "; ".join(f"{fw} {', '.join(ids[:5])}" for fw, ids in e.items()))
            print(f"\n{len(rules)} rules; control validation: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
            for p in problems:
                print("  -", p)
            return 0 if not problems else 1
        if args.cmd == "check-audit":
            rep = check_audit(Database(paths.db_path()), args.single_operator)
            if args.json:
                print(json.dumps(rep, indent=1))
            else:
                print(f"checked {rep['events_checked']} audit events (single-operator mode: {rep['single_operator']})")
                _print_findings(rep["violations"], "entity", store, "VIOLATION")
                _print_findings(rep["exceptions"], "entity", store, "EXCEPTION")
                if not rep["violations"] and not rep["exceptions"]:
                    print("  no findings")
            return 1 if rep["violations"] else 0
        rep = check_systems(args.file, args.today)
        if args.json:
            print(json.dumps(rep, indent=1))
        else:
            print(f"checked {rep['systems_checked']} system record(s)")
            _print_findings(rep["findings"], "system", store, "FINDING")
            if not rep["findings"]:
                print("  no findings")
        return 1 if rep["findings"] else 0
    except (PolicyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
