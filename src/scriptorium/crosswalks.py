"""Load user-supplied framework mappings into one normalized local store, and look pairs up.

Adapters (each fails loudly if the file's headers are not the expected ones):
    cis_controls_xlsx  CIS Controls workbook: safeguard titles and descriptions
    cis_mapping_xlsx   CIS Controls to NIST SP 800-53 Rev 5 workbook: safeguard-to-control pairs with relationship
    olir_xlsx          NIST OLIR template (NIST to ISO 27001, CSA CCM to NIST CSF)
    ctid_json          Mappings Explorer JSON: CIS safeguard to MITRE ATT&CK technique

Everything is written to `~/.scriptorium/crosswalks/` (outside the repository) and every input must be registered
in the source register first (`scriptorium.sources`). Results are always labelled published (an official pair) or
indirect (composed from two published links, never presented as an official mapping).

    python -m scriptorium.crosswalks build --cis-controls F --cis-nist F --nist-iso F --ccm-csf F --cis-attack F
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scriptorium import crosswalk as nist_ids
from scriptorium import ingest, paths, sources

FRAMEWORKS = {
    "nist-800-53r5": "NIST SP 800-53 Rev 5",
    "cis-8.1": "CIS Critical Security Controls v8.1",
    "iso-27001-2022": "ISO/IEC 27001:2022",
    "csf-2.0": "NIST Cybersecurity Framework 2.0",
    "ccm-4.0": "CSA Cloud Controls Matrix v4.0",
    "attack-19.1": "MITRE ATT&CK 19.1 (enterprise)",
}
INVERSE = {"subset": "superset", "superset": "subset"}
MAX_INDIRECT = 25
_ID_RE = re.compile(r"^[A-Za-z0-9().\-\[\]]{1,32}$")


class CrosswalkError(ValueError):
    pass


def _s(v: Any) -> str:
    """Text of a cell with non-breaking spaces and line breaks collapsed (workbook headers contain both)."""
    return re.sub(r"\s+", " ", str(v).replace("\xa0", " ")).strip() if v is not None else ""


def _need(header: list[str], expected: list[str], where: str) -> dict[str, int]:
    idx = {h: i for i, h in enumerate(header) if h}
    missing = [h for h in expected if h not in idx]
    if missing:
        raise CrosswalkError(f"{where}: expected columns {expected}; missing {missing}; found {[h for h in header if h]}")
    return idx


def _workbook(path: Path):
    try:
        import openpyxl
    except ImportError:
        raise CrosswalkError("reading workbooks needs openpyxl: uv sync --extra crosswalks") from None
    try:
        return openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        raise CrosswalkError(f"{path.name}: could not be read as an .xlsx workbook") from None


def _element(fw: str, id_: str, title: str = "", text: str = "", **meta: Any) -> dict[str, Any]:
    return {"framework": fw, "id": id_, "title": title, "text": text, **({"meta": meta} if meta else {})}


def _mapping(src: str, ff: str, fid: str, tf: str, tid: str, rel: str | None, rationale: str | None = None) -> dict[str, Any]:
    return {"source": src, "from_framework": ff, "from_id": fid, "to_framework": tf, "to_id": tid, "relationship": rel, "rationale": rationale}


# ---- adapters --------------------------------------------------------------------------------------------


def read_cis_controls(path: Path) -> tuple[list[dict], list[dict]]:
    wb = _workbook(path)
    sheet = next((ws for ws in wb.worksheets if ws.title.startswith("Controls v8")), None)
    if sheet is None:
        raise CrosswalkError(f"{path.name}: no sheet named 'Controls v8...'; found {[ws.title for ws in wb.worksheets]}")
    rows = list(sheet.iter_rows(values_only=True))
    idx = _need([_s(c) for c in rows[0]], ["CIS Control", "CIS Safeguard", "Asset Class", "Security Function", "Title", "Description"], "CIS controls sheet")
    els = []
    for r in rows[1:]:
        ctrl, sg = _s(r[idx["CIS Control"]]), _s(r[idx["CIS Safeguard"]])
        if not ctrl:
            continue
        title, desc = _s(r[idx["Title"]]), _s(r[idx["Description"]])
        if sg:
            els.append(_element("cis-8.1", sg, title, desc, control=ctrl, asset_class=_s(r[idx["Asset Class"]]), security_function=_s(r[idx["Security Function"]])))
        else:
            els.append(_element("cis-8.1", ctrl, title, desc, level="control"))
    if not any(e["meta"].get("control") for e in els if "meta" in e):
        raise CrosswalkError("CIS controls sheet had no safeguard rows")
    return els, []


def read_cis_mapping(path: Path) -> tuple[list[dict], list[dict]]:
    wb = _workbook(path)
    sheet = next((ws for ws in wb.worksheets if ws.title.startswith("All CIS Controls")), None)
    if sheet is None:
        raise CrosswalkError(f"{path.name}: no sheet named 'All CIS Controls...'; found {[ws.title for ws in wb.worksheets]}")
    rows = list(sheet.iter_rows(values_only=True))
    idx = _need([_s(c) for c in rows[0]], ["CIS Safeguard", "Title", "Relationship", "Control Identifier", "Control or Control Enhancement Name"], "CIS mapping sheet")
    els, maps, seen = [], [], set()
    for r in rows[1:]:
        sg, nid = _s(r[idx["CIS Safeguard"]]), _s(r[idx["Control Identifier"]])
        if not sg or not nid:
            continue
        rel = _s(r[idx["Relationship"]]).lower() or None
        maps.append(_mapping("cis-nist-mapping", "cis-8.1", sg, "nist-800-53r5", nist_ids.normalize_nist_id(nid) or nid, rel))
        if nid not in seen:
            seen.add(nid)
            els.append(_element("nist-800-53r5", nist_ids.normalize_nist_id(nid) or nid, _s(r[idx["Control or Control Enhancement Name"]])))
    return els, maps


def read_olir(path: Path, focal_fw: str, ref_fw: str, source: str) -> tuple[list[dict], list[dict]]:
    wb = _workbook(path)
    els, maps, seen = [], [], set()
    found = False
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows or not any("Focal Document Element" == _s(c) for c in rows[0]):
            continue
        found = True
        idx = _need([_s(c) for c in rows[0]], ["Focal Document Element", "Focal Document Element Description", "Reference Document Element"], f"OLIR sheet {ws.title!r}")
        strength_col = next((i for i, c in enumerate(rows[0]) if _s(c).startswith("Strength of Relationship")), None)
        for r in rows[1:]:
            fid, rid = _s(r[idx["Focal Document Element"]]), _s(r[idx["Reference Document Element"]])
            if not fid or not rid:
                continue
            if focal_fw == "nist-800-53r5":
                fid = nist_ids.normalize_nist_id(fid) or fid
            maps.append(_mapping(source, focal_fw, fid, ref_fw, rid, None))
            if strength_col is not None and _s(r[strength_col]):
                maps[-1]["rationale"] = f"strength of relationship: {_s(r[strength_col])}"
            if focal_fw != "nist-800-53r5" and fid not in seen:  # NIST text comes from our own OSCAL data
                seen.add(fid)
                els.append(_element(focal_fw, fid, "", _s(r[idx["Focal Document Element Description"]])))
    if not found:
        raise CrosswalkError(f"{path.name}: no sheet with a 'Focal Document Element' header (is this an OLIR template?)")
    return els, maps


def read_ctid(path: Path) -> tuple[list[dict], list[dict]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        objs = d["mapping_objects"]
    except (ValueError, KeyError, TypeError):
        raise CrosswalkError(f"{path.name}: expected a JSON object with 'mapping_objects'") from None
    els, maps, seen = [], [], set()
    for o in objs:
        for k in ("capability_id", "mapping_type"):
            if k not in o:
                raise CrosswalkError(f"{path.name}: a mapping object lacks {k!r}")
        cap = _s(o["capability_id"])
        sg = cap[4:] if cap.upper().startswith("CIS-") else cap
        tech = _s(o.get("attack_object_id"))
        if cap not in seen:
            seen.add(cap)
            els.append(_element("cis-8.1", sg, _s(o.get("capability_description"))))
        if not tech or o["mapping_type"] == "non_mappable":
            continue
        if ("t", tech) not in seen:
            seen.add(("t", tech))
            els.append(_element("attack-19.1", tech, _s(o.get("attack_object_name"))))
        maps.append(_mapping("cis-attack", "cis-8.1", sg, "attack-19.1", tech, _s(o["mapping_type"]).lower(), _s(o.get("comments")) or None))
    return els, maps


ADAPTERS = {
    "cis_controls_xlsx": lambda p, s: read_cis_controls(p),
    "cis_mapping_xlsx": lambda p, s: read_cis_mapping(p),
    "olir_xlsx": lambda p, s: read_olir(p, s.frameworks[0], s.frameworks[1], s.id),
    "ctid_json": lambda p, s: read_ctid(p),
}


# ---- build ---------------------------------------------------------------------------------------------


def store_dir() -> Path:
    return paths.scriptorium_home() / "crosswalks"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def build(files: dict[str, Path], out_dir: Path | None = None, accept_change: bool = False) -> dict[str, Any]:
    """Register and load each file, write the normalized store, and return a manifest (counts and provenance, no content)."""
    out_dir = out_dir or store_dir()
    all_els: dict[tuple[str, str], dict] = {}
    all_maps: list[dict] = []
    per_source = {}
    for sid, path in files.items():
        spec = sources.spec(sid)
        entry = sources.ensure_registered(sid, Path(path), accept_change=accept_change)
        els, maps = ADAPTERS[spec.adapter](Path(path), spec)
        for e in els:
            key = (e["framework"], e["id"])
            if key in all_els:  # keep the richest title/text seen
                old = all_els[key]
                old["title"] = old["title"] or e["title"]
                old["text"] = old["text"] or e["text"]
                old.setdefault("meta", {}).update(e.get("meta", {}))
            else:
                all_els[key] = e
        all_maps.extend(maps)
        per_source[sid] = {"file_name": entry["file_name"], "sha256": entry["sha256"], "license_status": entry["license_status"], "elements": len(els), "mappings": len(maps)}
    known = _known_nist_ids()
    nist_seen = {m[k] for m in all_maps for f, k in (("from_framework", "from_id"), ("to_framework", "to_id")) if m[f] == "nist-800-53r5"}
    unmatched = sorted(nist_seen - known) if known else []
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    _write_jsonl(out_dir / "elements.jsonl", sorted(all_els.values(), key=lambda e: (e["framework"], e["id"])))
    _write_jsonl(out_dir / "mappings.jsonl", all_maps)
    manifest = {
        "note": "Built from user-supplied files; local-only. Never commit this folder or content derived from it; publish aggregates only.",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": per_source,
        "totals": {"elements": len(all_els), "mappings": len(all_maps), "nist_ids_not_in_our_dataset": len(unmatched)},
    }
    fd = os.open(out_dir / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return manifest


def _known_nist_ids() -> set[str]:
    ids: set[str] = set()
    for name in ("train.jsonl", "test.jsonl"):
        p = ingest.PROCESSED_DIR / name
        if p.exists():
            ids |= {json.loads(line)["id"] for line in p.read_text(encoding="utf-8").splitlines() if line}
    return ids


# ---- lookup --------------------------------------------------------------------------------------------


class MappingStore:
    def __init__(self, elements: list[dict], mappings: list[dict]):
        self.elements = {(e["framework"], e["id"]): e for e in elements}
        self.mappings = mappings
        self._out: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._in: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m in mappings:
            self._out[(m["from_framework"], m["from_id"])].append(m)
            self._in[(m["to_framework"], m["to_id"])].append(m)

    @classmethod
    def load(cls, directory: Path | None = None) -> "MappingStore | None":
        d = directory or store_dir()
        if not (d / "mappings.jsonl").exists():
            return None
        rd = lambda n: [json.loads(line) for line in (d / n).read_text(encoding="utf-8").splitlines() if line]
        return cls(rd("elements.jsonl"), rd("mappings.jsonl"))

    def frameworks_present(self) -> list[str]:
        return sorted({m["from_framework"] for m in self.mappings} | {m["to_framework"] for m in self.mappings})

    @staticmethod
    def normalize(framework: str, element_id: str) -> str:
        if framework not in FRAMEWORKS:
            raise CrosswalkError(f"unknown framework; available: {sorted(FRAMEWORKS)}")
        if not isinstance(element_id, str) or not _ID_RE.match(element_id):
            raise CrosswalkError("element id has an invalid format")
        eid = element_id.strip()
        if framework == "nist-800-53r5":
            return nist_ids.normalize_nist_id(eid.upper()) or eid.upper()
        return eid.upper() if framework in ("csf-2.0", "ccm-4.0", "attack-19.1") else eid

    def _title(self, fw: str, id_: str) -> str | None:
        e = self.elements.get((fw, id_))
        return (e or {}).get("title") or None

    def published(self, framework: str, element_id: str) -> list[dict[str, Any]]:
        """Official pairs involving this element, in both directions, oriented from this element's side."""
        eid = self.normalize(framework, element_id)
        out = []
        for m in self._out.get((framework, eid), []):
            out.append({"framework": m["to_framework"], "id": m["to_id"], "title": self._title(m["to_framework"], m["to_id"]),
                        "relationship_stated": m["relationship"], "relationship_for_this_element": m["relationship"],
                        "stated_direction": "this element is the focal side", "source": m["source"], "rationale": m["rationale"], "kind": "published"})
        for m in self._in.get((framework, eid), []):
            rel = m["relationship"]
            out.append({"framework": m["from_framework"], "id": m["from_id"], "title": self._title(m["from_framework"], m["from_id"]),
                        "relationship_stated": rel, "relationship_for_this_element": INVERSE.get(rel, rel) if rel else None,
                        "stated_direction": "the other element is the focal side", "source": m["source"], "rationale": m["rationale"], "kind": "published"})
        return sorted(out, key=lambda x: (x["framework"], x["id"]))

    def indirect_attack(self, framework: str, element_id: str) -> dict[str, Any]:
        """ATT&CK techniques reachable from a NIST control through CIS safeguards. Composed from two published links."""
        eid = self.normalize(framework, element_id)
        if framework != "nist-800-53r5":
            return {"techniques": [], "total": 0}
        via_cis = [m for m in self._in.get(("nist-800-53r5", eid), []) if m["from_framework"] == "cis-8.1"]
        tech: dict[str, dict] = {}
        for link in via_cis:
            for m in self._out.get(("cis-8.1", link["from_id"]), []):
                if m["to_framework"] != "attack-19.1":
                    continue
                t = tech.setdefault(m["to_id"], {"framework": "attack-19.1", "id": m["to_id"], "title": self._title("attack-19.1", m["to_id"]), "via": [], "kind": "indirect"})
                t["via"].append({"cis_safeguard": link["from_id"], "cis_to_nist_relationship": link["relationship"], "cis_to_attack": m["relationship"]})
        ranked = sorted(tech.values(), key=lambda t: (-len(t["via"]), t["id"]))
        return {"techniques": ranked[:MAX_INDIRECT], "total": len(ranked), "note": "Indirect: composed from CIS-to-NIST and CIS-to-ATT&CK links; not an official NIST-to-ATT&CK mapping."}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for sid, flag in (("cis-controls", "--cis-controls"), ("cis-nist-mapping", "--cis-nist"), ("nist-iso27001", "--nist-iso"), ("ccm-csf", "--ccm-csf"), ("cis-attack", "--cis-attack")):
        b.add_argument(flag, type=Path, dest=sid.replace("-", "_"))
    b.add_argument("--accept-change", action="store_true", help="re-register a file whose hash differs from the registered one")
    args = ap.parse_args(argv)
    files = {sid: getattr(args, sid.replace("-", "_")) for sid in sources.SPECS if getattr(args, sid.replace("-", "_"), None)}
    if not files:
        print("error: give at least one file", file=sys.stderr)
        return 2
    try:
        m = build(files, accept_change=args.accept_change)
    except (CrosswalkError, sources.SourceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"sources": {k: {kk: v[kk] for kk in ("elements", "mappings", "license_status")} for k, v in m["sources"].items()}, "totals": m["totals"]}, indent=1))
    print(f"wrote {store_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
