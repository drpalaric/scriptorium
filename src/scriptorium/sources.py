"""Register of user-supplied data files.

Some inputs (CIS workbooks, the CIS-to-ATT&CK file, NIST OLIR mappings, SCF) are downloaded by the user, kept
on their machine, and never committed. Every such file must be registered here before a loader reads it. The
register records the file's hash, what it is, and its license status, so what is public and what is local is a
checkable fact, and so a loader can refuse a file that changed after it was registered.

The register lives in `~/.scriptorium/sources.json` (outside the repository). This module holds only the
public part: which source ids exist and where to get them. No file content is stored in the register.

    python -m scriptorium.sources list
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scriptorium import paths


class SourceError(ValueError):
    """A source is unknown, unregistered, or no longer matches what was registered."""


@dataclass(frozen=True)
class SourceSpec:
    id: str
    title: str
    adapter: str
    frameworks: tuple[str, ...]
    publisher: str
    where_to_get: str


SPECS: dict[str, SourceSpec] = {s.id: s for s in (
    SourceSpec("cis-controls", "CIS Critical Security Controls v8.1.x workbook (safeguard text)", "cis_controls_xlsx", ("cis-8.1",),
               "Center for Internet Security", "CIS Controls v8.1 spreadsheet from cisecurity.org/controls"),
    SourceSpec("cis-nist-mapping", "CIS Controls v8.1 mapping to NIST SP 800-53 Rev 5 (all baselines)", "cis_mapping_xlsx", ("cis-8.1", "nist-800-53r5"),
               "Center for Internet Security", "Mapping workbook from cisecurity.org/controls"),
    SourceSpec("nist-iso27001", "NIST SP 800-53 Rev 5 to ISO/IEC 27001:2022 (OLIR)", "olir_xlsx", ("nist-800-53r5", "iso-27001-2022"),
               "NIST", "Informative reference from NIST OLIR (csrc.nist.gov/projects/olir)"),
    SourceSpec("ccm-csf", "CSA CCM v4.0 to NIST CSF 2.0 concept crosswalk (OLIR template)", "olir_xlsx", ("csf-2.0", "ccm-4.0"),
               "Cloud Security Alliance / NIST", "Informative reference from NIST OLIR or CSA"),
    SourceSpec("cis-attack", "CIS Controls v8.1.2 to MITRE ATT&CK 19.1 (Mappings Explorer JSON)", "ctid_json", ("cis-8.1", "attack-19.1"),
               "Center for Threat-Informed Defense", "Mappings Explorer export"),
)}

_CC_RE = re.compile(r"Creative Commons\s+Attribution[\s-]*Non[\s-]*Commercial[\s-]*No[\s-]*Derivatives?\s+4\.0", re.IGNORECASE)


def register_path() -> Path:
    return paths.scriptorium_home() / "sources.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_license(path: Path) -> str | None:
    """A license the FILE ITSELF states (CIS workbooks carry a 'License for Use' sheet). None if it states none."""
    if path.suffix.lower() != ".xlsx":
        return None
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            if ws.title.lower().startswith("license"):
                text = " ".join(str(c) for row in ws.iter_rows(values_only=True) for c in row if c)
                if _CC_RE.search(text):
                    return "CC BY-NC-ND 4.0 (stated in the file: non-commercial, no distribution of modified material)"
    except Exception:  # unreadable workbooks are reported by the adapter, not here
        return None
    return None


def _load() -> dict[str, Any]:
    p = register_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(data: dict[str, Any]) -> None:
    p = register_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(p.parent, 0o700)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def spec(source_id: str) -> SourceSpec:
    if source_id not in SPECS:
        raise SourceError(f"unknown source id {source_id!r}; known: {sorted(SPECS)}")
    return SPECS[source_id]


def ensure_registered(source_id: str, path: Path, *, accept_change: bool = False) -> dict[str, Any]:
    """Register `path` as `source_id` if new, or verify it still matches. Returns the register entry."""
    s = spec(source_id)
    path = Path(path)
    if not path.is_file():
        raise SourceError(f"{source_id}: file not found")
    digest = sha256_file(path)
    data = _load()
    entry = data.get(source_id)
    if entry is not None and entry["sha256"] != digest and not accept_change:
        raise SourceError(f"{source_id}: the file's hash differs from the registered one; check you have the right file, or re-register with accept_change")
    if entry is None or entry["sha256"] != digest:
        lic = detect_license(path)
        entry = {
            "title": s.title, "frameworks": list(s.frameworks), "publisher": s.publisher, "file_name": path.name, "sha256": digest,
            "size_bytes": path.stat().st_size, "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "license_stated_in_file": lic, "license_status": "stated-in-file" if lic else "unverified",
            "handling": "local-only: never commit this file or content derived from it; publish aggregates only",
        }
        data[source_id] = entry
        _save(data)
    return entry


def registered() -> dict[str, Any]:
    return _load()


def main(argv: list[str] | None = None) -> int:
    reg = registered()
    for sid, s in SPECS.items():
        e = reg.get(sid)
        status = f"registered {e['file_name']} sha256 {e['sha256'][:12]} license: {e['license_status']}" if e else "not registered"
        print(f"{sid:18} {s.title}\n{'':18} {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
