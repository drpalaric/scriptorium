"""Shared synthetic-catalog helpers for tests. No real catalog content."""

import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """No test may read or write the real ~/.scriptorium."""
    monkeypatch.setenv("SCRIPTORIUM_HOME", str(tmp_path_factory.mktemp("scriptorium_home")))

VOCAB = {
    "AC": "account access privilege authorize login permission",
    "AU": "audit log record event timestamp review",
    "SC": "encrypt network boundary transmission cryptographic channel",
}


def rows_for(prefix_map, n_bases=12, per_base=2, pad=False):
    rows = []
    for fam, words in VOCAB.items():
        w = words.split()
        for b in range(1, n_bases + 1):
            for k in range(per_base):
                base = f"{prefix_map[fam]}-{b:02d}" if pad else f"{prefix_map[fam]}-{b}"
                rows.append(
                    {
                        "id": base if k == 0 else f"{base}({k})",
                        "base_id": base,
                        "family": prefix_map[fam],
                        "family_title": f"{prefix_map[fam]} title",
                        "title": f"title {base}",
                        "text": " ".join(w[(b + k + i) % len(w)] for i in range(4)) + f" item{b}{k}",
                    }
                )
    return rows


def write(dirpath, rows):
    dirpath.mkdir(parents=True, exist_ok=True)
    test = [r for r in rows if int(r["base_id"].split("-")[1]) > 9]
    train = [r for r in rows if int(r["base_id"].split("-")[1]) <= 9]
    for name, rs in (("train.jsonl", train), ("test.jsonl", test)):
        (dirpath / name).write_text("".join(json.dumps(r) + "\n" for r in rs))


def make_store(tmp_path):
    """A FeedbackStore backed by a fresh database under tmp_path."""
    from scriptorium.db import Database
    from scriptorium.feedback import FeedbackStore

    return FeedbackStore(Database(tmp_path / "home" / "scriptorium.db"))


def xlsx(path, sheets):
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)
    return path


def crosswalk_files(tmp_path):
    cis = xlsx(tmp_path / "cis.xlsx", {
        "Introduction": [["intro"]],
        "Controls v8.1.2": [
            ["CIS Control", "CIS Safeguard", "Asset Class", "Security Function", "Title", "Description", "IG1"],
            ["1\xa0", "", "", "", "Asset Control", "Manage assets", ""],
            ["1", "1.1", "Devices", "Identify", "Inventory", "Keep an inventory of devices", "x"],
            ["1", "1.2", "Devices", "Respond", "Unauthorized", "Handle unauthorized devices", "x"],
        ]})
    mapping = xlsx(tmp_path / "map.xlsx", {
        "All CIS Controls & Safeguards": [
            ["", "CIS Control", "CIS Safeguard", "Asset Type", "Security Function", "Title", "Description", "IG1", "IG2", "IG3", "Relationship", "Control Identifier",
             "Control or Control Enhancement Name", "Control Text"],
            ["", "1", "", "", "", "Asset Control", "", "", "", "", "", "", "", ""],
            ["", "1", "1.1", "Devices", "Identify", "Inventory", "d", "x", "x", "x", "Subset", "CM-8", "System Component Inventory", "txt"],
            ["", "1", "1.1", "Devices", "Identify", "Inventory", "d", "x", "x", "x", "Superset", "CM-8(1)", "Updates", "txt"],
            ["", "1", "1.2", "Devices", "Respond", "Unauthorized", "d", "x", "x", "x", "Equivalent", "CM-8(3)", "Detection", "txt"],
        ]})
    iso = xlsx(tmp_path / "iso.xlsx", {
        "Relationships-AC": [
            ["Focal Document\nElement", "Focal Document Element Description", "Security Control Baseline", "Reference Document\nElement", "Reference Document Element Description (Optional)", "Comments", "Strength of Relationship (Optional)"],
            ["CM-08", "a. inventory", "Low", "A.5.9", "", "", ""],
            ["CM-08(01)", "update", "Low", "A.5.9", "", "", ""],
            ["CM-08", "a. inventory", "Low", "7.5.1", "", "", ""],
        ],
        "Definitions": [["Term"]]})
    ccm = xlsx(tmp_path / "ccm.xlsx", {
        "General Information": [["Informative Reference Submission Form"]],
        "Relationships": [
            ["Focal Document Element", "Focal Document Element Description", "Reference Document Element", "Reference Document Element Description (Optional)", "Comments (Optional)", "Strength of Relationship (Optional)"],
            ["GV.OC-01", "The mission is understood", "BCR-01", "", "", ""],
            ["GV.OC-01", "The mission is understood", "BCR-07", "", "", ""],
        ]})
    attack = tmp_path / "attack.json"
    attack.write_text(json.dumps({"metadata": {}, "mapping_objects": [
        {"capability_id": "CIS-1.1", "capability_description": "Inventory", "mapping_type": "mitigates", "attack_object_id": "T1200", "attack_object_name": "Hardware Additions", "comments": "why"},
        {"capability_id": "CIS-1.2", "capability_description": "Unauthorized", "mapping_type": "mitigates", "attack_object_id": "T1200", "attack_object_name": "Hardware Additions", "comments": "why2"},
        {"capability_id": "CIS-1.2", "capability_description": "Unauthorized", "mapping_type": "mitigates", "attack_object_id": "T1078", "attack_object_name": "Valid Accounts", "comments": ""},
        {"capability_id": "CIS-3.1", "capability_description": "Policy", "mapping_type": "non_mappable", "attack_object_id": None, "attack_object_name": None, "comments": ""},
    ]}))
    return {"cis-controls": cis, "cis-nist-mapping": mapping, "nist-iso27001": iso, "ccm-csf": ccm, "cis-attack": attack}
