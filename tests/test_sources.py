"""Tests for the source register."""

import json
import stat

import pytest

from scriptorium import sources
from scriptorium.sources import SourceError

openpyxl = pytest.importorskip("openpyxl")


def make_xlsx(path, license_text=None):
    wb = openpyxl.Workbook()
    wb.active.title = "Data"
    wb.active.append(["a", "b"])
    if license_text:
        ws = wb.create_sheet("License for Use")
        ws.append([license_text])
    wb.save(path)


CC = "This work is licensed under a Creative Commons Attribution-NonCommercial-No Derivatives 4.0 International Public License"


def test_register_records_hash_license_and_handling_privately(tmp_path):
    f = tmp_path / "cis.xlsx"
    make_xlsx(f, CC)
    e = sources.ensure_registered("cis-controls", f)
    assert e["sha256"] == sources.sha256_file(f) and e["license_status"] == "stated-in-file" and "NC-ND" in e["license_stated_in_file"]
    assert "local-only" in e["handling"] and e["file_name"] == "cis.xlsx"
    p = sources.register_path()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert str(tmp_path) not in p.read_text()  # only the file name is recorded, never a path


def test_files_without_a_stated_license_are_marked_unverified(tmp_path):
    f = tmp_path / "x.xlsx"
    make_xlsx(f)
    assert sources.ensure_registered("nist-iso27001", f)["license_status"] == "unverified"
    j = tmp_path / "m.json"
    j.write_text("{}")
    assert sources.ensure_registered("cis-attack", j)["license_stated_in_file"] is None


def test_reregistering_the_same_file_is_idempotent_and_a_changed_file_is_refused(tmp_path):
    f = tmp_path / "x.xlsx"
    make_xlsx(f)
    a = sources.ensure_registered("ccm-csf", f)
    assert sources.ensure_registered("ccm-csf", f) == a
    f2 = tmp_path / "x2.xlsx"
    make_xlsx(f2, "different")
    with pytest.raises(SourceError, match="hash differs"):
        sources.ensure_registered("ccm-csf", f2)
    assert sources.ensure_registered("ccm-csf", f2, accept_change=True)["sha256"] == sources.sha256_file(f2)


def test_unknown_ids_and_missing_files_are_refused(tmp_path):
    with pytest.raises(SourceError, match="unknown source id"):
        sources.ensure_registered("something-else", tmp_path / "x")
    with pytest.raises(SourceError, match="not found"):
        sources.ensure_registered("cis-controls", tmp_path / "missing.xlsx")


def test_every_spec_names_an_adapter_frameworks_and_a_source():
    for s in sources.SPECS.values():
        assert s.adapter and s.frameworks and s.publisher and s.where_to_get


def test_unreadable_workbook_reports_no_license_instead_of_crashing(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not a workbook")
    assert sources.detect_license(bad) is None
