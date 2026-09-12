from pathlib import Path

import pytest

from core.release_audit import ReleaseAuditError, assert_publishable, audit_repository


def test_release_audit_flags_secret_and_private_file(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("API_KEY = 'ds-abcdefghijklmnop'", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=x", encoding="utf-8")
    report = audit_repository(tmp_path)
    assert not report["publishable"]
    assert {item["kind"] for item in report["errors"]} == {"secret_pattern", "private_file"}
    with pytest.raises(ReleaseAuditError, match="blockers"):
        assert_publishable(tmp_path)


def test_release_audit_skips_binary_data_and_accepts_clean_source(tmp_path: Path):
    (tmp_path / "data.xlsx").write_bytes(b"not source")
    (tmp_path / "main.py").write_text("VALUE = 3\n", encoding="utf-8")
    report = audit_repository(tmp_path)
    assert report["publishable"]
    assert report["errors"] == []


def test_release_audit_reports_lock_and_license_presence_without_claiming_legal_review(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.0\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    report = audit_repository(tmp_path)
    assert report["dependency_lock"]["status"] == "present"
    assert report["license_file"]["status"] == "present"
