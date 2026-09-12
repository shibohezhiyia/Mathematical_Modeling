"""Bounded pre-publication audit for a source repository.

This is a release gate, not a full security scanner or legal opinion.  It
looks for high-signal credential/private-file patterns, generated caches and
oversized files, and reports every finding instead of silently deleting data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "mathmodel.release-audit/v1"
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "third_party"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".sqlite3", ".db", ".xlsx", ".xls", ".parquet", ".feather"}
_SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("api_assignment", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]")),
    ("provider_token", re.compile(r"(?i)\b(?:sk|rk|ds)-[A-Za-z0-9_-]{12,}\b")),
)
_PRIVATE_FILE_PATTERNS = (".env", ".env.local", ".env.production", "credentials.json", "secrets.json")
_LOCKFILES = ("requirements.txt", "requirements.lock", "poetry.lock", "uv.lock", "Pipfile.lock", "package-lock.json")
_LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")


class ReleaseAuditError(ValueError):
    """Raised for invalid audit configuration."""


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    detail: str
    severity: str


def audit_repository(root: str | Path, *, max_files: int = 20_000,
                     max_file_bytes: int = 8_000_000) -> dict[str, Any]:
    """Scan a bounded source tree without following symlinks or reading data files."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise ReleaseAuditError("repository_root_required")
    if type(max_files) is not int or not 1 <= max_files <= 100_000:
        raise ReleaseAuditError("invalid_max_files")
    if type(max_file_bytes) is not int or not 1_024 <= max_file_bytes <= 100_000_000:
        raise ReleaseAuditError("invalid_max_file_bytes")
    findings: list[Finding] = []
    scanned = 0
    oversized = 0
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        if scanned >= max_files:
            findings.append(Finding("scan_budget", ".", "file_count_limit_reached", "warning"))
            break
        if not path.is_file() or path.is_symlink() or any(part in _SKIP_DIRS for part in path.relative_to(base).parts):
            continue
        relative = path.relative_to(base).as_posix()
        if path.name.lower() in _PRIVATE_FILE_PATTERNS:
            findings.append(Finding("private_file", relative, "private_environment_or_credentials_file", "error"))
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if size > max_file_bytes:
            oversized += 1
            findings.append(Finding("oversized_file", relative, f"{size} bytes exceeds audit limit", "warning"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        for kind, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                # Test fixtures intentionally contain sentinel secrets to test
                # redaction.  Keep them visible to reviewers without blocking
                # a release; source/config findings remain hard blockers.
                severity = "warning" if relative.startswith("tests/") else "error"
                findings.append(Finding("secret_pattern", relative, kind, severity))
    errors = [asdict(item) for item in findings if item.severity == "error"]
    warnings = [asdict(item) for item in findings if item.severity != "error"]
    lockfiles = [name for name in _LOCKFILES if (base / name).is_file()]
    license_files = [name for name in _LICENSE_FILES if (base / name).is_file()]
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(base),
        "scanned_files": scanned,
        "oversized_files": oversized,
        "findings": [asdict(item) for item in findings],
        "errors": errors,
        "warnings": warnings,
        "publishable": not errors,
        "dependency_lock": {"status": "present" if lockfiles else "not_assessed", "files": lockfiles},
        "license_file": {"status": "present" if license_files else "not_assessed", "files": license_files},
        "policy": "bounded_high_signal_audit_not_a_complete_security_or_license_opinion",
    }


def assert_publishable(root: str | Path) -> dict[str, Any]:
    """Run the audit and fail closed when high-signal privacy findings exist."""
    report = audit_repository(root)
    if not report["publishable"]:
        raise ReleaseAuditError("release_audit_found_blockers")
    return report


__all__ = ["SCHEMA_VERSION", "ReleaseAuditError", "Finding", "audit_repository", "assert_publishable"]
