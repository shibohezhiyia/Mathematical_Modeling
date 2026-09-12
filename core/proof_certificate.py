"""Small, explicit proof-certificate envelope for mathematical claims.

The envelope does not prove a theorem itself.  It records the conditions and
the independent checker that did so, and keeps execution/empirical evidence
out of the deductive status field.  This prevents the common ``verified=True``
shortcut from mixing numerical success with a mathematical proof.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


class ProofCertificateError(ValueError):
    pass


SCHEMA_VERSION = "mathmodel.proof-certificate/v1"


def _text(value: Any, name: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ProofCertificateError(f"invalid_{name}")
    return value.strip()


def _text_list(value: Any, name: str, limit: int = 128) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value or len(value) > limit:
        raise ProofCertificateError(f"invalid_{name}")
    result = [_text(item, name, 1000) for item in value]
    if len(set(result)) != len(result):
        raise ProofCertificateError(f"duplicate_{name}")
    return result


def issue_proof_certificate(
    *, statement: str, assumptions: Sequence[str], proof_steps: Sequence[str],
    checker_name: str, checker_version: str, checker_status: str,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a certificate only from an explicit independent checker result."""
    statement = _text(statement, "statement", 2000)
    assumptions = _text_list(assumptions, "assumptions")
    proof_steps = _text_list(proof_steps, "proof_steps", 1024)
    checker_name = _text(checker_name, "checker_name", 160)
    checker_version = _text(checker_version, "checker_version", 160)
    if checker_status not in {"pass", "fail", "not_assessed"}:
        raise ProofCertificateError("invalid_checker_status")
    if scope is not None and (not isinstance(scope, Mapping) or len(scope) > 64):
        raise ProofCertificateError("invalid_certificate_scope")
    payload = {
        "schema_version": SCHEMA_VERSION, "statement": statement,
        "assumptions": assumptions, "proof_steps": proof_steps,
        "checker": {"name": checker_name, "version": checker_version},
        "scope": dict(scope or {}),
        "status": "deductively_verified" if checker_status == "pass" else "not_assessed",
        "checker_status": checker_status,
        "policy": "deductive_certificate_is_separate_from_execution_and_empirical_tests",
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    payload["certificate_digest"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_proof_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Check certificate integrity and status without re-running its checker."""
    if not isinstance(certificate, Mapping):
        raise ProofCertificateError("certificate_must_be_mapping")
    digest = certificate.get("certificate_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ProofCertificateError("certificate_digest_required")
    payload = {key: certificate[key] for key in certificate if key != "certificate_digest"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    valid = sha256(canonical.encode("utf-8")).hexdigest() == digest
    return {"schema_version": SCHEMA_VERSION, "integrity": "pass" if valid else "fail",
            "status": certificate.get("status", "not_assessed"),
            "policy": "integrity_check_does_not_reexecute_checker"}


__all__ = ["SCHEMA_VERSION", "ProofCertificateError", "issue_proof_certificate", "verify_proof_certificate"]
