"""Auditable exports for verdicts and the final writing boundary.

The web/API layer can use these helpers to expose traceable evidence without
silently turning a preview into a conclusion.  The paper-writing payload is a
read-only projection of an already assembled verdict; it cannot invent or
edit numbers, assumptions, boundaries, or evidence references.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.verdict-export/v1"


class VerdictExportError(ValueError):
    """Raised when a verdict cannot be exported with its evidence boundary."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VerdictExportError("verdict_not_json_safe") from exc


def build_trace_bundle(*, verdict: Mapping[str, Any], inputs: Mapping[str, Any] | None = None,
                       model_ir: Mapping[str, Any] | None = None,
                       execution: Mapping[str, Any] | None = None,
                       evidence: Sequence[Mapping[str, Any]] | None = None,
                       manifest: Mapping[str, Any] | None = None,
                       certificate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble a bounded trace bundle linking conclusion to its run inputs."""
    if not isinstance(verdict, Mapping) or not verdict.get("schema_version"):
        raise VerdictExportError("verdict_schema_required")
    if len(verdict) > 256:
        raise VerdictExportError("verdict_too_large")
    refs = verdict.get("evidence_refs", [])
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        refs = []
    evidence_list = [] if evidence is None else [dict(item) for item in evidence
                                                   if isinstance(item, Mapping)][:256]
    evidence_ids = {str(item.get("id", item.get("evidence_id", ""))) for item in evidence_list}
    missing = [str(ref) for ref in refs if str(ref) not in evidence_ids]
    if certificate is not None and certificate.get("schema_version") != "mathmodel.conclusion-certificate/v1":
        raise VerdictExportError("invalid_conclusion_certificate")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": deepcopy(dict(verdict)),
        "inputs": deepcopy(dict(inputs or {})),
        "model_ir": deepcopy(dict(model_ir or {})),
        "execution": deepcopy(dict(execution or {})),
        "evidence": evidence_list,
        "manifest": deepcopy(dict(manifest or {})),
        "certificate": deepcopy(dict(certificate or {})),
        "traceability": {
            "evidence_refs_checked": len(refs),
            "missing_evidence_refs": missing,
            "status": "complete" if not missing else "incomplete",
        },
        "policy": "trace_bundle_is_audit_projection_not_a_proof",
    }
    payload["bundle_digest"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def approved_writing_projection(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Return only explicitly approved material for a paper-writing API.

    The returned object is a deep copy.  Callers cannot mutate the source
    verdict through it, and unresolved/conditional candidates are excluded.
    """
    if not isinstance(verdict, Mapping) or verdict.get("schema_version") is None:
        raise VerdictExportError("verdict_schema_required")
    approved_ids = verdict.get("approved_candidate_ids", [])
    candidates = verdict.get("candidates", [])
    if not isinstance(approved_ids, Sequence) or isinstance(approved_ids, (str, bytes)):
        raise VerdictExportError("approved_candidate_ids_required")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise VerdictExportError("candidates_required")
    by_id = {str(item.get("id")): item for item in candidates if isinstance(item, Mapping)}
    selected = []
    for identifier in approved_ids:
        item = by_id.get(str(identifier))
        if item is None or item.get("state") != "approved":
            raise VerdictExportError("approved_candidate_missing_or_not_approved")
        if not item.get("evidence_refs") and not item.get("approval_basis"):
            raise VerdictExportError("approved_candidate_requires_evidence")
        selected.append(deepcopy(dict(item)))
    if not selected:
        raise VerdictExportError("no_approved_conclusion")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_verdict_schema": str(verdict.get("schema_version")),
        "run_id": verdict.get("run_id"),
        "approved_candidates": selected,
        "assumptions": deepcopy(verdict.get("assumptions", [])),
        "counterexamples": deepcopy(verdict.get("counterexamples", [])),
        "evidence_refs": deepcopy(verdict.get("evidence_refs", [])),
        "policy": "writing_api_may_rephrase_only_approved_conclusions; numbers_and_boundaries_are_immutable",
    }


__all__ = ["SCHEMA_VERSION", "VerdictExportError", "build_trace_bundle", "approved_writing_projection"]
