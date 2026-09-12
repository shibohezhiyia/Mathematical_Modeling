"""Bind metadata-only holdout intake to the sealed blind manifest."""

from __future__ import annotations

from typing import Any, Mapping

from .blind_benchmark import BlindBenchmarkError, BlindManifest
from .holdout_intake import HoldoutIntakeError, validate_holdout_intake


class HoldoutBindingError(ValueError):
    pass


def bind_holdout_intake(
    intake: Mapping[str, Any], manifest: BlindManifest | Mapping[str, Any],
    *, min_cases: int = 20, min_families: int = 4,
) -> dict[str, Any]:
    """Verify every intake commitment matches the sealed public manifest."""
    try:
        intake_result = validate_holdout_intake(intake, min_cases=min_cases, min_families=min_families)
    except HoldoutIntakeError as exc:
        raise HoldoutBindingError(f"intake_invalid:{exc}") from exc
    if intake_result["status"] != "ready_for_sealing":
        raise HoldoutBindingError("intake_not_ready")
    if not isinstance(manifest, BlindManifest):
        if not isinstance(manifest, Mapping):
            raise HoldoutBindingError("manifest_required")
        try:
            manifest = BlindManifest.from_payload(dict(manifest))
        except BlindBenchmarkError as exc:
            raise HoldoutBindingError(f"manifest_invalid:{exc}") from exc
    cases_by_id = {row["id"]: row for row in manifest.public().get("cases", [])}
    intake_cases = {row["id"]: row for row in intake.get("cases", [])}
    if set(cases_by_id) != set(intake_cases):
        raise HoldoutBindingError("case_id_set_mismatch")
    for case_id, row in intake_cases.items():
        sealed = cases_by_id[case_id]
        if sealed.get("split") != "unseen" or sealed.get("provenance") != "external_real":
            raise HoldoutBindingError("sealed_case_not_external_unseen")
        for field in ("statement_sha256", "statement_bytes", "answer_sha256", "answer_bytes"):
            if sealed.get(field) != row.get(field):
                raise HoldoutBindingError(f"{field}_mismatch")
        if sealed.get("attachment_sha256", []) != row.get("attachment_sha256", []):
            raise HoldoutBindingError("attachment_digest_mismatch")
        if sealed.get("family") != row.get("family"):
            raise HoldoutBindingError("case_family_mismatch")
    return {
        "schema_version": "mathmodel.holdout-binding/v1",
        "status": "bound",
        "protocol_id": intake_result["protocol_id"],
        "manifest_digest": manifest.digest,
        "intake_metadata_digest": intake_result["metadata_digest"],
        "case_count": len(cases_by_id),
        "family_count": len({row["family"] for row in intake_cases.values()}),
        "policy": "metadata_only;sealed_manifest_matches_intake_commitments;statistical_gates_still_required",
    }


__all__ = ["HoldoutBindingError", "bind_holdout_intake"]
