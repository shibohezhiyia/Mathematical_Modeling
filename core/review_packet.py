"""Build an anonymised blind-review packet for frozen modeling runs.

This is a practical fallback when an organiser cannot provide private tasks or
gold answers.  It deliberately produces rubric rows, not correctness labels.
The original case IDs remain in a separate private mapping so reviewers can be
shown outputs without being told the source year/problem number.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from .blind_benchmark import BLIND_CASE_SCHEMA, RUN_SCHEMA, BlindBenchmarkError, BlindManifest
from .rubric_review import RubricReviewError, aggregate_reviews, record_review


PACKET_SCHEMA = "mathmodel.review-packet/v1"
_MAX_CASES = 10_000


class ReviewPacketError(ValueError):
    """Raised when a review packet would be ambiguous or leak source identity."""


def _review_id(manifest_digest: str, case_id: str, ordinal: int) -> str:
    digest = sha256(f"{manifest_digest}:{case_id}".encode("utf-8")).hexdigest()[:12]
    return f"review-{ordinal:04d}-{digest}"


def build_review_packet(
    manifest: BlindManifest | Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rubric: Mapping[str, Any],
    *,
    output_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create an anonymised packet and a private case mapping.

    ``output_paths`` is optional and is never exposed in the public packet;
    reviewers receive only opaque IDs.  The returned ``private_mapping`` must
    be stored outside the public repository.
    """
    if not isinstance(manifest, BlindManifest):
        try:
            manifest = BlindManifest.from_payload(manifest)
        except BlindBenchmarkError as exc:
            raise ReviewPacketError(str(exc)) from exc
    if not isinstance(rubric, Mapping) or rubric.get("schema_version") != "mathmodel.review-rubric/v1":
        raise ReviewPacketError("rubric_required")
    criteria = rubric.get("criteria")
    scale = rubric.get("scale")
    if not isinstance(criteria, list) or not criteria or any(type(item) is not str or not item.strip() for item in criteria):
        raise ReviewPacketError("rubric_criteria_invalid")
    if not isinstance(scale, Mapping) or type(scale.get("min")) is not int or type(scale.get("max")) is not int or scale["min"] != 0 or scale["max"] < 1:
        raise ReviewPacketError("rubric_scale_invalid")
    if type(rubric.get("minimum_reviewers")) is not int or not 2 <= rubric["minimum_reviewers"] <= 8:
        raise ReviewPacketError("rubric_minimum_reviewers_invalid")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise ReviewPacketError("runs_required")
    if len(runs) > _MAX_CASES:
        raise ReviewPacketError("too_many_runs")
    case_ids = {case.case_id for case in manifest.cases()}
    seen: set[str] = set()
    clean_runs: dict[str, Mapping[str, Any]] = {}
    for row in runs:
        if not isinstance(row, Mapping) or row.get("schema_version") != RUN_SCHEMA:
            raise ReviewPacketError("invalid_run")
        if row.get("manifest_digest") != manifest.digest:
            raise ReviewPacketError("run_scope_mismatch")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in case_ids or case_id in seen:
            raise ReviewPacketError("duplicate_or_unknown_case")
        if row.get("status") not in {"completed", "failed", "blocked", "timed_out", "not_run"}:
            raise ReviewPacketError("invalid_run_status")
        seen.add(case_id)
        clean_runs[case_id] = row
    paths = output_paths or {}
    if not isinstance(paths, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in paths.items()):
        raise ReviewPacketError("output_paths_invalid")
    cases = []
    private_mapping = []
    for ordinal, case in enumerate(manifest.cases(), start=1):
        opaque = _review_id(manifest.digest, case.case_id, ordinal)
        run = clean_runs.get(case.case_id)
        cases.append({
            "review_case_id": opaque,
            "run_status": run.get("status", "not_run") if run else "not_run",
            "score_fields": list(criteria),
            "notes_allowed": True,
        })
        private_mapping.append({
            "review_case_id": opaque,
            "case_id": case.case_id,
            "run_id": run.get("run_id") if run else None,
            "output_path": paths.get(case.case_id),
        })
    return {
        "packet": {
            "schema_version": PACKET_SCHEMA,
            "manifest_digest": manifest.digest,
            "rubric_schema_version": rubric["schema_version"],
            "case_count": len(cases),
            "cases": cases,
            "policy": "descriptive_independent_blind_review_only;_not_gold_truth;not_real_unseen_accuracy",
        },
        "private_mapping": {
            "schema_version": "mathmodel.review-packet-private/v1",
            "manifest_digest": manifest.digest,
            "mappings": private_mapping,
            "policy": "keep_outside_public_repository_and_share_only_with_review_coordinator",
        },
    }


def build_review_form(packet: Mapping[str, Any], rubric: Mapping[str, Any], *, reviewer_id: str) -> dict[str, Any]:
    """Create an empty form for one reviewer; scores must be filled manually."""
    if not isinstance(packet, Mapping) or packet.get("schema_version") != PACKET_SCHEMA:
        raise ReviewPacketError("packet_required")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip() or len(reviewer_id) > 120 or any(ch.isspace() for ch in reviewer_id):
        raise ReviewPacketError("reviewer_id_invalid")
    criteria = rubric.get("criteria") if isinstance(rubric, Mapping) else None
    if not isinstance(criteria, list) or not criteria:
        raise ReviewPacketError("rubric_criteria_invalid")
    rows = []
    for case in packet.get("cases", []):
        if not isinstance(case, Mapping) or not isinstance(case.get("review_case_id"), str):
            raise ReviewPacketError("packet_case_invalid")
        rows.append({"review_case_id": case["review_case_id"], "scores": {name: None for name in criteria}, "notes": ""})
    return {
        "schema_version": "mathmodel.review-form/v1",
        "packet_schema_version": PACKET_SCHEMA,
        "manifest_digest": packet.get("manifest_digest"),
        "reviewer_id": reviewer_id.strip(),
        "rows": rows,
        "policy": "reviewer_must_not_infer_case_identity;_scores_are_descriptive_until_aggregate_and_adjudication",
    }


def aggregate_review_forms(
    packet: Mapping[str, Any], private_mapping: Mapping[str, Any], rubric: Mapping[str, Any],
    forms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate completed anonymous forms and aggregate them by original case.

    The private mapping is required only at the coordinator side.  It must
    never be sent to reviewers or committed with the public packet.
    """
    if not isinstance(packet, Mapping) or packet.get("schema_version") != PACKET_SCHEMA:
        raise ReviewPacketError("packet_required")
    if not isinstance(private_mapping, Mapping) or private_mapping.get("schema_version") != "mathmodel.review-packet-private/v1":
        raise ReviewPacketError("private_mapping_required")
    if packet.get("manifest_digest") != private_mapping.get("manifest_digest"):
        raise ReviewPacketError("mapping_scope_mismatch")
    mapping_rows = private_mapping.get("mappings")
    if not isinstance(mapping_rows, list):
        raise ReviewPacketError("mapping_rows_invalid")
    mapping: dict[str, str] = {}
    for row in mapping_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("review_case_id"), str) or not isinstance(row.get("case_id"), str):
            raise ReviewPacketError("mapping_row_invalid")
        if row["review_case_id"] in mapping:
            raise ReviewPacketError("duplicate_review_case_id")
        mapping[row["review_case_id"]] = row["case_id"]
    if not isinstance(forms, Sequence) or isinstance(forms, (str, bytes)) or not forms:
        raise ReviewPacketError("forms_required")
    reviews = []
    seen: set[tuple[str, str]] = set()
    for form in forms:
        if not isinstance(form, Mapping) or form.get("schema_version") != "mathmodel.review-form/v1" or form.get("packet_schema_version") != PACKET_SCHEMA:
            raise ReviewPacketError("form_invalid")
        if form.get("manifest_digest") != packet.get("manifest_digest"):
            raise ReviewPacketError("form_scope_mismatch")
        reviewer_id = form.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ReviewPacketError("reviewer_id_invalid")
        rows = form.get("rows")
        if not isinstance(rows, list):
            raise ReviewPacketError("form_rows_invalid")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("review_case_id") not in mapping:
                raise ReviewPacketError("unknown_review_case_id")
            key = (row["review_case_id"], reviewer_id.strip())
            if key in seen:
                raise ReviewPacketError("duplicate_review")
            seen.add(key)
            try:
                reviews.append(record_review(
                    rubric, case_id=mapping[row["review_case_id"]], evaluator_id=reviewer_id,
                    scores=row.get("scores", {}), notes=row.get("notes", ""),
                ))
            except RubricReviewError as exc:
                raise ReviewPacketError(str(exc)) from exc
    try:
        return aggregate_reviews(rubric, reviews)
    except RubricReviewError as exc:
        raise ReviewPacketError(str(exc)) from exc


__all__ = ["PACKET_SCHEMA", "ReviewPacketError", "build_review_packet", "build_review_form", "aggregate_review_forms"]
