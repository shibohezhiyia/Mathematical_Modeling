"""Audit whether an evaluation source can support unseen-problem claims.

Public problem collections are useful regression inputs, but they cannot by
themselves support a claim about *real unseen* accuracy.  This module keeps
that distinction machine-readable and deliberately separates source-level
eligibility from the case-level sealed/statistical protocol in
``core.blind_statistics``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


AUDIT_SCHEMA = "mathmodel.evaluation-source-audit/v1"
_DOMAINS = frozenset({"low", "medium", "high"})
_ACCESS = frozenset({"public", "private", "restricted"})
_BOOL_FIELDS = frozenset({
    "private_holdout", "independent_reference", "expert_review",
    "open_ended", "separate_from_development",
})


class EvaluationSourceAuditError(ValueError):
    """Raised when source evidence is incomplete or contradictory."""


def _text(value: Any, code: str, limit: int = 240) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise EvaluationSourceAuditError(code)
    return value.strip()


def _url(value: Any) -> str:
    value = _text(value, "source_url_invalid", 2000)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise EvaluationSourceAuditError("source_url_invalid")
    return value


@dataclass(frozen=True)
class EvaluationSourceAssessment:
    source_id: str
    title: str
    url: str
    domain_match: str
    access: str
    private_holdout: bool
    independent_reference: bool
    expert_review: bool
    open_ended: bool
    separate_from_development: bool
    notes: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvaluationSourceAssessment":
        if not isinstance(payload, Mapping):
            raise EvaluationSourceAuditError("source_must_be_object")
        required = {
            "id", "title", "url", "domain_match", "access",
            *_BOOL_FIELDS,
        }
        if set(payload) - required - {"notes"} or not required.issubset(payload):
            raise EvaluationSourceAuditError("source_fields_invalid")
        source_id = _text(payload["id"], "source_id_invalid", 120)
        if any(ch.isspace() for ch in source_id):
            raise EvaluationSourceAuditError("source_id_invalid")
        domain_match = _text(payload["domain_match"], "domain_match_invalid")
        access = _text(payload["access"], "source_access_invalid")
        if domain_match not in _DOMAINS:
            raise EvaluationSourceAuditError("domain_match_invalid")
        if access not in _ACCESS:
            raise EvaluationSourceAuditError("source_access_invalid")
        flags: dict[str, bool] = {}
        for field in _BOOL_FIELDS:
            value = payload[field]
            if type(value) is not bool:
                raise EvaluationSourceAuditError(f"{field}_must_be_boolean")
            flags[field] = value
        # A source cannot be declared private while access is public.  The
        # reverse is allowed (a private holdout can expose metadata publicly).
        if flags["private_holdout"] and access == "public":
            raise EvaluationSourceAuditError("private_holdout_public_access_conflict")
        notes = payload.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 2000:
            raise EvaluationSourceAuditError("source_notes_invalid")
        return cls(
            source_id=source_id,
            title=_text(payload["title"], "source_title_invalid"),
            url=_url(payload["url"]),
            domain_match=domain_match,
            access=access,
            private_holdout=flags["private_holdout"],
            independent_reference=flags["independent_reference"],
            expert_review=flags["expert_review"],
            open_ended=flags["open_ended"],
            separate_from_development=flags["separate_from_development"],
            notes=notes.strip(),
        )

    @property
    def accuracy_eligible(self) -> bool:
        """Whether this source has the minimum source-level evidence.

        This does not bypass case sealing, score unlocking, sample-size or
        statistical gates.  It only answers whether the source is worth
        entering the blind protocol.
        """
        return bool(
            self.domain_match == "high"
            and self.private_holdout
            and self.independent_reference
            and self.expert_review
            and self.separate_from_development
        )

    @property
    def suitability(self) -> str:
        if self.accuracy_eligible:
            return "candidate_for_blind_protocol"
        if self.private_holdout and self.independent_reference and self.expert_review:
            return "methodology_reference_only"
        return "public_or_insufficient_evidence"

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "title": self.title,
            "url": self.url,
            "domain_match": self.domain_match,
            "access": self.access,
            "private_holdout": self.private_holdout,
            "independent_reference": self.independent_reference,
            "expert_review": self.expert_review,
            "open_ended": self.open_ended,
            "separate_from_development": self.separate_from_development,
            "suitability": self.suitability,
            "accuracy_eligible": self.accuracy_eligible,
            "statistical_claim_eligible": False,
            "notes": self.notes,
        }


class EvaluationSourceRegistry:
    """Validated source audit with conservative claim eligibility."""

    def __init__(self, sources: Iterable[EvaluationSourceAssessment]):
        self.sources = tuple(sources)
        if not self.sources:
            raise EvaluationSourceAuditError("source_registry_empty")
        ids = [item.source_id for item in self.sources]
        if len(ids) != len(set(ids)):
            raise EvaluationSourceAuditError("duplicate_source_id")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EvaluationSourceRegistry":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != AUDIT_SCHEMA:
            raise EvaluationSourceAuditError("source_registry_schema_invalid")
        rows = payload.get("sources")
        if not isinstance(rows, list):
            raise EvaluationSourceAuditError("source_registry_sources_invalid")
        return cls(EvaluationSourceAssessment.from_payload(row) for row in rows)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationSourceRegistry":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationSourceAuditError("source_registry_unreadable") from exc
        return cls.from_payload(payload)

    def eligible(self) -> tuple[EvaluationSourceAssessment, ...]:
        return tuple(item for item in self.sources if item.accuracy_eligible)

    def get(self, source_id: str) -> EvaluationSourceAssessment:
        for item in self.sources:
            if item.source_id == source_id:
                return item
        raise EvaluationSourceAuditError("source_not_found")

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA,
            "source_count": len(self.sources),
            "eligible_source_count": len(self.eligible()),
            "sources": [item.public_metadata() for item in self.sources],
            "policy": (
                "source_audit_is_not_a_gold_truth_or_significance_test; "
                "case_level_sealing_and_independent_unlocked_scoring_required"
            ),
        }


__all__ = [
    "AUDIT_SCHEMA", "EvaluationSourceAssessment", "EvaluationSourceAuditError",
    "EvaluationSourceRegistry",
]
