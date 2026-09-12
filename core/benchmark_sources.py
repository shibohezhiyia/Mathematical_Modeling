"""Machine-readable registry for external mathematical-modeling sources.

The registry deliberately separates *problem inputs*, *rubric material*,
*method comparisons* and *community reference solutions*.  A public winning
paper is useful for human review, but it is not an independent gold answer:
different valid models can produce different conclusions on an open-ended
contest problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


SOURCE_SCHEMA = "mathmodel.benchmark-source-registry/v1"
_KINDS = frozenset({
    "problem_input", "rubric", "research_benchmark", "reference_solution",
    "archive", "rules",
})
_ROLES = frozenset({
    "task_input", "rubric_material", "method_comparison", "reference_only",
    "policy_reference",
})
_ACCESS = frozenset({"public", "membership_required", "repository_access"})
_AUTHORITIES = frozenset({"official", "research", "community"})
_LICENSE = frozenset({"official_public", "metadata_only", "repository_license_required"})


class BenchmarkSourceError(ValueError):
    """Raised when a source entry would make benchmark provenance ambiguous."""


def _text(value: Any, code: str, *, max_length: int = 240) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > max_length:
        raise BenchmarkSourceError(code)
    return value.strip()


def _url(value: Any) -> str:
    value = _text(value, "source_url_required", max_length=2000)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BenchmarkSourceError("source_url_invalid")
    return value


@dataclass(frozen=True)
class BenchmarkSource:
    source_id: str
    kind: str
    title: str
    url: str
    authority: str
    role: str
    access: str
    license_status: str
    notes: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BenchmarkSource":
        if not isinstance(payload, Mapping):
            raise BenchmarkSourceError("source_must_be_object")
        required = {
            "id", "kind", "title", "url", "authority", "role", "access",
            "license_status",
        }
        if set(payload) - required - {"notes"} or not required.issubset(payload):
            raise BenchmarkSourceError("source_fields_invalid")
        source_id = _text(payload["id"], "source_id_invalid", max_length=120)
        if any(ch.isspace() for ch in source_id):
            raise BenchmarkSourceError("source_id_invalid")
        kind = _text(payload["kind"], "source_kind_invalid")
        authority = _text(payload["authority"], "source_authority_invalid")
        role = _text(payload["role"], "source_role_invalid")
        access = _text(payload["access"], "source_access_invalid")
        license_status = _text(payload["license_status"], "source_license_invalid")
        if kind not in _KINDS:
            raise BenchmarkSourceError("source_kind_invalid")
        if authority not in _AUTHORITIES:
            raise BenchmarkSourceError("source_authority_invalid")
        if role not in _ROLES:
            raise BenchmarkSourceError("source_role_invalid")
        if access not in _ACCESS:
            raise BenchmarkSourceError("source_access_invalid")
        if license_status not in _LICENSE:
            raise BenchmarkSourceError("source_license_invalid")
        if kind == "reference_solution" and role != "reference_only":
            raise BenchmarkSourceError("reference_solution_cannot_be_gold_truth")
        if role == "task_input" and kind not in {"problem_input", "archive"}:
            raise BenchmarkSourceError("task_input_kind_invalid")
        if role == "rubric_material" and kind not in {"rubric", "rules"}:
            raise BenchmarkSourceError("rubric_kind_invalid")
        if authority == "official" and license_status == "repository_license_required":
            raise BenchmarkSourceError("official_repository_license_mismatch")
        notes = payload.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 2000:
            raise BenchmarkSourceError("source_notes_invalid")
        return cls(
            source_id=source_id, kind=kind, title=_text(payload["title"], "source_title_invalid"),
            url=_url(payload["url"]), authority=authority, role=role, access=access,
            license_status=license_status, notes=notes.strip(),
        )


class BenchmarkSourceRegistry:
    """Validated registry with role-based lookup, never a truth oracle."""

    def __init__(self, sources: Iterable[BenchmarkSource]):
        items = tuple(sources)
        if not items:
            raise BenchmarkSourceError("source_registry_empty")
        ids = [item.source_id for item in items]
        if len(set(ids)) != len(ids):
            raise BenchmarkSourceError("duplicate_source_id")
        self.sources = items

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BenchmarkSourceRegistry":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SOURCE_SCHEMA:
            raise BenchmarkSourceError("source_registry_schema_invalid")
        rows = payload.get("sources")
        if not isinstance(rows, list):
            raise BenchmarkSourceError("source_registry_sources_invalid")
        return cls(BenchmarkSource.from_payload(row) for row in rows)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSourceRegistry":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkSourceError("source_registry_unreadable") from exc
        return cls.from_payload(payload)

    def by_role(self, role: str) -> tuple[BenchmarkSource, ...]:
        if role not in _ROLES:
            raise BenchmarkSourceError("source_role_invalid")
        return tuple(item for item in self.sources if item.role == role)

    def get(self, source_id: str) -> BenchmarkSource:
        for item in self.sources:
            if item.source_id == source_id:
                return item
        raise BenchmarkSourceError("source_not_found")

    def public_metadata(self) -> dict[str, Any]:
        """Return safe metadata; this never downloads or embeds source content."""
        return {
            "schema_version": SOURCE_SCHEMA,
            "source_count": len(self.sources),
            "sources": [
                {"id": item.source_id, "kind": item.kind, "title": item.title,
                 "url": item.url, "authority": item.authority, "role": item.role,
                 "access": item.access, "license_status": item.license_status,
                 "notes": item.notes}
                for item in self.sources
            ],
            "gold_truth_policy": "no_public_source_is_gold_truth; independent_review_or_sealed_reference_required",
        }


__all__ = ["BenchmarkSource", "BenchmarkSourceError", "BenchmarkSourceRegistry", "SOURCE_SCHEMA"]
