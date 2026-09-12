"""Authenticated source ledger for multi-task and unseen benchmarks."""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


class DataProvenanceError(ValueError):
    pass


def content_shingle_digests(text: str, *, shingle_size: int = 5, max_chars: int = 2_000_000) -> list[str]:
    """Return bounded normalized word-shingle fingerprints without retaining text."""
    if not isinstance(text, str) or len(text) > max_chars:
        raise DataProvenanceError("content_out_of_bounds")
    if type(shingle_size) is not int or not 2 <= shingle_size <= 32:
        raise DataProvenanceError("invalid_shingle_size")
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.casefold())
    if len(tokens) > 500_000:
        raise DataProvenanceError("token_budget_exceeded")
    return sorted({sha256(" ".join(tokens[index:index + shingle_size]).encode("utf-8")).hexdigest()
                   for index in range(max(0, len(tokens) - shingle_size + 1))})


def _digest(value: Any) -> str:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def build_source_ledger(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not 1 <= len(records) <= 10_000:
        raise DataProvenanceError("records_must_be_bounded_sequence")
    rows = []
    seen_ids, seen_digests = set(), {}
    for record in records:
        if not isinstance(record, Mapping):
            raise DataProvenanceError("source_record_must_be_object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 128:
            raise DataProvenanceError("source_id_required")
        source_id = source_id.strip()
        if source_id in seen_ids:
            raise DataProvenanceError("duplicate_source_id")
        seen_ids.add(source_id)
        uri = record.get("uri", "")
        if not isinstance(uri, str) or not uri.strip() or len(uri) > 4096 or any(ord(ch) < 32 for ch in uri):
            raise DataProvenanceError("source_uri_required")
        try:
            parsed = urlparse(uri)
            hostname = parsed.hostname
        except ValueError as exc:
            raise DataProvenanceError("source_uri_must_be_https_or_local_file") from exc
        # Credentials in a provenance URI are both a secret-leak risk and not
        # part of source identity; authentication must be handled out of band.
        if (parsed.scheme not in {"https", "file"} or
                (parsed.scheme == "https" and (not hostname or parsed.username is not None or parsed.password is not None)) or
                (parsed.scheme == "file" and parsed.netloc not in ("", "localhost"))):
            raise DataProvenanceError("source_uri_must_be_https_or_local_file")
        content_hash = record.get("content_sha256")
        if not isinstance(content_hash, str) or len(content_hash) != 64 or any(ch not in "0123456789abcdef" for ch in content_hash):
            raise DataProvenanceError("content_sha256_required")
        owner = record.get("owner")
        license_name = record.get("license")
        if not isinstance(owner, str) or not owner.strip() or not isinstance(license_name, str) or not license_name.strip():
            raise DataProvenanceError("source_owner_and_license_required")
        authorized = record.get("authorized", False)
        if type(authorized) is not bool:
            raise DataProvenanceError("authorized_must_be_boolean")
        row = {"source_id": source_id, "uri": uri.strip(), "content_sha256": content_hash,
               "owner": owner.strip(), "license": license_name.strip(), "authorized": authorized,
               "retrieved_at": str(record.get("retrieved_at", ""))[:64]}
        rows.append(row)
        seen_digests.setdefault(content_hash, []).append(source_id)
    collisions = [ids for ids in seen_digests.values() if len(ids) > 1]
    return {"schema_version": "mathmodel.data-provenance/v1", "sources": rows,
            "source_count": len(rows), "authorized_count": sum(row["authorized"] for row in rows),
            "duplicate_content_groups": collisions,
            "status": "pass" if all(row["authorized"] for row in rows) else "needs_authorization",
            "policy": "hashes_and_rights_are_evidence;_source_pages_are_not_model_truth"}


def audit_contamination(*, development_digests: Sequence[str], final_digests: Sequence[str],
                        prompt_digests: Sequence[str] = (), development_shingles: Sequence[str] = (),
                        final_shingles: Sequence[str] = (), prompt_shingles: Sequence[str] = ()) -> dict[str, Any]:
    sets = {}
    for name, values in (("development", development_digests), ("final", final_digests), ("prompt", prompt_digests)):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise DataProvenanceError(f"{name}_digests_must_be_sequence")
        normalized = set()
        for value in values:
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise DataProvenanceError("digest_must_be_sha256")
            normalized.add(value)
        sets[name] = normalized
    dev_final = sets["development"] & sets["final"]
    prompt_final = sets["prompt"] & sets["final"]
    shingle_sets = {}
    for name, values in (("development", development_shingles), ("final", final_shingles), ("prompt", prompt_shingles)):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise DataProvenanceError(f"{name}_shingles_must_be_sequence")
        normalized = set()
        for value in values:
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise DataProvenanceError("shingle_digest_must_be_sha256")
            normalized.add(value)
        shingle_sets[name] = normalized
    shingle_dev_final = shingle_sets["development"] & shingle_sets["final"]
    shingle_prompt_final = shingle_sets["prompt"] & shingle_sets["final"]
    return {"schema_version": "mathmodel.contamination-audit/v1",
            "development_final_overlap": len(dev_final), "prompt_final_overlap": len(prompt_final),
            "overlap_digests": sorted(dev_final | prompt_final),
            "development_final_shingle_overlap": len(shingle_dev_final),
            "prompt_final_shingle_overlap": len(shingle_prompt_final),
            "overlap_shingles": sorted(shingle_dev_final | shingle_prompt_final),
            "status": "fail" if dev_final or prompt_final or shingle_dev_final or shingle_prompt_final else "pass",
            "policy": "exact_and_normalized_shingle_overlap_screen;_semantic_similarity_requires_external_review"}


__all__ = ["DataProvenanceError", "build_source_ledger", "content_shingle_digests", "audit_contamination"]
