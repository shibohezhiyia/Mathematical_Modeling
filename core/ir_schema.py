"""Versioned envelopes and integrity checks for intermediate representations.

The envelope is deliberately independent of any particular domain or solver.
It gives caches, evidence ledgers and transport layers one canonical hash and
an explicit migration boundary.  It does not validate mathematical meaning;
the relevant IR validator must still run after unwrapping.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.ir-envelope/v1"
SUPPORTED_KINDS = frozenset({"hypothesis_ir", "problem_contract", "search_experiment", "primitive_graph"})


class IRSchemaError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _walk(value: Any, *, depth: int = 0, count: list[int] | None = None) -> None:
    count = [0] if count is None else count
    count[0] += 1
    if depth > 24 or count[0] > 50_000:
        raise IRSchemaError("ir_complexity_limit")
    if type(value) is dict:
        if len(value) > 10_000 or any(type(key) is not str or len(key) > 256 for key in value):
            raise IRSchemaError("ir_object_limit")
        for item in value.values():
            _walk(item, depth=depth + 1, count=count)
    elif type(value) is list:
        if len(value) > 50_000:
            raise IRSchemaError("ir_array_limit")
        for item in value:
            _walk(item, depth=depth + 1, count=count)
    elif type(value) in (float, int):
        if isinstance(value, float) and not math.isfinite(value):
            raise IRSchemaError("ir_nonfinite_number")
    elif value is not None and type(value) not in (str, bool):
        raise IRSchemaError("ir_non_json_value")


def canonical_json(value: Any) -> str:
    _walk(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IRSchemaError("ir_not_json") from exc
    if len(encoded.encode("utf-8")) > 2_000_000:
        raise IRSchemaError("ir_payload_limit")
    return encoded


def ir_digest(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        raise IRSchemaError("ir_payload_must_be_object")
    return sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def wrap_ir(payload: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    if kind not in SUPPORTED_KINDS:
        raise IRSchemaError("unsupported_ir_kind")
    body = dict(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "payload": body,
        "payload_digest": ir_digest(body),
    }


def migrate_ir(value: Mapping[str, Any], *, kind: str | None = None) -> dict[str, Any]:
    """Normalize a current envelope or explicitly wrap a legacy bare payload.

    Legacy migration is intentionally lossless: the original mapping is copied
    as the payload, and no fields, operators or assumptions are invented.
    """
    if not isinstance(value, Mapping):
        raise IRSchemaError("ir_must_be_object")
    if value.get("schema_version") == SCHEMA_VERSION:
        expected = {"schema_version", "kind", "payload", "payload_digest"}
        if set(value) != expected or value.get("kind") not in SUPPORTED_KINDS:
            raise IRSchemaError("invalid_ir_envelope")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise IRSchemaError("ir_payload_must_be_object")
        digest = value.get("payload_digest")
        if digest != ir_digest(payload):
            raise IRSchemaError("ir_digest_mismatch")
        return json.loads(canonical_json(dict(value)))
    if kind is None or kind not in SUPPORTED_KINDS:
        raise IRSchemaError("legacy_ir_kind_required")
    return wrap_ir(value, kind=kind)


def unwrap_ir(value: Mapping[str, Any], *, expected_kind: str | None = None) -> dict[str, Any]:
    envelope = migrate_ir(value, kind=expected_kind)
    if expected_kind is not None and envelope["kind"] != expected_kind:
        raise IRSchemaError("ir_kind_mismatch")
    return dict(envelope["payload"])


__all__ = [
    "SCHEMA_VERSION", "SUPPORTED_KINDS", "IRSchemaError", "canonical_json",
    "ir_digest", "wrap_ir", "migrate_ir", "unwrap_ir",
]
