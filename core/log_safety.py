"""Bounded diagnostics for persisted logs and model feedback.

Internal exceptions may contain paths, keys, or user rows. This module creates
a small public diagnostic envelope instead of forwarding the raw exception.
It is a defense-in-depth policy, not an OS security boundary.
"""
from __future__ import annotations

from typing import Any, Mapping

from .experience_store import redact_payload


class LogSafetyError(ValueError):
    pass


_ALLOWED = {"code", "category", "message", "node_id", "operation", "counterexample", "resource", "retryable"}


def safe_diagnostic(value: Mapping[str, Any], *, max_bytes: int = 16_384) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not 128 <= int(max_bytes) <= 1_000_000:
        raise LogSafetyError("invalid_diagnostic_contract")
    clean: dict[str, Any] = {}
    for key in _ALLOWED:
        if key not in value:
            continue
        item = value[key]
        if key == "message" and item is not None:
            item = str(item).splitlines()[0][:512]
        if key in {"counterexample", "resource"} and isinstance(item, Mapping):
            item = redact_payload(dict(item))
        clean[key] = item
    clean.setdefault("code", "diagnostic_unclassified")
    clean.setdefault("category", "unknown")
    clean["traceback_included"] = False
    clean["raw_input_included"] = False
    import json
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) > max_bytes:
        clean["counterexample"] = {"truncated": True}
        clean["resource"] = {"truncated": True}
        encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) > max_bytes:
        raise LogSafetyError("diagnostic_budget_exceeded")
    return clean


__all__ = ["LogSafetyError", "safe_diagnostic"]
