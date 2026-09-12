"""Explicit cache identity contracts for reproducible modeling runs.

The cache is an optimisation only.  A cache key must describe every input that
can change an intermediate result; it must never be used as evidence that a
model is valid.  This module keeps the contract in one place so individual
backends do not silently omit a split, tolerance or random seed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.cache-contract/v1"


class CacheContractError(ValueError):
    """Raised when a cache identity is incomplete or not deterministic."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CacheContractError("cache_identity_not_json_safe") from exc


def _text(value: Any, name: str, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise CacheContractError(f"invalid_{name}")
    return value.strip()


@dataclass(frozen=True)
class CacheIdentity:
    """All declared dimensions of a deterministic intermediate computation."""

    ir_digest: str
    contract_digest: str
    data_view_digest: str
    preprocessing_digest: str
    solver_version: str
    tolerance: float
    seed: int
    model: str | None = None
    prompt_digest: str | None = None
    config_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("ir_digest", "contract_digest", "data_view_digest", "preprocessing_digest",
                     "solver_version"):
            _text(getattr(self, name), name)
        if not isinstance(self.tolerance, (int, float)) or isinstance(self.tolerance, bool):
            raise CacheContractError("invalid_tolerance")
        if not math.isfinite(float(self.tolerance)) or float(self.tolerance) <= 0:
            raise CacheContractError("invalid_tolerance")
        if type(self.seed) is not int or self.seed < 0 or self.seed > 2**63 - 1:
            raise CacheContractError("invalid_seed")
        for name in ("model", "prompt_digest", "config_digest"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)

    def payload(self, *, api_response: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        value["tolerance"] = float(self.tolerance)
        value["kind"] = "api_response" if api_response else "intermediate"
        if api_response and (not self.model or not self.prompt_digest or not self.config_digest):
            raise CacheContractError("api_cache_requires_model_prompt_and_config")
        if not api_response:
            value.pop("model", None)
            value.pop("prompt_digest", None)
            value.pop("config_digest", None)
        return value

    def key(self, *, api_response: bool = False) -> str:
        return hashlib.sha256(_canonical(self.payload(api_response=api_response)).encode("utf-8")).hexdigest()


def build_cache_key(*, ir_digest: str, contract_digest: str, data_view_digest: str,
                    preprocessing_digest: str, solver_version: str, tolerance: float,
                    seed: int, model: str | None = None, prompt_digest: str | None = None,
                    config_digest: str | None = None, api_response: bool = False) -> str:
    """Build a cache key whose identity is explicit and reviewable."""
    return CacheIdentity(ir_digest, contract_digest, data_view_digest,
                         preprocessing_digest, solver_version, tolerance, seed,
                         model, prompt_digest, config_digest).key(api_response=api_response)


__all__ = ["SCHEMA_VERSION", "CacheContractError", "CacheIdentity", "build_cache_key"]
