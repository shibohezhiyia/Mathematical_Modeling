"""Evidence-ranked proposal backend routing with one shared validation gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Callable, Mapping, Sequence


class ProposalRouterError(ValueError):
    pass


@dataclass(frozen=True)
class BackendProfile:
    name: str
    backend: Any
    legal_rate: float
    repair_gain: float
    latency_ms: float
    cost_per_call: float
    max_calls: int = 1

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip() or len(self.name) > 128:
            raise ProposalRouterError("invalid_backend_name")
        if not callable(getattr(self.backend, "complete", None)):
            raise ProposalRouterError("backend_complete_required")
        for field in ("legal_rate", "repair_gain", "latency_ms", "cost_per_call"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
                raise ProposalRouterError(f"invalid_{field}")
        if float(self.legal_rate) > 1 or float(self.repair_gain) > 1:
            raise ProposalRouterError("rate_must_be_between_zero_and_one")
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 16:
            raise ProposalRouterError("invalid_backend_call_budget")


def route_proposal_backends(
    profiles: Sequence[BackendProfile], messages: Sequence[Mapping[str, Any]],
    validator: Callable[[str], Mapping[str, Any]], *, max_calls: int = 8,
) -> dict[str, Any]:
    """Try ranked proposal backends; only the validator can accept a result."""
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)) or not profiles or len(profiles) > 32:
        raise ProposalRouterError("profiles_required")
    if any(not isinstance(item, BackendProfile) for item in profiles):
        raise ProposalRouterError("profiles_must_be_unique")
    if len({item.name for item in profiles}) != len(profiles):
        raise ProposalRouterError("profiles_must_be_unique")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or len(messages) > 64:
        raise ProposalRouterError("messages_invalid")
    if not callable(validator):
        raise ProposalRouterError("validator_required")
    if type(max_calls) is not int or not 1 <= max_calls <= 32:
        raise ProposalRouterError("invalid_total_call_budget")
    ranked = sorted(profiles, key=lambda item: (-float(item.legal_rate), -float(item.repair_gain),
                                                  float(item.latency_ms), float(item.cost_per_call), item.name))
    attempts: list[dict[str, Any]] = []
    calls = 0
    for profile in ranked:
        for _ in range(profile.max_calls):
            if calls >= max_calls:
                break
            calls += 1
            try:
                response = profile.backend.complete(messages)
                if not isinstance(response, str) or not response.strip() or len(response.encode("utf-8")) > 2_000_000:
                    raise ProposalRouterError("backend_response_invalid")
                verdict = validator(response)
                if not isinstance(verdict, Mapping) or verdict.get("status") not in {"accepted", "rejected", "not_assessed"}:
                    raise ProposalRouterError("validator_result_invalid")
                item = {"backend": profile.name, "status": verdict["status"],
                        "response_digest": sha256(response.encode("utf-8")).hexdigest(),
                        "diagnostic": str(verdict.get("diagnostic", ""))[:500]}
                attempts.append(item)
                if verdict["status"] == "accepted":
                    return {"schema_version": "mathmodel.proposal-router/v1", "status": "accepted",
                            "backend": profile.name, "response": response, "attempts": attempts,
                            "policy": "ranking_is_declared_evidence_only;_shared_validator_is_authoritative"}
            except Exception as exc:
                attempts.append({"backend": profile.name, "status": "not_assessed",
                                 "error": type(exc).__name__})
        if calls >= max_calls:
            break
    return {"schema_version": "mathmodel.proposal-router/v1", "status": "not_assessed",
            "backend": None, "response": None, "attempts": attempts,
            "policy": "no_backend_response_is_authorized_without_shared_validation"}


__all__ = ["ProposalRouterError", "BackendProfile", "route_proposal_backends"]
