"""Keep parameter what-if, scenario and causal intervention claims separate."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.scenario-protocol/v1"
_KINDS = frozenset({"parameter_perturbation", "scenario_analysis", "causal_intervention"})


class ScenarioProtocolError(ValueError):
    pass


def build_scenario_contract(kind: Any, values: Mapping[str, Any], *, identification_assumptions: list[str] | None = None,
                            design: str | None = None) -> dict[str, Any]:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ScenarioProtocolError("scenario_kind_invalid")
    if not isinstance(values, Mapping) or not values:
        raise ScenarioProtocolError("scenario_values_must_be_nonempty_object")
    try:
        encoded = json.dumps(dict(values), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ScenarioProtocolError("scenario_values_must_be_finite_json") from exc
    if len(encoded) > 32_000:
        raise ScenarioProtocolError("scenario_values_too_large")
    assumptions = identification_assumptions or []
    if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item.strip() for item in assumptions):
        raise ScenarioProtocolError("identification_assumptions_invalid")
    if design is not None and (not isinstance(design, str) or len(design) > 500):
        raise ScenarioProtocolError("causal_design_invalid")
    identified = kind == "causal_intervention" and bool(assumptions) and bool(str(design or "").strip())
    status = "identified_proposal" if identified else ("not_assessed" if kind == "causal_intervention" else "scenario_proposal")
    return {
        "schema_version": SCHEMA_VERSION, "kind": kind, "values": dict(values),
        "identification_assumptions": list(assumptions), "design": design,
        "status": status, "causal_claim_authorized": False,
        "scope": "causal_effect_only_if_external_identification_and_design_evidence_is_checked",
        "contract_digest": sha256(encoded).hexdigest(),
    }


__all__ = ["SCHEMA_VERSION", "ScenarioProtocolError", "build_scenario_contract"]
