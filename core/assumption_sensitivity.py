"""Bounded what-if evaluation for UI assumption sliders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence


class AssumptionSensitivityError(ValueError):
    pass


def assess_assumption_sensitivity(base: Mapping[str, Any], variants: Sequence[Mapping[str, Any]],
                                  evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]], *,
                                  max_variants: int = 64) -> dict[str, Any]:
    if not isinstance(base, Mapping) or not callable(evaluate):
        raise AssumptionSensitivityError("base_and_evaluator_required")
    if type(max_variants) is not int or not 1 <= max_variants <= 256:
        raise AssumptionSensitivityError("invalid_variant_budget")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)) or not variants or len(variants) > max_variants:
        raise AssumptionSensitivityError("variants_out_of_bounds")
    try:
        raw_reference = evaluate(deepcopy(dict(base)))
        if not isinstance(raw_reference, Mapping):
            raise TypeError("evaluator_result_must_be_mapping")
        reference = dict(raw_reference)
    except Exception as exc:
        return {"schema_version": "mathmodel.assumption-sensitivity/v1", "status": "not_assessed",
                "reason": "reference_evaluation_failed", "error_code": type(exc).__name__}
    reference_decision = reference.get("decision")
    rows = []
    seen_ids: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping) or not isinstance(variant.get("overrides", {}), Mapping):
            raise AssumptionSensitivityError("variant_overrides_required")
        identifier = str(variant.get("id", f"variant_{index}")).strip()
        if not identifier or identifier in seen_ids:
            raise AssumptionSensitivityError("variant_id_must_be_unique")
        seen_ids.add(identifier)
        overrides = dict(variant["overrides"])
        if len(overrides) > 128 or any(not isinstance(key, str) or not key.strip() for key in overrides):
            raise AssumptionSensitivityError("variant_overrides_invalid")
        assumptions = deepcopy(dict(base)); assumptions.update(deepcopy(overrides))
        try:
            raw_result = evaluate(assumptions)
            if not isinstance(raw_result, Mapping):
                raise TypeError("evaluator_result_must_be_mapping")
            result = dict(raw_result)
            status = str(result.get("status", "assessed"))
            changed = result.get("decision") != reference_decision
            rows.append({"id": identifier,
                         "overrides": deepcopy(overrides), "status": status,
                         "decision": result.get("decision"), "changed_decision": bool(changed),
                         "objective": result.get("objective")})
        except Exception as exc:
            rows.append({"id": identifier,
                         "overrides": deepcopy(overrides), "status": "not_assessed",
                         "decision": None, "changed_decision": None, "error_code": type(exc).__name__})
    return {"schema_version": "mathmodel.assumption-sensitivity/v1", "status": "assessed",
            "reference": {"decision": reference_decision, "objective": reference.get("objective")},
            "variants": rows, "decision_change_count": sum(row["changed_decision"] is True for row in rows),
            "policy": "finite_what_if_evaluations_are_not_global_sensitivity_or_causal_effects"}


__all__ = ["AssumptionSensitivityError", "assess_assumption_sensitivity"]
