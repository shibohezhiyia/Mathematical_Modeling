"""Evidence-gated ablation of variables or mechanisms.

The evaluator is supplied by the domain backend.  This module only enforces
bounded comparisons and conservative conclusions; it never edits a model or
interprets a small loss difference as a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence


class SimplificationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SimplificationConfig:
    max_candidates: int = 64
    objective_tolerance: float = 1e-6
    stress_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 512:
            raise SimplificationError("invalid_simplification_budget")
        for name, value in (("objective_tolerance", self.objective_tolerance),
                            ("stress_tolerance", self.stress_tolerance)):
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise SimplificationError(f"invalid_{name}")


def _metric(result: Mapping[str, Any], key: str) -> float:
    value = result.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise SimplificationError(f"missing_or_nonfinite_{key}")
    return float(value)


def _evaluate(evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]], model: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = evaluate(model)
    except Exception:
        raise SimplificationError("evaluation_failed")
    if not isinstance(result, Mapping):
        raise SimplificationError("evaluation_result_must_be_object")
    feasible = result.get("feasible")
    if type(feasible) is not bool:
        raise SimplificationError("feasibility_must_be_bool")
    return {
        "objective_loss": _metric(result, "objective_loss"),
        "feasible": feasible,
        "stress_objective_loss": _metric(result, "stress_objective_loss")
        if "stress_objective_loss" in result else None,
    }


def assess_simplifications(
    model: Mapping[str, Any],
    removable_ids: Sequence[str],
    remove: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    config: SimplificationConfig | None = None,
) -> dict[str, Any]:
    """Compare bounded ablations against a full model and return evidence."""
    if not isinstance(model, Mapping) or not callable(remove) or not callable(evaluate):
        raise SimplificationError("invalid_simplification_inputs")
    config = config or SimplificationConfig()
    ids = list(dict.fromkeys(removable_ids))
    if not ids or len(ids) > config.max_candidates or any(type(item) is not str or not item for item in ids):
        raise SimplificationError("invalid_removable_ids")
    baseline = _evaluate(evaluate, model)
    records = []
    for identifier in ids:
        try:
            reduced = remove(model, identifier)
            if not isinstance(reduced, Mapping):
                raise SimplificationError("reduced_model_must_be_object")
            result = _evaluate(evaluate, reduced)
            objective_delta = result["objective_loss"] - baseline["objective_loss"]
            stress_delta = None
            if baseline["stress_objective_loss"] is not None and result["stress_objective_loss"] is not None:
                stress_delta = result["stress_objective_loss"] - baseline["stress_objective_loss"]
            feasible = result["feasible"]
            counterexample = (not feasible or objective_delta > config.objective_tolerance
                              or (stress_delta is not None and stress_delta > config.stress_tolerance))
            records.append({
                "id": identifier,
                "status": "counterexample_found" if counterexample else "tested_not_falsified",
                "feasible": feasible,
                "objective_delta": objective_delta,
                "stress_objective_delta": stress_delta,
                "removal_recommended": not counterexample,
            })
        except SimplificationError as exc:
            records.append({"id": identifier, "status": "not_assessed", "error_code": exc.code,
                            "removal_recommended": False})
        except Exception:
            records.append({"id": identifier, "status": "not_assessed", "error_code": "candidate_failed_safe",
                            "removal_recommended": False})
    return {
        "schema_version": "mathmodel.model-simplification/v1",
        "baseline": baseline,
        "candidates": records,
        "recommended_ids": [item["id"] for item in records if item.get("removal_recommended")],
        "policy": {
            "no_fixed_deletion_ratio": True,
            "stress_domain_is_not_global_proof": True,
            "model_not_mutated": True,
        },
    }


__all__ = ["SimplificationError", "SimplificationConfig", "assess_simplifications"]
