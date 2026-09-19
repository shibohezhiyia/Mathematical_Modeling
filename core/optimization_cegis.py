"""Bounded linear-program CEGIS adapter.

The adapter reuses the validated universal LP contract and HiGHS backend.
Candidates are mathematical contracts, not generated source code.  Cases may
provide bounded perturbations and/or an expected objective value; solver
failures remain ``not_assessed`` rather than being misreported as
counterexamples.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cegis_controller import CEGISConfig
from .model_family_adapters import ModelFamilyAdapter, run_model_family_cegis
from .universal_math_solvers import UniversalRelationValidator, UniversalSolverRegistry


class OptimizationCEGISError(ValueError):
    pass


def compile_linear_program_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise OptimizationCEGISError("lp_candidate_must_be_object")
    payload = dict(candidate)
    payload.setdefault("kind", "linear_program")
    verified = UniversalRelationValidator.verify(payload)
    if verified.get("parse_status") != "machine_verified" or verified.get("validation_errors"):
        raise OptimizationCEGISError("lp_contract_invalid")
    if verified.get("kind") != "linear_program":
        raise OptimizationCEGISError("lp_kind_required")
    verified["id"] = str(candidate.get("id", "lp_candidate"))[:120]
    return verified


def _scenario(compiled: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise OptimizationCEGISError("lp_case_must_be_object")
    override = case.get("overrides", {})
    if override is None:
        override = {}
    if not isinstance(override, Mapping):
        raise OptimizationCEGISError("lp_case_overrides_must_be_object")
    allowed = {"objective_coefficients", "bounds", "A_ub", "b_ub", "A_eq", "b_eq", "direction"}
    if set(override) - allowed:
        raise OptimizationCEGISError("lp_case_override_not_allowed")
    scenario = dict(compiled)
    for key, value in override.items():
        scenario[key] = value
    scenario.pop("parse_status", None)
    scenario.pop("validation_errors", None)
    verified = UniversalRelationValidator.verify(scenario)
    if verified.get("parse_status") != "machine_verified" or verified.get("validation_errors"):
        raise OptimizationCEGISError("lp_case_contract_invalid")
    return verified


def evaluate_linear_program_candidate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-7) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 128:
        raise OptimizationCEGISError("lp_cases_invalid")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise OptimizationCEGISError("lp_tolerance_invalid")
    registry = UniversalSolverRegistry()
    violations, objectives, solver_evidence = [], [], []
    objective_errors: list[float] = []
    max_constraint_violation = 0.0
    for index, raw_case in enumerate(cases):
        try:
            relation = _scenario(compiled, raw_case)
            result = registry.execute("linear_program/v1", relation)
        except Exception as exc:
            return {"status": "not_assessed", "failure_code": "lp_solver_failed", "diagnostic": type(exc).__name__, "violations": [], "cost_units": index + 1}
        objective = float(result.get("objective_value"))
        objectives.append(objective)
        constraint_violation = result.get("maximum_constraint_violation", 0.0)
        try:
            constraint_violation = float(constraint_violation)
        except (TypeError, ValueError):
            return {"status": "not_assessed", "failure_code": "lp_constraint_audit_invalid", "violations": [], "cost_units": index + 1}
        if not math.isfinite(constraint_violation):
            return {"status": "not_assessed", "failure_code": "lp_constraint_audit_nonfinite", "violations": [], "cost_units": index + 1}
        convergence = result.get("convergence", {})
        convergence_status = convergence.get("status") if isinstance(convergence, Mapping) else None
        solver_evidence.append({"case": index, "maximum_constraint_violation": constraint_violation,
                                "convergence_status": convergence_status})
        max_constraint_violation = max(max_constraint_violation, constraint_violation)
        if constraint_violation > float(tolerance):
            violations.append({"reason": "lp_constraint_violation", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80],
                               "maximum_constraint_violation": constraint_violation})
        if convergence_status == "fail":
            violations.append({"reason": "lp_solver_certificate_failed", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80]})
        expected = raw_case.get("expected_objective") if isinstance(raw_case, Mapping) else None
        if expected is not None:
            try:
                expected_value = float(expected)
            except (TypeError, ValueError):
                return {"status": "not_assessed", "failure_code": "lp_expected_objective_invalid", "violations": [], "cost_units": index + 1}
            if not math.isfinite(expected_value) or abs(objective - expected_value) > float(tolerance):
                violations.append({"reason": "lp_objective_deviation", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80], "absolute_error": abs(objective - expected_value)})
            objective_errors.append(abs(objective - expected_value))
    metrics: dict[str, float] = {
        "complexity": float(len(compiled.get("objective_coefficients", []))
                             + len(compiled.get("A_ub", [])) + len(compiled.get("A_eq", []))),
        "constraint_violation": max_constraint_violation,
        "instability": max_constraint_violation,
    }
    if objective_errors:
        metrics["validation_loss"] = float(np.mean(objective_errors))
    return {"status": "pass" if not violations else "fail", "score": float(np.mean(objectives)) if objectives else None,
            "predictions": objectives, "metrics": metrics,
            "violations": violations[:16], "solver_evidence": solver_evidence[:128], "cost_units": len(cases),
            "policy": "validated_linear_program_replay;_not_a_global_model_proof;validation_loss_requires_expected_objective"}


def diagnose_linear_program_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    return {"step": 0.05, "failure_code": str(feedback.get("failure_code", ""))[:100]} if isinstance(feedback, Mapping) else {"step": 0.05}


def patch_linear_program_objective(candidate: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    normalized = compile_linear_program_candidate(candidate)
    step = float(diagnostic.get("step", 0.05)) if isinstance(diagnostic, Mapping) else 0.05
    step = min(max(abs(step), 1e-6), 1e3)
    objective = np.asarray(normalized["objective_coefficients"], dtype=float)
    for index in range(objective.size):
        for delta, label in ((-step, "down"), (step, "up")):
            revised = dict(normalized)
            values = objective.copy()
            values[index] += delta
            revised["objective_coefficients"] = values.tolist()
            revised["id"] = f"{normalized['id']}_r{index}_{label}"
            yield revised


def build_optimization_cegis_adapter(*, tolerance: float = 1e-7) -> ModelFamilyAdapter:
    return ModelFamilyAdapter(
        family="linear_program",
        compile=compile_linear_program_candidate,
        evaluate=lambda compiled, cases: evaluate_linear_program_candidate(compiled, cases, tolerance=tolerance),
        diagnose=diagnose_linear_program_feedback,
        patch=patch_linear_program_objective,
        replay=lambda compiled, cases: evaluate_linear_program_candidate(compiled, cases, tolerance=tolerance),
    )


def run_optimization_cegis(initial_candidates: Iterable[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], *, config: CEGISConfig | None = None, tolerance: float = 1e-7) -> dict[str, Any]:
    return run_model_family_cegis(build_optimization_cegis_adapter(tolerance=tolerance), initial_candidates, cases, config=config)


_OPTIMIZATION_KINDS = {
    "linear_program": "linear_program/v1",
    "mixed_integer_linear_program": "mixed_integer_linear_program/v1",
    "quadratic_program": "quadratic_program/v1",
}


def compile_optimization_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one of the bounded universal optimization contracts."""
    if not isinstance(candidate, Mapping):
        raise OptimizationCEGISError("optimization_candidate_must_be_object")
    payload = dict(candidate)
    kind = str(payload.get("kind", "linear_program"))
    if kind not in _OPTIMIZATION_KINDS:
        raise OptimizationCEGISError("optimization_kind_not_supported_by_cegis")
    verified = UniversalRelationValidator.verify(payload)
    if verified.get("parse_status") != "machine_verified" or verified.get("validation_errors"):
        raise OptimizationCEGISError("optimization_contract_invalid")
    verified["id"] = str(candidate.get("id", f"{kind}_candidate"))[:120]
    return verified


def _optimization_scenario(compiled: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise OptimizationCEGISError("optimization_case_must_be_object")
    override = case.get("overrides", {}) or {}
    if not isinstance(override, Mapping):
        raise OptimizationCEGISError("optimization_case_overrides_must_be_object")
    kind = str(compiled.get("kind"))
    allowed = {"objective_coefficients", "linear_coefficients", "quadratic_matrix", "bounds", "A_ub", "b_ub", "A_eq", "b_eq", "integrality", "direction"}
    if set(override) - allowed:
        raise OptimizationCEGISError("optimization_case_override_not_allowed")
    scenario = dict(compiled)
    scenario.update(dict(override))
    scenario.pop("parse_status", None); scenario.pop("validation_errors", None)
    verified = UniversalRelationValidator.verify(scenario)
    if verified.get("parse_status") != "machine_verified" or verified.get("validation_errors"):
        raise OptimizationCEGISError("optimization_case_contract_invalid")
    if str(verified.get("kind")) != kind:
        raise OptimizationCEGISError("optimization_case_kind_changed")
    return verified


def evaluate_optimization_candidate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-7) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 128:
        raise OptimizationCEGISError("optimization_cases_invalid")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise OptimizationCEGISError("optimization_tolerance_invalid")
    kind = str(compiled.get("kind"))
    executor = _OPTIMIZATION_KINDS.get(kind)
    if executor is None:
        raise OptimizationCEGISError("optimization_kind_not_supported_by_cegis")
    registry = UniversalSolverRegistry(); violations = []; objectives = []; solver_evidence = []
    objective_errors: list[float] = []
    max_constraint_violation = 0.0
    max_integrality_violation = 0.0
    for index, raw_case in enumerate(cases):
        try:
            result = registry.execute(executor, _optimization_scenario(compiled, raw_case))
            objective = float(result.get("objective_value"))
        except Exception as exc:
            return {"status": "not_assessed", "failure_code": "optimization_solver_failed", "diagnostic": type(exc).__name__, "violations": [], "cost_units": index + 1}
        objectives.append(objective)
        constraint_violation = result.get("maximum_constraint_violation", 0.0)
        try:
            constraint_violation = float(constraint_violation)
        except (TypeError, ValueError):
            return {"status": "not_assessed", "failure_code": "optimization_constraint_audit_invalid", "violations": [], "cost_units": index + 1}
        if not math.isfinite(constraint_violation):
            return {"status": "not_assessed", "failure_code": "optimization_constraint_audit_nonfinite", "violations": [], "cost_units": index + 1}
        integrality_violation = result.get("maximum_integrality_violation", 0.0)
        try:
            integrality_violation = float(integrality_violation)
        except (TypeError, ValueError):
            return {"status": "not_assessed", "failure_code": "optimization_integrality_audit_invalid", "violations": [], "cost_units": index + 1}
        if not math.isfinite(integrality_violation):
            return {"status": "not_assessed", "failure_code": "optimization_integrality_audit_nonfinite", "violations": [], "cost_units": index + 1}
        convergence = result.get("convergence", {})
        convergence_status = convergence.get("status") if isinstance(convergence, Mapping) else None
        solver_evidence.append({"case": index, "maximum_constraint_violation": constraint_violation,
                                "maximum_integrality_violation": integrality_violation,
                                "convergence_status": convergence_status})
        max_constraint_violation = max(max_constraint_violation, constraint_violation)
        max_integrality_violation = max(max_integrality_violation, integrality_violation)
        if constraint_violation > float(tolerance):
            violations.append({"reason": "optimization_constraint_violation", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80],
                               "maximum_constraint_violation": constraint_violation})
        if integrality_violation > float(tolerance):
            violations.append({"reason": "optimization_integrality_violation", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80],
                               "maximum_integrality_violation": integrality_violation})
        if convergence_status == "fail":
            violations.append({"reason": "optimization_solver_certificate_failed", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80]})
        expected = raw_case.get("expected_objective") if isinstance(raw_case, Mapping) else None
        if expected is not None:
            try:
                expected_value = float(expected)
            except (TypeError, ValueError):
                return {"status": "not_assessed", "failure_code": "optimization_expected_objective_invalid", "violations": [], "cost_units": index + 1}
            if not math.isfinite(expected_value) or abs(objective - expected_value) > float(tolerance):
                violations.append({"reason": "optimization_objective_deviation", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80], "absolute_error": abs(objective - expected_value)})
            objective_errors.append(abs(objective - expected_value))
    complexity = float(len(compiled.get("objective_coefficients", compiled.get("linear_coefficients", [])))
                       + len(compiled.get("A_ub", [])) + len(compiled.get("A_eq", [])))
    metrics: dict[str, float] = {
        "complexity": complexity,
        "constraint_violation": max(max_constraint_violation, max_integrality_violation),
        # This is a solver-feasibility margin, not a claim about global
        # optimization stability.
        "instability": max(max_constraint_violation, max_integrality_violation),
    }
    if objective_errors:
        metrics["validation_loss"] = float(np.mean(objective_errors))
    return {"status": "pass" if not violations else "fail", "score": float(np.mean(objectives)),
            "predictions": objectives, "metrics": metrics,
            "violations": violations[:16], "solver_evidence": solver_evidence[:128], "cost_units": len(cases),
            "policy": "validated_optimization_contract_replay;_not_a_global_model_proof;validation_loss_requires_expected_objective"}


def patch_optimization_objective(candidate: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    normalized = compile_optimization_candidate(candidate)
    step = min(max(abs(float(diagnostic.get("step", 0.05))) if isinstance(diagnostic, Mapping) else 0.05, 1e-6), 1e3)
    field = "linear_coefficients" if normalized["kind"] == "quadratic_program" else "objective_coefficients"
    objective = np.asarray(normalized[field], dtype=float)
    for index in range(objective.size):
        for delta, label in ((-step, "down"), (step, "up")):
            revised = dict(normalized); values = objective.copy(); values[index] += delta
            revised[field] = values.tolist(); revised["id"] = f"{normalized['id']}_r{index}_{label}"
            yield revised


def build_optimization_family_adapter(kind: str, *, tolerance: float = 1e-7) -> ModelFamilyAdapter:
    if kind not in _OPTIMIZATION_KINDS:
        raise OptimizationCEGISError("optimization_kind_not_supported_by_cegis")
    return ModelFamilyAdapter(
        family=kind, compile=compile_optimization_candidate,
        evaluate=lambda compiled, cases: evaluate_optimization_candidate(compiled, cases, tolerance=tolerance),
        diagnose=diagnose_linear_program_feedback, patch=patch_optimization_objective,
        replay=lambda compiled, cases: evaluate_optimization_candidate(compiled, cases, tolerance=tolerance),
    )


def run_optimization_family_cegis(kind: str, initial_candidates: Iterable[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], *, config: CEGISConfig | None = None, tolerance: float = 1e-7) -> dict[str, Any]:
    return run_model_family_cegis(build_optimization_family_adapter(kind, tolerance=tolerance), initial_candidates, cases, config=config)


__all__ = ["OptimizationCEGISError", "build_optimization_cegis_adapter", "compile_linear_program_candidate", "evaluate_linear_program_candidate", "run_optimization_cegis", "compile_optimization_candidate", "evaluate_optimization_candidate", "build_optimization_family_adapter", "run_optimization_family_cegis"]
