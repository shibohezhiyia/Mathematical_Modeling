"""Execution gate for completed, typed structure candidates.

Candidate generation remains deliberately open-ended, but execution is not:
only a candidate containing a closed arithmetic primitive graph can cross this
gate.  The original proposal is retained in the returned envelope so callers
can distinguish a runtime result from a language-model suggestion.
"""

from __future__ import annotations

import math
import copy
from typing import Any, Mapping, Sequence

import numpy as np

from .primitive_graph_runtime import PrimitiveGraphRuntimeError, execute_primitive_graph
from .cegis_controller import CEGISConfig, run_cegis


class CandidateExecutionError(ValueError):
    pass


def execute_structure_candidate(
    candidate: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    output_ids: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run one explicitly bound candidate and preserve a strict status."""
    if not isinstance(candidate, Mapping):
        raise CandidateExecutionError("candidate_must_be_mapping")
    graph = candidate.get("primitive_graph")
    if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), list):
        raise CandidateExecutionError("candidate_graph_required")
    if candidate.get("status") == "rejected":
        raise CandidateExecutionError("rejected_candidate_cannot_execute")
    requested = list(output_ids) if output_ids else list(graph.get("output_ids", []))
    try:
        result = execute_primitive_graph(graph["nodes"], bindings, output_ids=requested)
    except PrimitiveGraphRuntimeError as exc:
        return {
            "schema_version": "mathmodel.candidate-execution/v1",
            "candidate_id": candidate.get("id"),
            "status": "rejected",
            "reason": str(exc),
            "proposal_status": candidate.get("status"),
            "policy": "failed_typed_execution_is_not_a_model_result",
        }
    return {
        "schema_version": "mathmodel.candidate-execution/v1",
        "candidate_id": candidate.get("id"),
        "status": "executed",
        "proposal_status": candidate.get("status"),
        "result": result,
        "evidence": {
            "type_checked": True,
            "source_execution": False,
            "bindings_explicit": True,
            "independent_validation": "not_assessed",
        },
        "policy": "candidate_execution_is_not_truth_or_generalization_proof",
    }


def evaluate_structure_candidate(
    candidate: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    output_id: str | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Evaluate a candidate on explicit cases for use by a CEGIS controller.

    The evaluator returns bounded numeric witnesses only.  It never mutates a
    candidate and treats malformed cases or an incomplete graph as
    ``not_assessed`` rather than as a mathematical counterexample.
    """
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 256:
        raise CandidateExecutionError("cases_must_be_nonempty_bounded_sequence")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise CandidateExecutionError("candidate_tolerance_invalid")
    errors: list[float] = []
    violations: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not isinstance(case.get("bindings"), Mapping) or "expected" not in case:
            return {"status": "not_assessed", "failure_code": "case_contract_invalid", "violations": [], "cost_units": index + 1}
        execution = execute_structure_candidate(candidate, case["bindings"], output_ids=[output_id] if output_id else ())
        if execution["status"] != "executed":
            return {"status": "not_assessed", "failure_code": str(execution.get("reason", "candidate_execution_failed"))[:120],
                    "violations": [], "cost_units": index + 1}
        outputs = execution["result"]["outputs"]
        selected_id = output_id or next(iter(outputs), None)
        if selected_id is None or selected_id not in outputs:
            return {"status": "not_assessed", "failure_code": "candidate_output_missing", "violations": [], "cost_units": index + 1}
        try:
            prediction = np.asarray(outputs[selected_id], dtype=float)
            expected = np.asarray(case["expected"], dtype=float)
        except (TypeError, ValueError):
            return {"status": "not_assessed", "failure_code": "case_expected_non_numeric", "violations": [], "cost_units": index + 1}
        if prediction.shape != expected.shape or prediction.size == 0 or not np.isfinite(prediction).all() or not np.isfinite(expected).all():
            return {"status": "not_assessed", "failure_code": "case_shape_or_finite_invalid", "violations": [], "cost_units": index + 1}
        absolute = np.abs(prediction - expected)
        errors.extend(absolute.reshape(-1).tolist())
        worst = float(np.max(absolute))
        if worst > float(tolerance):
            violations.append({"reason": "absolute_error_exceeds_tolerance", "witness_id": str(case.get("id", f"case_{index}"))[:80],
                               "absolute_error": worst})
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    return {"status": "pass" if not violations else "fail", "score": rmse,
            "violations": violations[:16], "cost_units": len(cases),
            "policy": "case_evaluation_for_cegis; finite_witnesses_are_not_a_proof"}


def propose_arithmetic_repairs(candidate: Mapping[str, Any], feedback: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Propose bounded constant-only repairs while preserving graph topology."""
    if not isinstance(candidate, Mapping) or not isinstance(feedback, Mapping):
        return []
    graph = candidate.get("primitive_graph")
    if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), list):
        return []
    proposals: list[dict[str, Any]] = []
    for index, node in enumerate(graph["nodes"]):
        if not isinstance(node, Mapping) or node.get("op") != "constant":
            continue
        value = node.get("attributes", {}).get("value") if isinstance(node.get("attributes"), Mapping) else None
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            continue
        magnitude = abs(float(value))
        # Use a deterministic decimal step near ordinary coefficients so a
        # small repair can actually reach a nearby value; scale only for very
        # large coefficients to keep the search bounded.
        step = 0.1 if magnitude <= 10.0 else min(1e6, 0.1 * magnitude)
        for delta in (-step, step):
            revised = copy.deepcopy(dict(candidate))
            revised_node = revised["primitive_graph"]["nodes"][index]
            revised_node["attributes"]["value"] = float(value) + delta
            revised["id"] = f"{candidate.get('id', 'candidate')}_repair_{index}_{'up' if delta > 0 else 'down'}"
            proposals.append(revised)
            if len(proposals) >= 8:
                return proposals
    return proposals


def run_arithmetic_candidate_cegis(
    initial_candidate: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], *,
    config: CEGISConfig | None = None, output_id: str | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Run a small, deterministic CEGIS loop over constant repairs."""
    if not isinstance(initial_candidate, Mapping):
        raise CandidateExecutionError("candidate_must_be_mapping")
    return run_cegis(
        [dict(initial_candidate)],
        lambda candidate: evaluate_structure_candidate(candidate, cases, output_id=output_id, tolerance=tolerance),
        propose_arithmetic_repairs,
        config=config,
    )


__all__ = ["CandidateExecutionError", "execute_structure_candidate", "evaluate_structure_candidate",
           "propose_arithmetic_repairs", "run_arithmetic_candidate_cegis"]
