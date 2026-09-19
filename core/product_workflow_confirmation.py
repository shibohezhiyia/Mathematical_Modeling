"""Prospective raw-user workflow cases for the ordinary research entry.

The tasks use arbitrary business response names and prose-only query binding.
They test whether the product entry reaches a validated answer or explicitly
abstains; they are not a representative sample of real competition problems.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .modeling_assistant import MathModelingAssistant


@dataclass(frozen=True)
class ProductWorkflowCase:
    case_id: str
    structure_group: str
    dataset_name: str
    input_column: str
    target_column: str
    problem: str
    records: tuple[Mapping[str, float], ...]
    expected_status: str
    expected_prediction: float | None
    tolerance: float

    def public_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "structure_group": self.structure_group,
            "dataset_name": self.dataset_name, "target": self.target_column,
            "problem": self.problem, "records": [dict(row) for row in self.records],
        }

    def hidden_reference(self) -> dict[str, Any]:
        return {
            "expected_status": self.expected_status,
            "expected_prediction": self.expected_prediction,
            "tolerance": self.tolerance,
        }


def build_product_workflow_confirmation(*, seed: int = 20261014) -> tuple[ProductWorkflowCase, ...]:
    """Build eight deterministic cases after the caller has frozen source."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("product_workflow_seed_invalid")
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.4, 2.4, 96)

    specs: list[tuple[str, str, str, str, Callable[[np.ndarray], np.ndarray], float, str]] = [
        ("affine-load", "workflow-affine", "ambient_temp_c", "net_demand_kw",
         lambda z: 1.7 * z - 0.4, 1.25, "completed"),
        ("quadratic-yield", "workflow-quadratic", "dose_level", "batch_yield_pct",
         lambda z: 0.6 * z * z - 0.3 * z + 2.0, 1.1, "completed"),
        ("sine-sensor", "workflow-sine", "shaft_angle", "sensor_voltage",
         lambda z: 1.2 + 2.1 * np.sin(0.8 * z - 0.2), 1.35, "completed"),
        ("rational-flow", "workflow-rational", "valve_position", "flow_rate_lpm",
         lambda z: (1.0 + 1.5 * z) / (1.0 + 0.2 * z), 1.4, "completed"),
        ("nested-response", "workflow-nested", "control_signal", "measured_output",
         lambda z: 0.3 + 1.7 * np.tanh(np.sin(1.1 * z + 0.2)), 1.2, "completed"),
        ("step-boundary", "workflow-step-outside-grammar", "threshold_input", "switch_output",
         lambda z: np.where(z >= 0.0, 2.0, -1.0), 0.75, "abstain"),
        ("absolute-kink", "workflow-absolute-outside-grammar", "offset_input", "loss_measure",
         lambda z: np.abs(z - 0.3) + 0.2, 1.05, "abstain"),
    ]
    result: list[ProductWorkflowCase] = []
    for case_id, group, input_name, target_name, function, query, expected_status in specs:
        values = function(x)
        order = rng.permutation(len(x))
        records = tuple({input_name: float(x[index]), target_name: float(values[index])}
                        for index in order)
        result.append(ProductWorkflowCase(
            case_id=case_id, structure_group=group, dataset_name=f"raw_{case_id}",
            input_column=input_name, target_column=target_name,
            problem=f"根据原始观测建立关系，并预测 {input_name} = {query} 时的结果。",
            records=records, expected_status=expected_status,
            expected_prediction=(float(function(np.asarray([query]))[0])
                                 if expected_status == "completed" else None),
            tolerance=2e-3,
        ))

    missing_x = np.linspace(-2.0, 2.0, 96)
    missing_order = rng.permutation(len(missing_x))
    result.append(ProductWorkflowCase(
        case_id="missing-query", structure_group="workflow-missing-query",
        dataset_name="raw_missing_query", input_column="driver_value",
        target_column="business_metric",
        problem="根据原始观测建立关系并给出模型；尚未指定需要预测的输入位置。",
        records=tuple({"driver_value": float(missing_x[index]),
                       "business_metric": float(2.0 * missing_x[index] + 1.0)}
                      for index in missing_order),
        expected_status="needs_input", expected_prediction=None, tolerance=0.0,
    ))
    return tuple(result)


def run_product_workflow_case(
    public: Mapping[str, Any], *, portfolio: bool, discontinuity_gate: bool = True,
    routing_policy: str = "current_then_fallback",
    solver_arm_budget: int = 2,
) -> dict[str, Any]:
    """Execute one case through the same assistant used by the web workflow."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mathmodel-product-confirmation-") as output_dir:
        result = MathModelingAssistant(
            output_dir=str(Path(output_dir)), feedback_optimization=False,
            enable_symbolic_portfolio=bool(portfolio),
            enable_symbolic_discontinuity_gate=bool(discontinuity_gate),
            symbolic_routing_policy=routing_policy,
            symbolic_solver_arm_budget=solver_arm_budget,
        ).run(
            str(public["problem"]),
            {str(public["dataset_name"]): pd.DataFrame(public["records"])},
            target=str(public["target"]), run_modeling=False, generate_plots=False,
        )
    modeled = dict(result.specialized_results.get("automatic_modeling") or {})
    return {
        "status": modeled.get("status"), "reason": modeled.get("reason"),
        "result_grade": modeled.get("result_grade"),
        "recommended_action": modeled.get("recommended_action"),
        "predictions": modeled.get("predictions"), "model": modeled.get("model"),
        "entrypoint": modeled.get("entrypoint"),
        "original_response_variable": modeled.get("original_response_variable"),
        "usage": modeled.get("usage"), "execution_supervision": modeled.get("execution_supervision"),
        "routing_evidence": modeled.get("routing_evidence"),
        "duration_seconds": float(time.monotonic() - started),
    }


def score_product_workflow_output(output: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    expected_status = reference.get("expected_status")
    actual_status = output.get("status")
    grade = output.get("result_grade")
    covered = actual_status == "completed"
    abstained = actual_status == "needs_input" and grade == "abstain"
    if expected_status == "completed":
        prediction = output.get("predictions")
        try:
            value = float(prediction[0]) if isinstance(prediction, list) and len(prediction) == 1 else math.nan
        except (TypeError, ValueError):
            value = math.nan
        expected = float(reference["expected_prediction"])
        error = abs(value - expected) if math.isfinite(value) else None
        accepted_correct = bool(covered and error is not None
                                and error <= float(reference["tolerance"]))
        incorrect_prediction_accept = bool(covered and not accepted_correct)
        return {"valid": accepted_correct, "absolute_error": error,
                "covered": covered, "abstained": abstained,
                "accepted_correct": accepted_correct,
                "validated_accepted": covered and grade == "validated_candidate",
                "out_of_scope_accept": False,
                "incorrect_prediction_accept": incorrect_prediction_accept,
                "false_abstain": abstained,
                "unsafe_accept": incorrect_prediction_accept}
    if expected_status == "abstain":
        return {"valid": abstained, "absolute_error": None, "covered": covered,
                "abstained": abstained, "accepted_correct": False,
                "validated_accepted": covered and grade == "validated_candidate",
                "out_of_scope_accept": covered, "incorrect_prediction_accept": False,
                "false_abstain": False, "unsafe_accept": covered}
    needs_input = actual_status == "needs_input"
    return {"valid": needs_input, "absolute_error": None, "covered": covered,
            "abstained": abstained, "accepted_correct": False,
            "validated_accepted": covered and grade == "validated_candidate",
            "out_of_scope_accept": covered, "incorrect_prediction_accept": False,
            "false_abstain": False, "unsafe_accept": covered}


def summarize_product_workflow_scores(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep coverage, accepted-error types, and selective accuracy separate."""
    count = len(rows)
    accepted = sum(bool(row.get("covered")) for row in rows)
    correct = sum(bool(row.get("accepted_correct")) for row in rows)
    return {
        "task_count": count,
        "valid_count": sum(bool(row.get("valid")) for row in rows),
        "accepted_count": accepted,
        "accepted_correct_count": correct,
        "accepted_accuracy": correct / accepted if accepted else None,
        "validated_accepted_count": sum(bool(row.get("validated_accepted")) for row in rows),
        "abstain_count": sum(bool(row.get("abstained")) for row in rows),
        "out_of_scope_accept_count": sum(bool(row.get("out_of_scope_accept")) for row in rows),
        "incorrect_prediction_accept_count": sum(bool(row.get("incorrect_prediction_accept")) for row in rows),
        "false_abstain_count": sum(bool(row.get("false_abstain")) for row in rows),
        "unsafe_accept_count": sum(bool(row.get("unsafe_accept")) for row in rows),
    }


__all__ = ["ProductWorkflowCase", "build_product_workflow_confirmation",
           "run_product_workflow_case", "score_product_workflow_output",
           "summarize_product_workflow_scores"]
