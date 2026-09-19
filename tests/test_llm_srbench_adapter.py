import pyarrow as pa
import pyarrow.parquet as pq

from core.llm_srbench_adapter import (
    load_llm_srbench_cases, run_llm_srbench_comparison, run_llm_srbench_pilot,
    select_llm_srbench_instances,
)
from core.model_submission_evaluator import reexecute_submitted_model


def test_adapter_withholds_truth_and_reorders_queries_to_model_column_order(tmp_path):
    table = pa.table({
        "instance_id": ["case-1"], "description": ["discover y from z and a"],
        "train_input": [[[2.0, 1.0]] * 40], "train_output": [[[3.0]] * 40],
        "test_input": [[[4.0, 3.0]] * 20], "test_output": [[[7.0]] * 20],
        "input_vars": [["z", "a"]], "output_vars": [["y"]],
        "gt_expression": ["z+a"], "subset": ["fixture"],
    })
    pq.write_table(table, tmp_path / "part.parquet")
    case = load_llm_srbench_cases(
        tmp_path, instance_ids=["case-1"], source_revision="a" * 40,
        train_limit=32, test_limit=16,
    )[0]
    assert "gt_expression" not in str(case["public_input"])
    assert case["public_input"]["query_inputs"][0] == [3.0, 4.0]
    assert case["hidden_reference"]["ground_truth_expression"] == "z+a"


def test_pilot_computes_metrics_without_passing_hidden_reference_to_solver():
    seen = []
    case = {"instance_id": "x", "subset": "fixture",
            "public_input": {"query_inputs": [[1.0], [2.0]]},
            "hidden_reference": {"test_output": [2.0, 4.0],
                                 "ground_truth_expression": "2*x"}}
    def solver(payload):
        seen.append(payload)
        return {"model": {"family": "modeling_algebra", "structure": "affine",
                          "input_variables": ["x"], "coefficients": [2.0, 0.0]},
                "predictions": [2.0, 4.0],
                "execution_supervision": {"process_isolated": True}}
    report = run_llm_srbench_pilot([case], solver=solver)
    assert seen == [case["public_input"]]
    assert report["rows"][0]["nmse"] == 0.0
    assert report["rows"][0]["acc_0.1"] == 1.0
    assert report["rows"][0]["model_prediction_consistent"] is True
    assert report["rows"][0]["model_reexecution_status"] == "verified"
    assert report["rows"][0]["model_structure"] == "affine"
    assert len(report["rows"][0]["model_digest"]) == 64
    assert report["rows"][0]["submitted_model"]["coefficients"] == [2.0, 0.0]
    assert "not_frozen_confirmation" in report["policy"]


def test_pilot_rejects_predictions_that_do_not_match_submitted_model():
    case = {"instance_id": "mismatch", "subset": "fixture",
            "public_input": {"query_inputs": [[1.0], [2.0]]},
            "hidden_reference": {"test_output": [2.0, 4.0],
                                 "ground_truth_expression": "2*x"}}
    def solver(_payload):
        return {"model": {"family": "modeling_algebra", "structure": "affine",
                          "input_variables": ["x"], "coefficients": [1.0, 100.0]},
                "predictions": [2.0, 4.0]}
    report = run_llm_srbench_pilot([case], solver=solver)
    row = report["rows"][0]
    assert row["status"] == "failed"
    assert row["reason"] == "model_prediction_inconsistent"
    assert row["nmse"] is None
    assert row["model_prediction_consistent"] is False


def test_hash_selection_uses_only_ids_and_arity(tmp_path):
    table = pa.table({"instance_id": ["a", "b", "c"],
                      "input_vars": [["x"], ["x", "z"], ["a", "b", "c", "d", "e"]],
                      "gt_expression": ["secret-a", "secret-b", "secret-c"]})
    pq.write_table(table, tmp_path / "part.parquet")
    selected = select_llm_srbench_instances(tmp_path, count=2, seed=7, maximum_arity=4)
    assert set(selected) == {"a", "b"}


def test_independent_model_executor_covers_symbolic_and_polynomial_models():
    symbolic = {"structure": "compositional_symbolic", "input_variables": ["x"],
                "operator_signature": ["sin"], "parameters": [1.0, 2.0, 3.0, 0.5]}
    queries = [[0.2], [0.7]]
    expected = [1.0 + 2.0 * __import__("numpy").sin(3.0 * row[0] + 0.5) for row in queries]
    assert reexecute_submitted_model(symbolic, queries).tolist() == expected
    polynomial = {"structure": "bilinear", "input_variables": ["a", "b"],
                  "coefficients": [1.0, 2.0, 3.0, 4.0],
                  "terms": [{"kind": "intercept", "indices": None},
                            {"kind": "linear", "indices": [0]},
                            {"kind": "linear", "indices": [1]},
                            {"kind": "interaction", "indices": [0, 1]}]}
    assert reexecute_submitted_model(polynomial, [[2.0, 5.0]]).tolist() == [60.0]


def test_comparison_can_run_distinct_executable_solvers():
    case = {"instance_id": "solver-routing", "subset": "fixture",
            "public_input": {"query_inputs": [[1.0], [2.0]]},
            "hidden_reference": {"test_output": [2.0, 4.0],
                                 "ground_truth_expression": "2*x"}}
    def solver(slope):
        def run(_payload):
            return {"model": {"structure": "affine", "input_variables": ["x"],
                              "coefficients": [slope, 0.0]},
                    "predictions": [slope, 2.0 * slope]}
        return run
    report = run_llm_srbench_comparison(
        [case], treatment_name="good", baseline_name="weak",
        treatment_solver=solver(2.0), baseline_solver=solver(1.0),
    )
    assert report["summary"]["good"]["valid_count"] == 1
    assert report["summary"]["weak"]["valid_count"] == 0
