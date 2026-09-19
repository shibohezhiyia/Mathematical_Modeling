import numpy as np
import pandas as pd
import pytest

from core.calibration import assess_four_layer_uncertainty
from core.dynamic_model_compiler import compile_and_execute_model, DynamicCompilerError
from core.system_benchmark import compare_frozen_systems
from core.multitable_cegis import materialize_multitable_candidate, run_multitable_cegis
from core.execution_service import ExecutionService, ExecutionServiceError
from core.work_queue import WorkQueueLimits
from core.causal_dag import discover_temporal_causal_graph
import time


def test_four_layer_uncertainty_is_explicit_and_bounded():
    base = np.linspace(0, 1, 20)
    result = assess_four_layer_uncertainty({
        "semantic": base, "structure": base + 0.1,
        "parameter": base - 0.05, "numerical": base + 0.01,
    }, seed=3, bootstrap_replicates=200)
    assert result["status"] == "assessed"
    assert set(result["variance_share"]) == {"semantic", "structure", "parameter", "numerical"}
    assert abs(sum(result["variance_share"].values()) - 1) < 1e-8
    assert result["policy"].startswith("conditional_scenario")


def test_temporal_interaction_screen_recovers_lagged_predictive_edge():
    rng = np.random.default_rng(2)
    n = 140
    source = rng.normal(size=n)
    target = np.zeros(n)
    target[0] = rng.normal()
    for index in range(1, n):
        target[index] = 0.85 * source[index - 1] + 0.1 * rng.normal()
    result = discover_temporal_causal_graph(
        np.column_stack([source, target]), ["source", "target"],
        max_lag=2, edge_threshold=0.2, bootstrap=8, random_state=1,
    )
    assert result["status"] == "hypothesis_found"
    assert any(edge["source"] == "source" and edge["target"] == "target" and edge["lag"] == 1
               and edge["stability"] >= 0.75 for edge in result["edges"])
    assert "not_interventional" in result["policy"]


def test_dynamic_compiler_executes_typed_graph():
    result = compile_and_execute_model("primitive_graph", {
        "nodes": [
            {"id": "x", "op": "variable", "inputs": [], "kind": "quantity", "dimensions": {}, "attributes": {"name": "x"}},
            {"id": "two", "op": "constant", "inputs": [], "kind": "quantity", "dimensions": {}, "attributes": {"value": 2}},
            {"id": "y", "op": "multiply", "inputs": ["x", "two"], "kind": "quantity", "dimensions": {}, "attributes": {}},
        ],
        "bindings": {"x": 3}, "output_ids": ["y"],
    })
    assert result["status"] == "executed"
    assert result["result"]["outputs"]["y"] == 6


def test_dynamic_compiler_rejects_source():
    try:
        compile_and_execute_model("primitive_graph", {"code": "__import__('os')"})
    except DynamicCompilerError as exc:
        assert str(exc) == "dynamic_model_source_or_path_forbidden"
    else:
        raise AssertionError("source must be rejected")


def test_problem_statement_compiler_returns_typed_ir_for_unseen_text():
    result = compile_and_execute_model("problem_statement", {
        "problem": "某系统需要研究资源变化、风险和多个约束，并提出可验证的方案。",
        "candidate_budget": 2,
    })
    assert result["kind"] == "problem_statement"
    assert result["status"] == "needs_input"
    payload = result["result"]
    assert payload["compiled_ir"]["input_policy"]["natural_language_is_executable"] is False
    assert payload["structure_candidates"]["candidate_count"] == 2
    assert payload["structure_candidates"]["candidates"][0]["status"] == "proposed_not_executed"
    assert payload["compiled_ir"]["missing_requirements"]


def test_problem_statement_compiler_can_execute_explicit_typed_contract():
    graph = {
        "nodes": [
            {"id": "x", "op": "variable", "inputs": [], "kind": "quantity", "dimensions": {},
             "attributes": {"name": "x"}},
            {"id": "two", "op": "constant", "inputs": [], "kind": "quantity", "dimensions": {},
             "attributes": {"value": 2}},
            {"id": "y", "op": "multiply", "inputs": ["x", "two"], "kind": "quantity",
             "dimensions": {}, "attributes": {}},
        ],
        "bindings": {"x": 3}, "output_ids": ["y"],
    }
    result = compile_and_execute_model("problem_statement", {
        "problem": "将输入量映射为输出量，并保留可审计证据。",
        "contracts": [{"kind": "primitive_graph", "payload": graph}],
    })
    assert result["status"] == "executed"
    executions = result["result"]["contract_executions"]
    assert executions[0]["result"]["outputs"]["y"] == 6


def test_problem_statement_compiler_induces_model_from_raw_records_without_typed_contract():
    result = compile_and_execute_model("problem_statement", {
        "problem": "根据观测识别关系并预测查询点",
        "attachments": [{"name": "raw", "format": "records", "rows": [
            {"input": -2.0, "response": -5.0}, {"input": -1.0, "response": -3.0},
            {"input": 0.0, "response": -1.0}, {"input": 1.0, "response": 1.0},
            {"input": 2.0, "response": 3.0},
        ]}],
        "query_inputs": [3.0],
    })
    modeled = result["result"]["automatic_modeling"]
    assert result["status"] == "executed"
    assert modeled["model"]["structure"] == "affine"
    assert modeled["predictions"] == pytest.approx([5.0])


def test_problem_statement_compiler_refuses_ambiguous_duplicate_dimension_join():
    result = compile_and_execute_model("problem_statement", {
        "problem": "按类别汇总金额",
        "attachments": [
            {"name": "facts", "format": "records", "rows": [{"entity": "E1", "amount": 2.0}]},
            {"name": "labels", "format": "records", "rows": [
                {"entity": "E1", "category": "a"}, {"entity": "E1", "category": "b"},
            ]},
        ],
    })
    modeled = result["result"]["automatic_modeling"]
    assert modeled["status"] == "needs_input"
    assert modeled["missing"] == ["dimension_deduplication_or_time_rule"]


def test_problem_statement_compiler_can_route_typed_dynamic_competition():
    result = compile_and_execute_model("problem_statement", {
        "problem": "研究一个受约束的衰减过程。",
        "dynamic_contract": {
            "families": [{
                "family": "ode", "comparison_group": "decay",
                "candidates": [{"id": "a", "state_dim": 1,
                                "basis": ["linear"], "coefficients": [[-0.5]]}],
                "cases": [{"times": [0.0, 0.5, 1.0], "initial": [1.0],
                           "observations": [[1.0], [0.7788008], [0.6065307]]}],
            }],
            "cegis_config": {"max_rounds": 1, "max_candidates": 2, "max_repairs": 0},
        },
    })
    assert result["status"] == "executed"
    assert result["result"]["dynamic_competition"]["status"] == "completed"


def test_problem_statement_compiler_executes_verified_ir_override():
    relation = {
        "id": "decay",
        "kind": "ode_system",
        "state_variables": ["x"],
        "rhs": {"x": "-k*x"},
        "initial_values": {"x": 12.0},
        "parameters": {"k": 0.3},
        "time_variable": "t",
        "time_span": [0.0, 4.0],
        "output_points": 21,
        "units": {"x": "1", "k": "1/s", "t": "s"},
    }
    result = compile_and_execute_model("problem_statement", {
        "problem": "研究一个随时间变化的系统。",
        "ir_override": {"relations": [relation]},
    })
    assert result["status"] == "executed"
    assert result["result"]["execution_summary"]["result_count"] == 1


def test_dynamic_competition_runs_ode_families_through_cegis_and_pareto():
    times = [0.0, 0.5, 1.0, 1.5]
    observations = [[1.0], [0.7788008], [0.6065307], [0.4723666]]
    result = compile_and_execute_model("dynamic_competition", {
        "families": [{
            "family": "ode", "comparison_group": "decay",
            "candidates": [
                {"id": "correct", "state_dim": 1, "basis": ["linear"], "coefficients": [[-0.5]]},
                {"id": "wrong", "state_dim": 1, "basis": ["linear"], "coefficients": [[0.5]]},
            ],
            "cases": [{"times": times, "initial": [1.0], "observations": observations}],
        }],
        "cegis_config": {"max_rounds": 4, "max_candidates": 8, "max_repairs": 2},
    })
    assert result["status"] == "completed"
    competition = result["result"]
    assert competition["policy"]["dynamic_execution_before_comparison"] is True
    assert competition["comparisons"]["decay"]["comparison"]["candidate_count"] == 1
    assert competition["candidate_count"] == 2


def test_dynamic_competition_preserves_uncertainty_and_question_context():
    result = compile_and_execute_model("dynamic_competition", {
        "families": [{
            "family": "ode", "comparison_group": "decay",
            "candidates": [{"id": "a", "state_dim": 1,
                            "basis": ["linear"], "coefficients": [[-0.5]]}],
            "cases": [{"times": [0.0, 0.5, 1.0], "initial": [1.0],
                       "observations": [[1.0], [0.7788008], [0.6065307]]}],
        }],
        "uncertainty": {
            "semantic": {"status": "partial", "evidence": ["ambiguous"]},
            "structural": {"status": "assessed", "evidence": ["two candidates"]},
            "parameter": {"status": "not_assessed", "evidence": []},
            "numerical": {"status": "assessed", "evidence": ["solver tolerance"]},
        },
        "assumptions": ["时间单位为秒"],
        "next_questions": [{"id": "q1", "text": "是否存在外部驱动？"}],
        "cegis_config": {"max_rounds": 1, "max_candidates": 2, "max_repairs": 0},
    })
    assert result["status"] == "completed"
    verdict = result["result"]["comparisons"]["decay"]["verdict"]
    assert verdict["uncertainty"]["structural"]["status"] == "assessed"
    assert verdict["next_questions"][0]["id"] == "q1"


def test_dynamic_competition_accepts_typed_external_method_family():
    x = np.linspace(0.0, 1.0, 16).tolist()
    y = (2.0 * np.asarray(x) + 1.0).tolist()
    result = compile_and_execute_model("dynamic_competition", {
        "families": [{
            "family": "llm_sr", "comparison_group": "symbolic",
            "candidates": [{
                "id": "linear", "method": "llm_sr",
                "payload": {
                    "target": y, "features": [[value] for value in x],
                    "expression": {"op": "add",
                                   "left": {"op": "multiply", "left": {"op": "param", "name": "a"},
                                            "right": {"op": "var", "name": "x"}},
                                   "right": {"op": "param", "name": "b"}},
                    "parameter_bounds": {"a": [-5.0, 5.0], "b": [-5.0, 5.0]},
                    "max_nfev": 100,
                },
            }],
            "cases": [{"id": "holdout", "max_metric": 0.1}],
        }],
        "cegis_config": {"max_rounds": 2, "max_candidates": 4, "max_repairs": 0},
    })
    assert result["status"] == "completed"
    assert result["result"]["families"][0]["family"] == "llm_sr"
    assert result["result"]["candidate_count"] == 1


def test_dynamic_competition_accepts_external_method_alias():
    # The alias keeps callers that choose a method at runtime on the same
    # typed route; the candidate still has to declare the concrete method.
    result = compile_and_execute_model("dynamic_competition", {
        "families": [{
            "family": "external_method", "method": "llm_sr", "comparison_group": "symbolic",
            "candidates": [{"id": "proposal", "method": "llm_sr", "payload": {"source": "x"}}],
            "cases": [{"id": "holdout", "max_metric": 0.1}],
        }],
        "cegis_config": {"max_rounds": 1, "max_candidates": 2, "max_repairs": 0},
    })
    assert result["status"] == "completed"
    assert result["result"]["families"][0]["family"] == "external_method"
    assert result["result"]["families"][0]["candidates"][0]["status"] == "not_assessed"


def test_dynamic_competition_routes_gnn_family_through_typed_adapter(monkeypatch):
    def fake_screen(frame, target, **kwargs):
        return {"status": "executed", "validation_rmse": 0.2,
                "rows": len(frame), "edges": [{"source": "a", "target": "b", "stability": 1.0}],
                "split_policy": "time_tail_holdout"}

    monkeypatch.setattr("core.gnn_model_cegis.discover_gnn_interactions", fake_screen)
    rows = [{"a": float(i), "b": float(i % 3), "y": float(i)} for i in range(60)]
    result = compile_and_execute_model("dynamic_competition", {
        "families": [{
            "family": "gnn", "comparison_group": "graph",
            "candidates": [{"id": "gated", "target": "y", "columns": ["a", "b"]}],
            "cases": [{"id": "case", "rows": rows, "target": "y", "max_metric": 1.0}],
        }],
        "cegis_config": {"max_rounds": 1, "max_candidates": 2, "max_repairs": 0},
    })
    assert result["status"] == "completed"
    family = result["result"]["families"][0]
    assert family["family"] == "gnn"
    assert family["candidates"][0]["status"] == "candidate_evaluated"
    assert result["result"]["comparisons"]["graph"]["comparison"]["candidate_count"] == 1


def test_current_system_comparison_is_paired_and_reproducible():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=80)
    report = compare_frozen_systems(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:],
                                    model_keys=["ridge"], n_splits=2)
    assert report["status"] == "completed"
    assert report["current_system_diagnostics"]["trained_model_count"] >= 1
    assert report["relative_rmse_effect_current_minus_baseline"] is not None


def test_multitable_cegis_aggregates_before_join():
    candidate = {"tables": {
        "orders": [{"customer": 1, "amount": 10}, {"customer": 1, "amount": 20}],
        "customers": [{"customer": 1, "segment": "A"}],
    }, "joins": [{"left_table": "orders", "right_table": "customers",
                   "left_key": "customer", "right_key": "customer", "aggregate": "first"}]}
    frame = materialize_multitable_candidate(candidate)
    assert len(frame) == 2
    result = run_multitable_cegis([candidate], [{"expected_rows": 2}])
    assert result["status"] in {"accepted_candidates", "completed", "candidate_set_inadequate"}


def test_execution_service_has_owner_scoped_lifecycle_and_cancel():
    service = ExecutionService(WorkQueueLimits(max_workers=1, max_pending=4, max_memory_mb=256))
    try:
        task = service.submit("owner-a", lambda: {"status": "verified"}, estimated_memory_mb=1)
        for _ in range(100):
            current = service.status("owner-a", task["task_id"])
            if current["status"] == "completed":
                break
            time.sleep(0.005)
        assert current["status"] == "completed"
        assert current["result"]["status"] == "verified"
        try:
            service.status("owner-b", task["task_id"])
        except ExecutionServiceError as exc:
            assert str(exc) == "execution_task_not_found"
        else:
            raise AssertionError("task leaked across owners")
    finally:
        service.shutdown()


def test_execution_service_cancelled_pending_task_does_not_execute():
    service = ExecutionService(WorkQueueLimits(max_workers=1, max_pending=4, max_memory_mb=256))
    started = []
    try:
        blocker = service.submit("owner", lambda: time.sleep(0.1), estimated_memory_mb=1)
        pending = service.submit("owner", lambda: started.append(True), estimated_memory_mb=1)
        cancelled = service.cancel("owner", pending["task_id"])
        assert cancelled["cancellation_requested"] is True
        for _ in range(100):
            current = service.status("owner", pending["task_id"])
            if current["status"] == "cancelled":
                break
            time.sleep(0.005)
        assert current["status"] == "cancelled"
        assert not started
        service.cancel("owner", blocker["task_id"])
    finally:
        service.shutdown()
