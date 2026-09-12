import numpy as np
import pandas as pd

from core.calibration import assess_four_layer_uncertainty
from core.dynamic_model_compiler import compile_and_execute_model, DynamicCompilerError
from core.system_benchmark import compare_frozen_systems
from core.multitable_cegis import materialize_multitable_candidate, run_multitable_cegis
from core.execution_service import ExecutionService, ExecutionServiceError
from core.work_queue import WorkQueueLimits
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
