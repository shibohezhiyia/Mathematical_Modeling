"""N05: actual process limits, isolation failures and truthful execution scope."""

import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from core import solver_runtime as runtime
from core.mechanistic_modeling import MechanisticModelingEngine, _SafeNumericExpression
from core.solver_process_limits import ResourceIsolationUnavailable
from core.solver_runtime import (
    EvaluationBudgetExceeded, EvaluationCounter, SolverLimits, SolverProcessRunner,
    SolverRuntimeError, decode_message, encode_message,
)


@pytest.fixture
def ode():
    return {
        "id": "decay", "kind": "ode_system", "state_variables": ["x"],
        "rhs": {"x": "-k*x"}, "initial_values": {"x": 12.0},
        "parameters": {"k": 0.3}, "time_variable": "t", "time_span": [0.0, 4.0],
        "output_points": 61, "units": {"x": "1", "k": "1/s", "t": "s"},
    }


@pytest.fixture
def nlp():
    return {
        "id": "quadratic", "kind": "optimization_problem", "decision_variables": ["x"],
        "objective": "(x-a)**2", "direction": "minimize", "parameters": {"a": 2.0},
        "bounds": {"x": [-5.0, 5.0]}, "initial_values": {"x": 0.0},
        "constraints": [], "units": {"x": "1", "a": "1"}, "multistart_trials": 4,
    }


@pytest.fixture
def probe(monkeypatch):
    processes, directories = [], []
    real_popen = subprocess.Popen

    def track(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        directories.append(Path(kwargs["cwd"]))
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", track)

    def select(mode):
        path = Path(__file__).parent / "fixtures" / "solver_worker_probe.py"
        monkeypatch.setattr(runtime, "_worker_command", lambda: [
            sys.executable, "-I", "-B", "-u", str(path), mode,
        ])
        return processes

    yield select
    assert all(process.poll() is not None for process in processes), "worker survived its request"
    assert all(not directory.exists() for directory in directories), "temporary run was not cleaned"


@pytest.mark.parametrize("changes", [
    {"wall_seconds": float("nan")}, {"wall_seconds": True}, {"wall_seconds": 0},
    {"wall_seconds": 121}, {"memory_mb": 0}, {"memory_mb": True},
    {"max_evaluations": 0}, {"max_evaluations": 2.5}, {"output_bytes": 10},
    {"disk_bytes": 1024}, {"disk_bytes": 4 * 1024 * 1024 * 1024 + 1},
])
def test_limits_are_bounded_plain_numbers(changes):
    with pytest.raises(ValueError):
        SolverLimits(**changes)


@pytest.mark.parametrize("expression", [
    "__import__('os')", "x.__class__", "[x]", "'x'", "True", "sqrt(x,x)", "min()",
    "1e309", "9" * 1000, "x" * 4097, "+" * 40 + "x", "+".join(["x"] * 150), "exp(x, y=2)",
])
def test_expression_language_is_small_and_bounded(expression):
    with pytest.raises((ValueError, SyntaxError)):
        _SafeNumericExpression(["x"]).compile(expression)


@pytest.mark.parametrize("expression,values", [
    ("x", {"x": float("nan")}), ("1e300*1e300", {}), ("(-1)**0.5", {}),
    ("10**10000000", {}), ("exp(1e300)", {}), ("1/x", {"x": 0}),
    ("sqrt(-1)", {}),
])
def test_numeric_runtime_rejects_nonfinite_complex_or_singular_values(expression, values):
    node = _SafeNumericExpression(values).compile(expression)
    with pytest.raises((FloatingPointError, OverflowError, ZeroDivisionError)):
        _SafeNumericExpression.evaluate(node, values)


def test_environment_values_cannot_supply_numeric_hooks():
    class Imposter:
        def __float__(self):
            raise AssertionError("untrusted conversion ran")
    with pytest.raises(ValueError, match="plain real"):
        _SafeNumericExpression.evaluate(_SafeNumericExpression(["x"]).compile("x"), {"x": Imposter()})


@pytest.mark.parametrize("payload", [
    {"x": float("inf")}, {"x": 2**5000}, {"x": object()}, {1: "x"}, {"x": (1, 2)},
])
def test_wire_format_cannot_smuggle_objects_or_nonfinite_values(payload):
    with pytest.raises(SolverRuntimeError):
        encode_message(payload, 4096)


@pytest.mark.parametrize("raw", [
    b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'{"x":' + b'[' * 1000 + b'0' + b']' * 1000 + b'}',
])
def test_decoder_fails_before_recursion_or_duplicate_key_confusion(raw):
    with pytest.raises(SolverRuntimeError):
        decode_message(raw, 10_000)


def test_evaluation_counter_never_overshoots():
    counter = EvaluationCounter(2)
    counter.consume()
    counter.consume()
    with pytest.raises(EvaluationBudgetExceeded):
        counter.consume()
    assert counter.used == 2


def test_unit_overflow_is_rejected_before_process_execution(ode, nlp):
    from core.mathematical_reasoning import check_expression_dimensions, parse_unit

    with pytest.raises((ValueError, SyntaxError)):
        _SafeNumericExpression(["x"]).compile("+" * 3500 + "x")
    assert parse_unit("km^10000") is None
    assert parse_unit("km^-10000") is None
    assert parse_unit("m/" + "m" * 300) is None
    assert check_expression_dimensions("x**10000", {"x": "km"})["status"] == "fail"
    ode["units"]["x"] = "km^10000"
    assert MechanisticModelingEngine._verify_structured_relation(ode)["parse_status"] != "machine_verified"
    nlp["objective"] = "x**10000"
    nlp["units"]["x"] = "km"
    assert MechanisticModelingEngine._verify_structured_relation(nlp)["parse_status"] != "machine_verified"


def test_process_matches_direct_trusted_ode_and_reports_scope(ode):
    direct = MechanisticModelingEngine._solve_ode_system(ode)
    result = SolverProcessRunner().execute("adaptive_ode/v1", ode)
    assert result["summary"]["x"]["final"] == pytest.approx(direct["summary"]["x"]["final"])
    assert result["summary"]["x"]["final"] == pytest.approx(12 * math.exp(-1.2), rel=1e-7)
    assert result["evaluation_usage"]["used"] == direct["evaluation_usage"]["used"]
    assert result["execution_supervision"]["process_isolated"] is True
    assert result["execution_supervision"]["permission_isolated"] is True
    assert result["execution_supervision"]["arbitrary_code_allowed"] is False


def test_process_runner_can_use_opt_in_cross_process_quota(ode, tmp_path):
    from core.shared_resource_quota import SharedResourceQuota, SharedQuotaLimits

    quota = SharedResourceQuota(
        tmp_path / "shared.sqlite3",
        limits=SharedQuotaLimits(max_memory_mb=1024, max_slots=1),
    )
    result = SolverProcessRunner().execute(
        "adaptive_ode/v1", ode,
        limits=SolverLimits(memory_mb=512, wall_seconds=30),
        shared_quota=quota,
    )
    assert result["execution_supervision"]["shared_quota"] == "enabled"
    assert result["execution_supervision"]["shared_quota_lease"] == "acquired"
    assert quota.snapshot()["active_leases"] == 0


def test_process_worker_exposes_only_registered_primitive_dynamics_routes():
    result = SolverProcessRunner().execute(
        "linear_ode/v1",
        {"matrix": [[-1.0]], "initial": [1.0], "times": [0.0, 1.0]},
    )
    assert result["status"] == "executed"
    assert result["states"][-1][0] == pytest.approx(math.exp(-1.0), rel=1e-6)
    event = SolverProcessRunner().execute(
        "threshold_event/v1",
        {"times": [0.0, 1.0], "values": [0.0, 2.0], "threshold": 1.0},
    )
    assert event["event_time"] == pytest.approx(0.5)


def test_process_worker_exposes_registered_quadratic_program_route():
    result = SolverProcessRunner().execute(
        "quadratic_program/v1",
        {
            "kind": "quadratic_program",
            "id": "qp_worker",
            "variables": ["x"],
            "linear_coefficients": [-4.0],
            "quadratic_matrix": [[2.0]],
            "direction": "minimize",
            "bounds": [[0.0, 10.0]],
            "A_ub": [[-1.0]],
            "b_ub": [0.0],
            "A_eq": [],
            "b_eq": [],
            "units": {"x": "1"},
        },
    )
    assert result["status"] == "executed"
    assert result["solution"]["x"] == pytest.approx(2.0, abs=1e-6)
    assert result["execution_supervision"]["process_isolated"] is True


def test_process_worker_exposes_bounded_geometry_and_statistics_routes():
    visibility = SolverProcessRunner().execute(
        "line_of_sight/v1",
        {"source": [0, 0], "target": [3, 0],
         "obstacles": [[[1, -1], [2, -1], [2, 1], [1, 1]]]},
    )
    assert visibility["visible"] is False
    bootstrap = SolverProcessRunner().execute(
        "bootstrap_mean/v1", {"values": [1, 2, 3, 4], "resamples": 100, "seed": 5},
    )
    assert bootstrap["seed"] == 5
    assert bootstrap["execution_supervision"]["process_isolated"] is True


def test_process_matches_direct_multistart_nlp(nlp):
    direct = MechanisticModelingEngine._solve_optimization_problem(nlp)
    result = SolverProcessRunner().execute("bounded_nlp/v1", nlp)
    assert result["solution"] == pytest.approx(direct["solution"])
    assert result["solution"]["x"] == pytest.approx(2.0, abs=1e-6)
    assert result["credibility_audit"]["status"] == "warning"  # Not a global-optimality proof.


@pytest.mark.parametrize("key,fixture", [("adaptive_ode/v1", "ode"), ("bounded_nlp/v1", "nlp")])
def test_numerical_budget_applies_inside_actual_worker(key, fixture, request):
    with pytest.raises(SolverRuntimeError) as caught:
        SolverProcessRunner().execute(key, request.getfixturevalue(fixture), limits=SolverLimits(max_evaluations=1))
    assert caught.value.code == "evaluation_limit"


def test_worker_reverifies_trust_flags_and_executor_kind(ode):
    bad = copy.deepcopy(ode)
    bad.update(parse_status="machine_verified", kind="optimization_problem")
    with pytest.raises(SolverRuntimeError) as caught:
        SolverProcessRunner().execute("adaptive_ode/v1", bad)
    assert caught.value.code == "invalid_contract"
    bad["kind"] = "ode_system"
    bad["rhs"]["x"] = "__import__('os')"
    with pytest.raises(SolverRuntimeError) as caught:
        SolverProcessRunner().execute("adaptive_ode/v1", bad)
    assert caught.value.code == "invalid_contract"


def test_api_never_accepts_a_code_or_module_executor():
    with pytest.raises(SolverRuntimeError) as caught:
        SolverProcessRunner().execute("python:eval", {"code": "2+2"})
    assert caught.value.code == "unsupported_executor"


def test_input_and_output_wire_caps_have_distinct_failure_codes(probe):
    processes = probe("inspect")
    with pytest.raises(SolverRuntimeError, match="input_limit"):
        SolverProcessRunner().execute("adaptive_ode/v1", {"text": "界" * 200_000})
    assert processes == []
    with pytest.raises(SolverRuntimeError, match="output_limit"):
        encode_message({"text": "界" * 200_000}, runtime.MAX_INPUT_BYTES)


def test_cancellation_during_response_validation_cannot_publish_late_result(probe, monkeypatch):
    probe("inspect")
    cancel = threading.Event()
    original_decode = runtime.decode_message

    def cancel_after_decode(*args):
        result = original_decode(*args)
        cancel.set()
        return result

    monkeypatch.setattr(runtime, "decode_message", cancel_after_decode)
    with pytest.raises(SolverRuntimeError, match="cancelled"):
        SolverProcessRunner().execute("adaptive_ode/v1", {}, cancel=cancel)


def test_environment_is_minimal_and_large_pipe_input_is_not_truncated(probe, monkeypatch):
    probe("inspect")
    monkeypatch.setenv("TEST_SOLVER_API_KEY", "private-test-token")
    monkeypatch.setenv("PYTHONPATH", "untrusted-test-location")
    result = SolverProcessRunner().execute("adaptive_ode/v1", {"text": "x" * 100_000})
    assert result["secret_inherited"] is False
    assert result["pythonpath_inherited"] is False
    assert result["temporary_directory"] is True
    assert result["threads"] == "1"
    assert result["payload_size"] == 100_000


@pytest.mark.parametrize("mode,code", [
    ("hang", "timeout"), ("flood", "output_limit"), ("stderr_flood", "output_limit"),
    ("crash", "worker_exit"), ("invalid", "invalid_response"), ("nonfinite", "invalid_payload"),
    ("disk", "disk_limit"),
])
def test_parent_stops_bad_workers_and_never_returns_partial_results(probe, mode, code):
    probe(mode)
    started = time.monotonic()
    limits = SolverLimits(wall_seconds=2, output_bytes=1024,
                          disk_bytes=1 * 1024 * 1024 if mode == "disk" else 64 * 1024 * 1024)
    with pytest.raises(SolverRuntimeError) as caught:
        SolverProcessRunner().execute("adaptive_ode/v1", {}, limits=limits)
    assert caught.value.code == code
    assert "PRIVATE_TEST_TRACEBACK" not in str(caught.value)
    assert time.monotonic() - started < 6


def test_actual_os_memory_limit_denies_oversized_allocation(probe):
    probe("memory")
    with pytest.raises(SolverRuntimeError) as caught:
        SolverProcessRunner().execute("adaptive_ode/v1", {}, limits=SolverLimits(memory_mb=128, wall_seconds=8))
    assert caught.value.code == "memory_limit"


def test_cancellation_terminates_running_worker(probe):
    probe("hang")
    cancel = threading.Event()
    timer = threading.Timer(0.4, cancel.set)
    timer.start()
    try:
        with pytest.raises(SolverRuntimeError) as caught:
            SolverProcessRunner().execute("adaptive_ode/v1", {}, cancel=cancel)
        assert caught.value.code == "cancelled"
    finally:
        timer.join()


def test_cancelled_or_queue_expired_request_does_not_spawn(probe, monkeypatch):
    processes = probe("hang")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SolverRuntimeError, match="cancelled"):
        SolverProcessRunner().execute("adaptive_ode/v1", {}, cancel=cancel)
    monkeypatch.setattr(runtime, "_SLOTS", threading.BoundedSemaphore(0))
    with pytest.raises(SolverRuntimeError, match="queue_timeout"):
        SolverProcessRunner().execute("adaptive_ode/v1", {}, limits=SolverLimits(wall_seconds=0.1))


def test_aggregate_memory_reservation_times_out_without_spawning(probe, monkeypatch):
    import core.solver_runtime as runtime
    processes = probe("inspect")
    with runtime._MEMORY_CONDITION:
        runtime._MEMORY_RESERVED_MB = runtime._MEMORY_CAP_MB
    try:
        with pytest.raises(SolverRuntimeError, match="queue_timeout"):
            SolverProcessRunner().execute("adaptive_ode/v1", {},
                                         limits=SolverLimits(memory_mb=64, wall_seconds=0.1))
        assert processes == []
    finally:
        with runtime._MEMORY_CONDITION:
            runtime._MEMORY_RESERVED_MB = 0
            runtime._MEMORY_CONDITION.notify_all()


def test_missing_resource_backend_fails_closed(monkeypatch, probe):
    processes = probe("inspect")

    def unavailable():
        raise ResourceIsolationUnavailable("private platform details")

    monkeypatch.setattr(runtime, "resource_backend", unavailable)
    with pytest.raises(SolverRuntimeError, match="isolation_unavailable") as caught:
        SolverProcessRunner().execute("adaptive_ode/v1", {})
    assert "private platform details" not in str(caught.value)
    assert processes == []


def test_worker_failure_preserves_other_nodes_and_explicit_failure_code(ode, nlp):
    nlp["objective"] = "1/x"
    result = MechanisticModelingEngine().analyze("计算轨迹并优化。", ir_override={"relations": [nlp, ode]})
    execution = result["solver_execution"]
    assert execution["status"] == "partially_executed"
    assert execution["results"][0]["relation_id"] == "decay"
    assert execution["failures"][0]["failure_code"] == "numeric_domain"
    plan = result["four_layer_pipeline"]["solver_plan"]
    assert all(node["resource_budget"]["wall_time_enforcement"] == "parent_process_deadline" for node in plan["nodes"])


@pytest.mark.skipif(os.name != "nt", reason="Windows assignment fail-closed test")
def test_job_assignment_failure_kills_child_before_releasing_contract(monkeypatch, probe):
    processes = probe("inspect")

    class FailedJob:
        def __init__(self, memory_bytes):
            pass

        def attach(self, process):
            raise ResourceIsolationUnavailable("cannot attach")

        def close(self):
            pass

    monkeypatch.setattr(runtime, "WindowsJob", FailedJob)
    with pytest.raises(SolverRuntimeError, match="isolation_unavailable"):
        SolverProcessRunner().execute("adaptive_ode/v1", {})
    assert len(processes) == 1


def test_timeout_releases_slot_and_next_contract_can_run(probe, monkeypatch):
    monkeypatch.setattr(runtime, "_SLOTS", threading.BoundedSemaphore(1))
    runner = SolverProcessRunner()
    probe("hang")
    with pytest.raises(SolverRuntimeError, match="timeout"):
        runner.execute("adaptive_ode/v1", {}, limits=SolverLimits(wall_seconds=0.3))
    probe("inspect")
    assert runner.execute("adaptive_ode/v1", {})["temporary_directory"] is True


def test_timeout_blocks_bound_downstream_but_preserves_independent_node(monkeypatch, ode, nlp):
    original_command, original_execute = runtime._worker_command, SolverProcessRunner.execute
    commands = []

    def first_hangs():
        commands.append(True)
        if len(commands) == 1:
            return [sys.executable, "-I", "-B", "-u",
                    str(Path(__file__).parent / "fixtures" / "solver_worker_probe.py"), "hang"]
        return original_command()

    def shorter_first(self, key, contract, *, limits=None, cancel=None):
        if contract["id"] == nlp["id"]:
            limits = SolverLimits(wall_seconds=0.5)
        return original_execute(self, key, contract, limits=limits, cancel=cancel)

    monkeypatch.setattr(runtime, "_worker_command", first_hangs)
    monkeypatch.setattr(SolverProcessRunner, "execute", shorter_first)
    downstream = copy.deepcopy(ode)
    downstream.update(id="dependent", input_bindings=[{
        "source_relation_id": nlp["id"], "source_path": "solution.x", "target_path": "initial_values.x",
    }])
    result = MechanisticModelingEngine().analyze(
        "计算轨迹和优化。", ir_override={"relations": [nlp, ode, downstream]},
    )
    execution = result["solver_execution"]
    assert len(commands) == 2  # No worker receives a made-up downstream value.
    assert execution["results"][0]["relation_id"] == "decay"
    assert {item["failure_code"] for item in execution["failures"]} == {"timeout", "upstream_failed"}
    assert all(item["mathematical_verdict"] == "not_assessed" for item in execution["failures"])


def test_report_explains_failures_without_requiring_raw_json(tmp_path, ode, nlp):
    from core.modeling_assistant import MathModelingAssistant

    nlp["objective"] = "1/x"
    result = MathModelingAssistant(output_dir=str(tmp_path), feedback_optimization=False).run(
        "计算轨迹与优化。", datasets=None, run_modeling=False, generate_plots=False,
        mechanistic_ir={"relations": [ode, nlp]},
    )
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "独立进程强制时限" in report
    assert "出现数值定义域或溢出错误" in report
    assert "运行失败不等于数学反例" in report
