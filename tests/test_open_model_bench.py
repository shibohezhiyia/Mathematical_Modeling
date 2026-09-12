import pytest
import json
from pathlib import Path

from core.open_model_bench import (
    BENCHMARK_SCHEMA, BenchmarkCase, BenchmarkRunner, BenchmarkValidationError, OpenModelBench,
    build_reproducibility_manifest,
)


def case(case_id="dev-1", split="development", family="data", truth=False, fingerprint=None):
    payload = {"id": case_id, "split": split, "family": family,
               "statement": "研究输入变量与目标变量之间的关系。",
               "tags": ["synthetic"], "has_ground_truth": truth}
    if fingerprint:
        payload["data_fingerprints"] = [fingerprint]
    return payload


def test_benchmark_is_deterministic_and_partitions_are_fingerprinted():
    bench = OpenModelBench.create([
        case("z-unseen", "unseen", "optimization"),
        case("a-dev"),
    ], name="small")
    assert bench.public()["schema_version"] == BENCHMARK_SCHEMA
    assert [item.case_id for item in bench.cases()] == ["a-dev", "z-unseen"]
    assert bench.partition_fingerprint("development") != bench.partition_fingerprint("unseen")
    assert OpenModelBench.from_payload(bench.public()).digest == bench.digest


def test_reproducibility_manifest_binds_seed_dependencies_and_benchmark():
    bench = OpenModelBench.create([case("d")])
    manifest = build_reproducibility_manifest(
        bench, seed=42, dependency_versions={"numpy": "2.x", "scipy": "1.x"}, runner_version="runner-1",
    )
    assert manifest["benchmark_digest"] == bench.digest
    assert len(manifest["manifest_digest"]) == 64
    with pytest.raises(BenchmarkValidationError, match="invalid_random_seed"):
        build_reproducibility_manifest(bench, seed=-1, dependency_versions={"numpy": "2"}, runner_version="1")


def test_data_fingerprints_cannot_cross_development_boundary():
    digest = "a" * 64
    with pytest.raises(BenchmarkValidationError, match="data_fingerprint_split_overlap"):
        OpenModelBench.create([case("d", fingerprint=digest), case("u", "unseen", fingerprint=digest)])


def test_result_protocol_requires_failure_attribution_and_keeps_unresolved_separate():
    bench = OpenModelBench.create([case("d"), case("u", "unseen")])
    with pytest.raises(BenchmarkValidationError, match="failed_result_requires_stage"):
        bench.record_result("d", status="fail")
    result = bench.record_result("d", status="fail", failure_stage="numeric",
                                 metrics={"numerically_correct": False, "interval_coverage": 0.4})
    unresolved = bench.record_result("u", status="unresolved", metrics={"interval_coverage": 0.8})
    summary = bench.summarize([result, unresolved])
    assert summary["status_counts"] == {"pass": 0, "fail": 1, "unresolved": 1, "not_run": 0}
    assert summary["failure_stage_counts"]["numeric"] == 1
    assert summary["observed_metrics"]["interval_coverage"] == {"n": 2, "mean": pytest.approx(0.6)}
    assert summary["unrun_count"] == 0
    assert summary["policy"]["accuracy_reported"] is False


def test_truth_requires_data_fingerprint_except_pure_mechanistic():
    with pytest.raises(BenchmarkValidationError, match="ground_truth_requires_data"):
        BenchmarkCase.from_payload(case("d", truth=True))
    pure = BenchmarkCase.from_payload(case("m", family="pure_mechanistic", truth=True))
    assert pure.has_ground_truth is True


def test_minimal_public_manifest_covers_all_task_families():
    path = Path(__file__).resolve().parents[1] / "examples" / "open_model_bench_minimal.json"
    bench = OpenModelBench.from_payload(json.loads(path.read_text(encoding="utf-8")))
    assert {item.family for item in bench.cases()} == {
        "data", "pure_mechanistic", "multi_table", "optimization", "dynamics"
    }
    assert {item.split for item in bench.cases()} == {
        "development", "unseen", "structure_transform", "adversarial"
    }


def test_compare_runs_is_paired_and_cannot_select_on_unseen_cases():
    bench = OpenModelBench.create([case("d"), case("t", "structure_transform"), case("u", "unseen")])
    baseline = [bench.record_result("d", status="pass")]
    search = [bench.record_result("d", status="fail", failure_stage="numeric"),
              bench.record_result("t", status="pass")]
    comparison = bench.compare_runs({"baseline": baseline, "search": search})
    assert comparison["winner_selected"] is False
    assert comparison["missing_cases_by_method"]["baseline"] == ["t"]
    assert comparison["paired_case_status"]["d"] == {"baseline": "pass", "search": "fail"}
    with pytest.raises(BenchmarkValidationError, match="final_split_in_selection"):
        bench.compare_runs({"baseline": baseline, "search": search}, allowed_splits=("unseen",))
    with pytest.raises(BenchmarkValidationError, match="result_outside_selection_split"):
        bench.compare_runs({"baseline": [bench.record_result("u", status="pass")], "search": search})


def test_benchmark_runner_normalizes_success_and_exception_without_leaking_traceback():
    path = Path(__file__).resolve().parents[1] / "examples" / "open_model_bench_minimal.json"
    bench = OpenModelBench.from_payload(json.loads(path.read_text(encoding="utf-8")))
    runner = BenchmarkRunner(bench)
    rows = runner.run(lambda case: {"status": "pass", "metrics": {"contract_correct": True}},
                      splits=("development",), max_cases=2)
    assert len(rows) == 2
    assert all(row["status"] == "pass" for row in rows)
    failing = runner.run(lambda case: (_ for _ in ()).throw(RuntimeError("sensitive details")),
                         splits=("development",), max_cases=2)
    assert all(row["status"] == "fail" and row["failure_stage"] == "proposal" for row in failing)
    assert all(row["runner_error_code"] == "runner_exception" for row in failing)


def test_benchmark_runner_rejects_invalid_runner_contract_as_validation_failure():
    path = Path(__file__).resolve().parents[1] / "examples" / "open_model_bench_minimal.json"
    bench = OpenModelBench.from_payload(json.loads(path.read_text(encoding="utf-8")))
    rows = BenchmarkRunner(bench).run(lambda case: {"status": "pass", "metrics": {"unknown": 1}},
                                      splits=("development",), max_cases=2)
    assert all(row["status"] == "fail" and row["failure_stage"] == "validation" for row in rows)
    assert all(row["runner_error_code"] == "invalid_result_metrics" for row in rows)


def test_benchmark_summary_rejects_tampered_metric_values_and_bad_run_rows():
    bench = OpenModelBench.create([case("d")])
    valid = bench.record_result("d", status="pass")
    tampered = {**valid, "metrics": {"interval_coverage": 2}}
    with pytest.raises(BenchmarkValidationError, match="metric_out_of_range"):
        bench.summarize([tampered])
    with pytest.raises(BenchmarkValidationError, match="result_scope_mismatch"):
        bench.compare_runs({"one": [None], "two": [valid]})


def test_benchmark_creation_bounds_generators_before_materializing_them():
    def too_many():
        for index in range(10_001):
            yield case(f"d-{index}")
    with pytest.raises(BenchmarkValidationError, match="invalid_case_count"):
        OpenModelBench.create(too_many())
