"""Bounded adapter for locally materialized LLM-SRBench Parquet snapshots."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class LLMSRBenchAdapterError(ValueError):
    pass


def select_llm_srbench_instances(
    dataset_dir: str | Path, *, count: int, seed: int,
    exclude: Sequence[str] = (), maximum_arity: int = 4,
) -> tuple[str, ...]:
    """Select cases by public identifiers and arity without reading truth or samples."""
    if type(count) is not int or not 1 <= count <= 64:
        raise LLMSRBenchAdapterError("selection_count_invalid")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise LLMSRBenchAdapterError("seed_invalid")
    if type(maximum_arity) is not int or not 1 <= maximum_arity <= 4:
        raise LLMSRBenchAdapterError("maximum_arity_invalid")
    root = Path(dataset_dir).resolve()
    if not list(root.glob("*.parquet")):
        raise LLMSRBenchAdapterError("parquet_snapshot_missing")
    try:
        import pyarrow.dataset as parquet_dataset
    except ImportError as exc:
        raise LLMSRBenchAdapterError("pyarrow_required") from exc
    metadata = parquet_dataset.dataset(root, format="parquet").to_table(
        columns=["instance_id", "input_vars"],
    ).to_pylist()
    excluded = set(exclude)
    eligible = [(sha256(f"{seed}:{row['instance_id']}".encode()).hexdigest(),
                 str(row["instance_id"])) for row in metadata
                if row["instance_id"] not in excluded
                and 1 <= len(row["input_vars"]) <= maximum_arity]
    selected = tuple(instance_id for _digest, instance_id in sorted(eligible)[:count])
    if len(selected) != count:
        raise LLMSRBenchAdapterError("insufficient_eligible_instances")
    return selected


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_llm_srbench_cases(
    dataset_dir: str | Path, *, instance_ids: Sequence[str], source_revision: str,
    train_limit: int = 512, test_limit: int = 256, seed: int = 20260930,
) -> tuple[dict[str, Any], ...]:
    """Load named public rows without exposing ground truth to the model payload."""
    if (not instance_ids or len(set(instance_ids)) != len(instance_ids)
            or any(type(item) is not str or not item for item in instance_ids)):
        raise LLMSRBenchAdapterError("instance_ids_invalid")
    if type(source_revision) is not str or len(source_revision) not in {40, 64}:
        raise LLMSRBenchAdapterError("source_revision_invalid")
    if type(train_limit) is not int or not 32 <= train_limit <= 4096:
        raise LLMSRBenchAdapterError("train_limit_invalid")
    if type(test_limit) is not int or not 16 <= test_limit <= 2048:
        raise LLMSRBenchAdapterError("test_limit_invalid")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise LLMSRBenchAdapterError("seed_invalid")
    root = Path(dataset_dir).resolve()
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise LLMSRBenchAdapterError("parquet_snapshot_missing")
    try:
        import pyarrow.dataset as parquet_dataset
    except ImportError as exc:
        raise LLMSRBenchAdapterError("pyarrow_required") from exc
    columns = ["instance_id", "description", "train_input", "train_output",
               "test_input", "test_output", "input_vars", "output_vars",
               "gt_expression", "subset"]
    table = parquet_dataset.dataset(root, format="parquet").to_table(
        columns=columns,
        filter=parquet_dataset.field("instance_id").isin(list(instance_ids)),
    )
    indexed = {str(row["instance_id"]): row for row in table.to_pylist()}
    if set(indexed) != set(instance_ids):
        raise LLMSRBenchAdapterError("requested_instances_missing")
    manifest = [{"name": path.name, "bytes": path.stat().st_size,
                 "sha256": _file_digest(path)} for path in files]
    result = []
    for ordinal, instance_id in enumerate(instance_ids):
        row = indexed[instance_id]
        variables = [str(item) for item in row["input_vars"]]
        outputs = [str(item) for item in row["output_vars"]]
        if not 1 <= len(variables) <= 4 or len(outputs) != 1:
            raise LLMSRBenchAdapterError("case_arity_unsupported")
        train_x = np.asarray(row["train_input"], dtype=float)
        train_y = np.asarray(row["train_output"], dtype=float).reshape(-1)
        test_x = np.asarray(row["test_input"], dtype=float)
        test_y = np.asarray(row["test_output"], dtype=float).reshape(-1)
        if (train_x.shape != (len(train_y), len(variables))
                or test_x.shape != (len(test_y), len(variables))
                or not all(np.isfinite(item).all() for item in (train_x, train_y, test_x, test_y))):
            raise LLMSRBenchAdapterError("case_numeric_data_invalid")
        rng = np.random.default_rng(seed + ordinal)
        train_indices = np.sort(rng.choice(len(train_y), min(train_limit, len(train_y)), replace=False))
        test_indices = np.sort(rng.choice(len(test_y), min(test_limit, len(test_y)), replace=False))
        sorted_variables = sorted(variables)
        order = [variables.index(name) for name in sorted_variables]
        observations = [{**{name: float(train_x[index, variables.index(name)])
                             for name in variables}, "response": float(train_y[index])}
                        for index in train_indices]
        public_input = {
            "problem": str(row["description"]),
            "attachments": [{"name": "public_benchmark_observations", "format": "records",
                             "rows": observations}],
            "query_inputs": test_x[test_indices][:, order].tolist(),
        }
        result.append({
            "instance_id": instance_id, "subset": str(row["subset"]),
            "public_input": public_input,
            "hidden_reference": {"test_output": test_y[test_indices].tolist(),
                                 "ground_truth_expression": str(row["gt_expression"])},
            "source": {"repository": "pkuHaowei/llm-srbench",
                       "relationship": "community_mirror_of_llm_srbench",
                       "revision": source_revision, "files": manifest},
        })
    return tuple(result)


def run_llm_srbench_pilot(
    cases: Sequence[Mapping[str, Any]], *,
    solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    status: str = "development_only_community_mirror",
    selection_policy: str = "selected_after_dataset_inspection;not_frozen_confirmation",
    source_policy: str = "mirror_not_official_snapshot",
) -> dict[str, Any]:
    """Run a development-only external-format pilot with official-style numeric metrics."""
    if solver is None:
        from .automatic_modeling import induce_and_solve_modeling_task_isolated
        solver = induce_and_solve_modeling_task_isolated
    report_status = status
    rows = []
    for case in cases:
        from .model_submission_evaluator import reexecute_submitted_model
        reference = np.asarray(case["hidden_reference"]["test_output"], dtype=float)
        output: Mapping[str, Any] = {}
        model_reexecution_status = "not_assessed"
        model_prediction_consistent = False
        model_prediction_max_abs_error = None
        try:
            output = solver(case["public_input"])
            if output.get("status") in {"needs_input", "not_assessed"} and "predictions" not in output:
                raise ValueError(str(output.get("reason") or "prediction_invalid"))
            prediction = np.asarray(output.get("predictions"), dtype=float)
            if prediction.shape != reference.shape or not np.isfinite(prediction).all():
                raise ValueError("prediction_invalid")
            reproduced = reexecute_submitted_model(
                output.get("model"), case["public_input"].get("query_inputs"),
            )
            if reproduced.shape != prediction.shape:
                raise ValueError("model_prediction_shape_mismatch")
            model_prediction_max_abs_error = float(np.max(np.abs(reproduced - prediction), initial=0.0))
            if not np.allclose(reproduced, prediction, rtol=1e-6, atol=1e-9):
                raise ValueError("model_prediction_inconsistent")
            model_reexecution_status = "verified"
            model_prediction_consistent = True
            mse = float(np.mean((prediction - reference) ** 2))
            variance = float(np.var(reference))
            nmse = mse / variance if variance > 0 else (0.0 if mse == 0 else float("inf"))
            relative = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1e-12)
            acc_01 = float(np.mean(relative <= 0.1))
            row_status, reason = "completed", "numeric_metrics_computed"
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            output = output if isinstance(output, Mapping) else {}
            nmse, acc_01 = None, None
            row_status, reason = "failed", str(exc)
        supervision = output.get("execution_supervision") if isinstance(output, Mapping) else None
        usage = output.get("usage") if isinstance(output, Mapping) else None
        submitted_model = output.get("model") if isinstance(output, Mapping) else None
        model_digest = None
        if isinstance(submitted_model, Mapping):
            try:
                model_digest = sha256(json.dumps(
                    submitted_model, ensure_ascii=True, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("ascii")).hexdigest()
            except (TypeError, ValueError):
                model_digest = None
        rows.append({"instance_id": case["instance_id"], "subset": case["subset"],
                     "status": row_status, "reason": reason, "nmse": nmse, "acc_0.1": acc_01,
                     "model_present": isinstance(output.get("model"), Mapping),
                     "model_structure": (submitted_model.get("structure")
                         if isinstance(submitted_model, Mapping) else None),
                     "model_digest": model_digest,
                     "submitted_model": dict(submitted_model)
                         if isinstance(submitted_model, Mapping) else None,
                     "model_reexecution_status": model_reexecution_status,
                     "model_prediction_consistent": model_prediction_consistent,
                     "model_prediction_max_abs_error": model_prediction_max_abs_error,
                     "resource_supervised": bool(isinstance(supervision, Mapping)
                                                 and supervision.get("process_isolated") is True),
                     "execution_elapsed_seconds": (float(supervision["elapsed_seconds"])
                         if isinstance(supervision, Mapping)
                         and type(supervision.get("elapsed_seconds")) in (int, float) else None),
                     "memory_limit_mb": (int(supervision["limits"]["memory_mb"])
                         if isinstance(supervision, Mapping)
                         and isinstance(supervision.get("limits"), Mapping)
                         and type(supervision["limits"].get("memory_mb")) is int else None),
                     "memory_backend": (str(supervision["memory_backend"])
                         if isinstance(supervision, Mapping)
                         and supervision.get("memory_backend") is not None else None),
                     "model_api_calls": (int(usage.get("model_api_calls", 0))
                         if isinstance(usage, Mapping) else None),
                     "manual_interventions": (int(usage.get("manual_interventions", 0))
                         if isinstance(usage, Mapping) else None),
                     "output_status": output.get("status") if isinstance(output, Mapping) else None,
                     "result_grade": output.get("result_grade") if isinstance(output, Mapping) else None,
                     "recommended_action": (output.get("recommended_action")
                         if isinstance(output, Mapping) else None),
                     "ground_truth_expression": case["hidden_reference"]["ground_truth_expression"]})
    numeric = [row for row in rows if row["nmse"] is not None]
    if (type(status) is not str or not status or type(selection_policy) is not str
            or not selection_policy or type(source_policy) is not str or not source_policy):
        raise LLMSRBenchAdapterError("report_policy_invalid")
    return {"schema_version": "mathmodel.llm-srbench-pilot/v1",
            "status": report_status,
            "case_count": len(rows), "numeric_case_count": len(numeric), "rows": rows,
            "summary": {"mean_nmse": (float(np.mean([row["nmse"] for row in numeric]))
                                        if numeric else None),
                        "mean_acc_0.1": (float(np.mean([row["acc_0.1"] for row in numeric]))
                                         if numeric else None),
                        "model_present_count": sum(row["model_present"] for row in rows),
                        "model_reexecution_verified_count": sum(
                            row["model_reexecution_status"] == "verified" for row in rows),
                        "resource_supervised_count": sum(row["resource_supervised"] for row in rows)},
            "policy": (f"ground_truth_withheld_from_solver;submitted_model_independently_reexecuted;"
                       f"{selection_policy};{source_policy}")}


def run_llm_srbench_comparison(
    cases: Sequence[Mapping[str, Any]], *,
    treatment_name: str = "current_multivariate_grammar",
    baseline_name: str = "polynomial_only_ablation",
    treatment_payload: Mapping[str, Any] | None = None,
    baseline_payload: Mapping[str, Any] | None = None,
    schema_version: str = "mathmodel.llm-srbench-comparison/v1",
    report_status: str = "frozen_community_mirror_confirmation",
    selection_policy: str = "hash_selected_before_truth_load;frozen_source_code",
    source_policy: str = "mirror_not_official_snapshot;public_pretraining_exposure_unknown",
    treatment_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    baseline_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare two configured arms on one paired external-source grid."""
    from copy import deepcopy
    from .automatic_modeling import induce_and_solve_modeling_task_isolated
    from .benchmark_statistics import paired_benchmark_effect

    if (not treatment_name or not baseline_name or treatment_name == baseline_name
            or not schema_version):
        raise LLMSRBenchAdapterError("comparison_configuration_invalid")
    arms = (treatment_name, baseline_name)
    additions = {
        treatment_name: dict(treatment_payload or {"enable_multivariate_nonlinear": True}),
        baseline_name: dict(baseline_payload or {"enable_multivariate_nonlinear": False}),
    }
    solvers = {treatment_name: treatment_solver or induce_and_solve_modeling_task_isolated,
               baseline_name: baseline_solver or induce_and_solve_modeling_task_isolated}
    rows, samples = [], []
    for case in cases:
        outcomes = {}
        for arm in arms:
            payload = deepcopy(case["public_input"])
            payload.update(additions[arm])
            report = run_llm_srbench_pilot(
                [{**case, "public_input": payload}], solver=solvers[arm],
                status=report_status, selection_policy=selection_policy,
                source_policy=source_policy,
            )
            row = {**report["rows"][0], "arm": arm}
            row["valid"] = bool(row["nmse"] is not None and row["nmse"] <= 0.01
                                and row["acc_0.1"] is not None and row["acc_0.1"] >= 0.9
                                and row["model_present"]
                                and row["model_reexecution_status"] == "verified"
                                and row["model_prediction_consistent"])
            rows.append(row)
            outcomes[arm] = row
        samples.append({"baseline": float(outcomes[baseline_name]["valid"]),
                        "treatment": float(outcomes[treatment_name]["valid"]),
                        "structure_group": str(case["instance_id"])})
    summary = {arm: {
        "case_count": sum(row["arm"] == arm for row in rows),
        "valid_count": sum(row["valid"] for row in rows if row["arm"] == arm),
        "mean_nmse": float(np.mean([row["nmse"] for row in rows
                                    if row["arm"] == arm and row["nmse"] is not None])),
        "mean_acc_0.1": float(np.mean([row["acc_0.1"] for row in rows
                                       if row["arm"] == arm and row["acc_0.1"] is not None])),
        "resource_supervised_count": sum(row["resource_supervised"] for row in rows
                                         if row["arm"] == arm),
        "model_reexecution_verified_count": sum(
            row["model_reexecution_status"] == "verified" for row in rows if row["arm"] == arm),
        "execution_elapsed_seconds": float(sum(
            row["execution_elapsed_seconds"] or 0.0 for row in rows if row["arm"] == arm)),
        "model_api_calls": sum(
            row["model_api_calls"] or 0 for row in rows if row["arm"] == arm),
        "manual_interventions": sum(
            row["manual_interventions"] or 0 for row in rows if row["arm"] == arm),
        "validated_candidate_count": sum(
            row["result_grade"] == "validated_candidate" for row in rows if row["arm"] == arm),
        "abstain_count": sum(
            row["result_grade"] == "abstain" for row in rows if row["arm"] == arm),
    } for arm in arms}
    return {"schema_version": schema_version,
            "status": report_status, "rows": rows, "summary": summary,
            "paired_effect": paired_benchmark_effect(samples, cluster="structure_group", min_samples=5),
            "primary_success_rule": ("nmse<=0.01;acc_0.1>=0.9;model_present;"
                                     "model_reexecution_verified;model_prediction_consistent"),
            "policy": ("paired_external_source_grid;configured_treatment_and_ablation;"
                       "ground_truth_withheld_from_solver;submitted_model_independently_reexecuted;"
                       f"failures_in_denominator;{source_policy}")}


__all__ = ["LLMSRBenchAdapterError", "select_llm_srbench_instances",
           "load_llm_srbench_cases", "run_llm_srbench_pilot", "run_llm_srbench_comparison"]
