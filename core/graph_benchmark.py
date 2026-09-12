"""Paired development-budget protocol, not a population accuracy benchmark."""
from dataclasses import asdict
import html
import json
from pathlib import Path
import sqlite3
import time

from .artifact_manager import RunArtifactManager, create_run_id
from .confirmation_registry import ConfirmationRegistry, confirm_frozen_model, validate_study_id
from .graph_confirmation import HeldoutCases
from .graph_experiments import SearchExperiment, restore_problem
from .graph_panel import FrozenModelPanel, confirm_frozen_panel
from .graph_search import GraphSearchBudget, GraphSearchSession
from .graph_search_artifacts import confirmation_report, confirmation_status_label
from .model_hypotheses import HypothesisIR, HypothesisValidationError, _canonical, _keys, _require, decode_proposal

BENCHMARK_VERSION = "mathmodel.graph-benchmark/v1"


def validate_benchmark(payload):
    spec = json.loads(_canonical(payload))
    _keys(spec, {"schema_version", "problem", "experiment", "candidates", "budget", "arms"})
    _require(spec["schema_version"] == BENCHMARK_VERSION, "benchmark_version_mismatch")
    contract = restore_problem(spec["problem"])
    experiment = SearchExperiment.from_payload(spec["experiment"], contract)
    _keys(spec["budget"], set(), {"max_candidates", "max_patch_attempts", "max_evaluations",
                                    "per_candidate_evaluations", "wall_seconds"})
    budget = GraphSearchBudget(**spec["budget"])
    _require(type(spec["candidates"]) is list and 1 <= len(spec["candidates"]) <= 16, "benchmark_candidate_budget")
    for candidate in spec["candidates"]:
        HypothesisIR.from_payload(candidate, contract)
    _require(type(spec["arms"]) is list and 2 <= len(spec["arms"]) <= 4, "benchmark_arm_budget")
    names = set()
    for arm in spec["arms"]:
        _keys(arm, {"id", "grammar_search", "reuse_intermediates"}, {"diagnostic_hints"})
        validate_study_id(arm["id"])
        _require(arm["id"] not in names, "duplicate_benchmark_arm")
        names.add(arm["id"])
        _require(type(arm["grammar_search"]) is bool and type(arm["reuse_intermediates"]) is bool,
                 "invalid_benchmark_switch")
        hints = arm.get("diagnostic_hints", [])
        _require(type(hints) is list and len(hints) <= 32, "diagnostic_hint_budget_exceeded")
        for hint in hints:
            _require(isinstance(hint, dict) and hint.get("status") == "proposal_not_executed"
                     and hint.get("hard_constraint") is False and hint.get("may_feed_search") is True
                     and hint.get("requires_current_validation") is True, "unsafe_diagnostic_hint_policy")
            primitives = hint.get("candidate_primitives")
            _require(type(primitives) is list and len(primitives) <= 16
                     and all(type(item) is str and 1 <= len(item) <= 80 for item in primitives),
                     "invalid_diagnostic_hint_primitives")
    spec["budget"] = {k: v for k, v in asdict(budget).items() if k != "reuse_intermediates"}
    return spec, contract, experiment


def _select_development(result):
    """Predeclared policy: simplest Pareto graph, then development error, then hash."""
    if not result["pareto_candidates"]:
        return None
    sizes = {g["hypothesis_hash"]: len(g["nodes"]) for g in result["hypotheses"]}
    latest = {r["hypothesis_hash"]: r for r in result["reports"]}
    return min(result["pareto_candidates"], key=lambda h: (sizes[h], latest[h]["search_rmse"] or 0.0, h))


def benchmark_report(result):
    def cell(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")
    lines = ["# 同预算数学图策略对照", "",
        "同一实验、初始候选、开发资源上限和验证规则；搜索开关按预登记设置变化。",
        "上限相同不表示必须耗尽预算。无可选模型的策略也计入总方法数，不删除失败项。", "",
        "| 方法 | 诊断提示数 | 开发候选数 | 计费评估次数 | 开发耗时（秒） | 最终状态 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for arm in result["arms"]:
        label = {"no_candidate_passed": "无可选模型，未执行留出", "not_run_cancelled": "已取消，未执行留出"}.get(
            arm["confirmation"]["status"], confirmation_status_label(arm["confirmation"]))
        lines.append("| " + " | ".join(cell(v) for v in (arm["id"], arm.get("diagnostic_hint_count", 0), arm["candidate_count"],
            arm["development_budget"]["evaluations_charged"], arm["development_budget"]["elapsed_seconds"],
            label)) + " |")
    lines.extend(["", f"完整留出通过的方法：{result['accepted_arms']} / {len(result['arms'])}。这不是未见题成功率。",
        "最终数据不回传搜索，也不用于评选方法冠军。资源不足不是数学反例。",
        "首次有效结果耗时、峰值内存、统计显著性和总体错误接受率尚未测量；不填成零。",
        "执行顺序固定，未校正系统负载/冷启动影响；开发与留出分别限时，不是整次运行的硬墙钟上限。", ""])
    for arm in result["arms"]:
        lines.extend(["## 方法 " + arm["id"], ""])
        if arm.get("development_failures"):
            lines.extend(["开发执行缺口：" + ", ".join(cell(c) for c in arm["development_failures"]) +
                          "；这些是执行问题，不是模型的数学反例。", ""])
        if arm["confirmation"]["status"] in ("no_candidate_passed", "not_run_cancelled"):
            lines.extend(["此方法未进入留出检验，仍保留在对照分母中。", ""])
        else:
            lines.extend(confirmation_report(arm["confirmation"]))
    return "\n".join(lines)


def run_graph_benchmark(payload, *, holdout_path, output_root, study, registry_path=None, cancel=None):
    validate_study_id(study)
    spec, contract, experiment = validate_benchmark(payload)
    started = time.monotonic()
    destination = Path(output_root).resolve() / create_run_id()
    destination.mkdir(parents=True, exist_ok=False)
    manager = RunArtifactManager(destination)
    arms, models = [], {}
    try:
        manager.write_json("benchmark.input", "evidence", "benchmark_input.json", spec, format_version=BENCHMARK_VERSION)
        manager.write_json("benchmark.protocol", "evidence", "benchmark_protocol.json", {
            "study": study, "selection": "minimum_nodes_then_development_rmse_then_hash",
            "confirmation_wall_seconds_per_arm": 30, "confirmation_evaluations_per_arm": 1000,
            "quota_redistribution": False, "no_candidate_arms_retained": True})
        for index, method in enumerate(spec["arms"]):
            session = GraphSearchSession(contract, experiment, budget=GraphSearchBudget(**spec["budget"],
                                        reuse_intermediates=method["reuse_intermediates"]))
            result = session.run(spec["candidates"], grammar_search=method["grammar_search"], cancel=cancel,
                                 diagnostic_hints=method.get("diagnostic_hints", []))
            selected = _select_development(result)
            if selected is not None:
                models[method["id"]] = session.freeze(selected)
            arms.append({"id": method["id"], "selected_hash": selected,
                "candidate_count": len(result["pareto_candidates"]), "development_budget": result["budget"],
                "development_failures": sorted({r.get("failure_code", "execution_incomplete")
                    for r in result["reports"] if r["status"] == "execution_incomplete"}),
                "termination": result["termination"], "diagnostic_hint_count": len(method.get("diagnostic_hints", [])),
                "confirmation": {
                    "status": "awaiting_confirmation" if selected is not None else "no_candidate_passed"}})
            manager.write_json(f"benchmark.arm-{index}", "evidence", f"development_{index}.json", result,
                               format_version=result["schema_version"])
        # Record ALL methods, including absent models, BEFORE any heldout read.
        manager.write_json("benchmark.selection", "evidence", "frozen_selection.json", {
            "schema_version": "mathmodel.benchmark-selection/v1", "arms": arms,
            "models": {k: v.public() for k, v in models.items()}, "may_feed_search": False})
        confirmation = {}
        if cancel is not None and cancel.is_set():
            confirmation = {k: {"status": "not_run_cancelled"} for k in models}
        elif models:
            panel = FrozenModelPanel.from_models(models) if len(models) >= 2 else None
            if panel is not None:
                manager.write_json("benchmark.panel", "evidence", "frozen_panel.json", panel.public())
            try:
                with Path(holdout_path).open("rb") as source:
                    raw = source.read(256001)
                _require(len(raw) <= 256000, "holdout_file_size_limit")
                data = HeldoutCases.from_payload(decode_proposal(raw.decode("utf-8-sig")), next(iter(models.values())))
                manager.write_json("benchmark.heldout", "evidence", "heldout_input.json", data.public())
                registry = ConfirmationRegistry(registry_path or Path(__file__).resolve().parents[1] /
                                                 "workspace" / "confirmation" / "usage.sqlite3")
                if panel:
                    final = confirm_frozen_panel(panel, data.public(), registry=registry, study=study, cancel=cancel)
                    confirmation = {a["id"]: a["result"] for a in final["arms"]}
                else:
                    name, model = next(iter(models.items()))
                    final = confirm_frozen_model(model, data.public(), registry=registry, study=study, cancel=cancel)
                    confirmation = {name: final}
                manager.write_json("benchmark.confirmation", "evidence", "confirmation.json", final)
            except (HypothesisValidationError, OSError, UnicodeError, sqlite3.Error) as exc:
                code = exc.code if isinstance(exc, HypothesisValidationError) else "confirmation_io_unavailable"
                confirmation = {k: {"status": "input_rejected" if isinstance(exc, HypothesisValidationError)
                                    else "execution_incomplete", "failure_code": code} for k in models}
        for arm in arms:
            arm["confirmation"] = confirmation.get(arm["id"], arm["confirmation"])
        result = {"schema_version": "mathmodel.graph-benchmark-result/v1", "arms": arms,
            "accepted_arms": sum(a["confirmation"]["status"] == "passed_finite_heldout_checks" for a in arms),
            "elapsed_seconds": round(time.monotonic() - started, 6), "first_valid_seconds": None,
            "peak_memory_bytes": None, "model_api_calls": 0, "model_api_cost": 0,
            "winner_selected": False, "may_feed_search": False, "statistical_significance_assessed": False}
        manager.write_json("benchmark.result", "evidence", "benchmark_result.json", result)
        manager.write_text("benchmark.report", "reports", "benchmark.md", benchmark_report(result),
                           media_type="text/markdown; charset=utf-8")
        manager.finalize("incomplete" if cancel is not None and cancel.is_set() else "complete")
        return result, destination
    except BaseException:
        manager.finalize("failed")
        raise
