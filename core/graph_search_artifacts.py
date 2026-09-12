"""Local experiment bundle entry point with the existing versioned run layout."""
from __future__ import annotations

import html
from pathlib import Path

from .artifact_manager import RunArtifactManager, create_run_id
from .graph_experiments import SearchExperiment, restore_problem
from .graph_search import GraphSearchBudget, GraphSearchSession
from .model_hypotheses import _keys, _require

BUNDLE_VERSION = "mathmodel.graph-search-bundle/v1"
_STATUS = {
    "eligible_for_confirmation": "有限检查通过，待独立确认",
    "rejected_by_replay": "旧反例复检不通过",
    "rejected_by_checks": "当前开发检查不通过",
    "not_assessed": "未获得完整评价",
    "not_executable": "数学图尚不可执行",
    "execution_incomplete": "资源或执行未完成",
}
_FIT_STATUS = {
    "not_required": "无需拟合参数", "local_fit_completed": "局部拟合完成（非全局最优证明）",
    "bounded_linear_fit_completed": "结构证明为仿射；有界线性拟合完成",
    "variable_projection_completed": "部分线性参数已消元；外层非线性搜索完成",
    "fit_budget_exhausted": "拟合预算不足，未判定结构错误",
    "numeric_failure": "拟合出现数值定义域错误", "not_converged": "参数拟合未收敛",
}
_CONFIRMATION_STATUS = {
    "passed_finite_heldout_checks": "留出检验通过（仅限实际样本与声明假设）",
    "failed_heldout_checks": "留出检验未通过",
    "execution_incomplete": "留出计算未完成，不能判定模型错误",
    "input_rejected": "留出输入或重复使用检查未通过",
    "awaiting_candidate_selection": "需要先选定一个开发候选，留出文件尚未读取",
    "no_selectable_candidate": "开发候选未通过，留出文件尚未读取",
}
_CONFIRMATION_FAILURES = {
    "holdout_already_consumed_in_study": "这些输入点已用于本研究，不能再作为新证据。",
    "holdout_development_overlap": "留出样本与训练、选参或开发探针重叠。",
    "holdout_file_unavailable": "无法读取留出文件，请检查路径、权限与编码。",
    "confirmation_registry_unavailable": "使用登记库不可用，本次没有获得新的可审计证据。",
}


def confirmation_status_label(result: dict) -> str:
    label = _CONFIRMATION_STATUS.get(result.get("status"), "留出阶段未执行")
    if result.get("failure_code") in _CONFIRMATION_FAILURES:
        label += "；" + _CONFIRMATION_FAILURES[result["failure_code"]]
    return label


def search_report(result: dict) -> str:
    def cell(value):
        return html.escape("—" if value is None else str(value), quote=True).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")

    lines = ["# 数学图搜索记录", "",
             "本报告记录开发阶段的结构竞争，不是最终数学判决或现实正确性证明。", "",
             f"- 题意契约指纹：`{result['contract_hash']}`",
             f"- 实验指纹：`{result['experiment_hash']}`",
             f"- 停止原因：`{result['termination']}`",
             f"- 数值执行次数：{result['budget']['executions']}；计费评估次数：{result['budget']['evaluations_charged']}",
             f"- 保留的历史反例：{len(result['counterexamples'])}；待独立确认候选：{len(result['pareto_candidates'])}",
             "- 题意及假设正确性：未证明；模型 API 调用：0。",
             "- 留出阶段：" + cell(confirmation_status_label(result.get("confirmation", {}))) + "。", "",
             "## 竞争过程", "", "| 候选指纹 | 状态 | 参数拟合 | 选参 RMSE | 历史反例数 |", "| --- | --- | --- | --- | --- |"]
    for entry in result["reports"]:
        label = _STATUS.get(entry["status"], entry["status"])
        if entry.get("selection_status") == "requires_new_replay":
            label += "（此轮后新增反例，不能直接入选）"
        fit = entry.get("fit", {})
        fit_label = _FIT_STATUS.get(fit.get("status"), "无拟合记录")
        if fit.get("attempts"):
            fit_label += f"；尝试 {len(fit['attempts'])} 个起点"
        lines.append("| " + " | ".join(cell(value) for value in (
            entry["hypothesis_hash"][:16], label, fit_label, entry.get("search_rmse", "—"), entry.get("replay_count", "—"))) + " |")
    lines.extend(["", "## 如何理解", "",
        "参数只使用训练数据拟合。选参样本和边界/内部探针用于反驳候选；它们被反复使用，不能称为独立测试。",
        "性质检查仅在明示假设和声明区间下成立。抽样没找到反例不证明整个连续区间正确，也不证明隐变量、因果或题意。",
        "增加新反例后，之前通过的候选必须重新检查；预算不足时留待复检，不继续保留为可选方案。",
        "当前自动变异只是小型原语语法基线；不存在题目名称分支，也没有承诺通用覆盖率。", "",
        "原始绑定、版本化实验、候选图、修改链、反例和证据分别保存在本次 evidence 目录。它们可能包含用户数据，默认仅本地保存。", ""])
    if result.get("confirmation"):
        lines.extend(confirmation_report(result["confirmation"]))
    return "\n".join(lines)


def confirmation_report(result: dict) -> list[str]:
    def safe(value):
        return html.escape("—" if value is None else str(value), quote=True).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")
    lines = ["## 冻结模型的留出检验", "", confirmation_status_label(result), "",
             "先选定模型并冻结结构、参数和阈值，再读取留出文件；本阶段不拟合参数、不选择其他候选、不回传搜索。", ""]
    if result.get("failure_code"):
        lines.append(f"输入/执行代码：`{safe(result['failure_code'])}`。")
    if "case_count" in result:
        lines.extend([f"样本数：{result['case_count']}；已执行检查：{result['check_count']}；失败检查：{result['failed_check_count']}。", "",
                      "| 输出 | 留出 RMSE | 最大绝对误差 | 训练均值基线 RMSE |", "| --- | --- | --- | --- |"])
        for metric in result.get("metrics", []):
            lines.append("| " + " | ".join(safe(metric.get(key, "—")) for key in
                ("output", "rmse", "max_absolute_error", "training_mean_baseline_rmse")) + " |")
    lines.extend(["", "当前只保证登记范围内的精确输入点不重叠；没有证明样本随机、实体/时间独立或具有代表性。",
        "通过有限样本不证明整个连续域、因果关系、现实语义或不确定性校准正确。失败后若修改模型，必须另取未使用的留出数据。",
        "使用记录仅约束同一 study 和本地登记库；重命名病例、排序或改变目标值不能重置记录。手动删除登记库或另建 study 不会使旧数据重新独立。", ""])
    return lines


def _confirm_from_file(session, result, manager, path, study, candidate_hash, registry_path, cancel):
    import sqlite3
    from .confirmation_registry import ConfirmationRegistry, confirm_frozen_model
    from .graph_confirmation import HeldoutCases
    from .model_hypotheses import EvidenceLedger, HypothesisIR, HypothesisValidationError, decode_proposal
    choices = result["pareto_candidates"]
    if not choices:
        return {"status": "no_selectable_candidate", "may_feed_search": False, "holdout_file_read": False}
    if candidate_hash is None and len(choices) != 1:
        return {"status": "awaiting_candidate_selection", "candidate_options": choices,
                "may_feed_search": False, "holdout_file_read": False}
    try:
        model = session.freeze(candidate_hash or choices[0])
    except HypothesisValidationError as exc:
        return {"status": "input_rejected", "failure_code": exc.code, "may_feed_search": False, "holdout_file_read": False}
    manager.write_json("confirmation.frozen-model", "evidence", "frozen_model.json", model.public(),
                       format_version=model.public()["schema_version"], metadata={"role": "frozen_before_holdout_read"})
    try:
        # The first heldout file read is strictly after private-session selection.
        with Path(path).open("rb") as source:
            raw = source.read(256001)
        _require(len(raw) <= 256000, "holdout_file_size_limit")
        payload = decode_proposal(raw.decode("utf-8-sig"))
        data = HeldoutCases.from_payload(payload, model)
        manager.write_json("confirmation.input", "evidence", "heldout_input.json", data.public(),
                           format_version=data.public()["schema_version"], metadata={"role": "heldout_only_never_search_input"})
        registry = ConfirmationRegistry(registry_path or Path(__file__).resolve().parents[1] /
                                        "workspace" / "confirmation" / "usage.sqlite3")
        confirmation = confirm_frozen_model(model, payload, registry=registry, study=study, cancel=cancel)
        if confirmation["status"] in ("passed_finite_heldout_checks", "failed_heldout_checks"):
            graph = HypothesisIR.from_payload(model.public()["hypothesis"], session.contract)
            ledger = EvidenceLedger().append(graph, method="numerical_test",
                outcome="pass" if confirmation["status"] == "passed_finite_heldout_checks" else "fail", scope={
                    "input_hash": data.digest, "evaluator_version": confirmation["schema_version"],
                    "domain": {"inputs": session.experiment.public()["domain"]}, "seed": None,
                    "frozen_model_hash": model.digest, "phase": "frozen_heldout", "may_feed_search": False,
                    "attempt_id": confirmation["consumption"]["attempt_id"], "semantic_verdict": "not_assessed"})
            manager.write_json("confirmation.evidence", "evidence", "heldout_evidence.json", ledger.public(),
                               format_version=ledger.public()["schema_version"], metadata={"role": "scoped_finite_heldout_evidence"})
        return confirmation
    except HypothesisValidationError as exc:
        return {"status": "input_rejected", "failure_code": exc.code, "may_feed_search": False,
                "frozen_model_hash": model.digest}
    except (OSError, UnicodeError):
        return {"status": "input_rejected", "failure_code": "holdout_file_unavailable", "may_feed_search": False,
                "frozen_model_hash": model.digest}
    except sqlite3.Error:
        return {"status": "execution_incomplete", "failure_code": "confirmation_registry_unavailable",
                "may_feed_search": False, "frozen_model_hash": model.digest}


def run_search_bundle(bundle: dict, *, output_root: Path, cancel=None, confirmation_path: Path | None = None,
                      study: str | None = None, candidate_hash: str | None = None,
                      registry_path: Path | None = None) -> tuple[dict, Path]:
    _keys(bundle, {"schema_version", "problem", "experiment", "candidates", "budget"},
          {"patches", "grammar_search", "diagnostic_hints"})
    _require(bundle["schema_version"] == BUNDLE_VERSION, "bundle_version_mismatch")
    if confirmation_path is not None:
        from .confirmation_registry import validate_study_id
        validate_study_id(study)
    contract = restore_problem(bundle["problem"])
    experiment = SearchExperiment.from_payload(bundle["experiment"], contract)
    _keys(bundle["budget"], set(), {"max_candidates", "max_patch_attempts", "max_evaluations",
                                     "per_candidate_evaluations", "wall_seconds", "reuse_intermediates"})
    budget = GraphSearchBudget(**bundle["budget"])
    session = GraphSearchSession(contract, experiment, budget=budget)
    # Always create a fresh run, never overwrite an old manifest or result.
    destination = Path(output_root).resolve() / create_run_id()
    destination.mkdir(parents=True, exist_ok=False)
    manager = RunArtifactManager(destination)
    manager.write_json("search.input", "evidence", "search_input.json", bundle,
                       format_version=BUNDLE_VERSION, metadata={"role": "local_input_snapshot", "contains_bound_data": True})
    try:
        result = session.run(bundle["candidates"], cancel=cancel,
                             grammar_search=bundle.get("grammar_search", True),
                             supplied_patches=bundle.get("patches"),
                             diagnostic_hints=bundle.get("diagnostic_hints"))
        if confirmation_path is not None:
            result["confirmation"] = _confirm_from_file(session, result, manager, confirmation_path, study,
                                                        candidate_hash, registry_path, cancel)
            manager.write_json("confirmation.result", "evidence", "heldout_confirmation.json", result["confirmation"],
                               format_version="mathmodel.scalar-confirmation/v1", metadata={"role": "readonly_final_stage"})
        manager.write_json("search.result", "evidence", "graph_search.json", result,
                           format_version=result["schema_version"], metadata={"role": "development_search_not_proof"})
        manager.write_json("search.counterexamples", "evidence", "counterexamples.json", {
            "schema_version": "mathmodel.graph-counterexamples/v1", "experiment_hash": experiment.digest,
            "records": result["counterexamples"]}, metadata={"role": "scoped_development_witnesses"})
        manager.write_text("search.report", "reports", "graph_search.md", search_report(result),
                           media_type="text/markdown; charset=utf-8", metadata={"role": "development_summary"})
        manager.finalize("incomplete" if result["termination"] == "cancelled" else "complete")
        return result, destination
    except BaseException:
        manager.finalize("failed")
        raise
