"""Versioned capability catalog for modules exposed by the project.

The catalog is intentionally explicit: a Python file or a unit test is not
evidence that a capability is active in the product.  Each entry names its
user-facing entry point, maturity, execution status and evidence scope.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "mathmodel.module-catalog/v1"
_STATUSES = {"active", "optional", "experimental", "legacy", "deprecated"}


class ModuleCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class ModuleEntry:
    key: str
    module: str
    entrypoint: str
    status: str
    evidence_scope: str
    user_visible: bool = False
    optional_dependency: bool = False
    notes: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key, "module": self.module, "entrypoint": self.entrypoint,
            "status": self.status, "evidence_scope": self.evidence_scope,
            "user_visible": self.user_visible, "optional_dependency": self.optional_dependency,
            "notes": self.notes,
        }


def default_module_catalog() -> dict[str, Any]:
    entries = [
        ModuleEntry("research_assistant", "core.modeling_assistant", "MathModelingAssistant.run", "active", "repository_regression", True),
        ModuleEntry("typed_graph_search", "core.graph_search_artifacts", "run_search_bundle", "active", "bounded_local_experiment", True),
        ModuleEntry("dynamic_model_compiler", "core.dynamic_model_compiler", "compile_and_execute_model", "active", "typed_dispatch_regression", True, notes="统一 JSON 数学契约路由；不执行生成源码"),
        ModuleEntry("execution_entry_audit", "core.execution_entry_audit", "audit_execution_entries", "active", "static_source_inventory", True, notes="静态入口盘点，不等于 OS 隔离证明"),
        ModuleEntry("multitable_cegis", "core.multitable_cegis", "run_multitable_cegis", "experimental", "bounded_grain_key_cegis", True, notes="多表粒度/键/聚合结构的有限反例修复，不等于因果联结"),
        ModuleEntry("typed_primitive_graph_runtime", "core.primitive_graph_runtime", "execute_primitive_graph", "active", "arithmetic_graph_regression", True, notes="仅执行通过类型/量纲检查的有限代数子图"),
        ModuleEntry("candidate_execution_gate", "core.candidate_execution", "execute_structure_candidate", "active", "typed_candidate_regression", True, notes="候选状态保留提议边界，未闭合机制明确拒绝"),
        ModuleEntry("model_family_cegis_adapter", "core.model_family_adapters", "run_model_family_cegis", "active", "adapter_contract_regression", True, notes="统一 compile/evaluate/diagnose/patch/replay 契约；具体模型族仍需独立适配"),
        ModuleEntry("ode_cegis_adapter", "core.ode_cegis", "run_ode_cegis", "experimental", "bounded_ode_trajectory_regression", True, notes="低维自治多项式 RHS；不是通用 ODE/PDE 发现器"),
        ModuleEntry("optimization_cegis_adapter", "core.optimization_cegis", "run_optimization_family_cegis", "experimental", "bounded_lp_milp_qp_regression", True, notes="线性/整数线性/凸二次合同的有限扰动和目标变异；不等于任意优化器"),
        ModuleEntry("external_method_cegis_adapter", "core.external_method_cegis", "run_external_method_cegis", "experimental", "typed_pde_ude_symbolic_regression", True, notes="PDE/UDE/神经 UDE/LLM-SR 的独立阈值 CEGIS；不接受源码且不等于通用求解器"),
        ModuleEntry("hypothesis_preview", "core.hypothesis_controls", "apply_hypothesis_controls", "active", "typed_graph_preview_regression", True, notes="有限控件 what-if 重算；不发布最终数学判决"),
        ModuleEntry("semantic_binding_contract", "core.binding_contract", "build_binding_contract", "active", "binding_contract_regression", True, notes="单位/角色未决时阻断编译"),
        ModuleEntry("external_method_runtime", "core.external_method_runtime", "execute_external_method", "optional", "method_specific_regression", True),
        ModuleEntry("interaction_graph_screen", "core.interaction_graph_screen", "discover_interaction_graph", "optional", "finite_association_screen", True),
        ModuleEntry("gnn_interaction_screen", "core.gnn_interaction_screen", "discover_gnn_interactions", "experimental", "finite_torch_validation_and_rolling_windows", True, optional_dependency=True, notes="轻量消息传递预测筛查与时间滚动稳定性，不是因果发现"),
        ModuleEntry("sindy_discovery", "core.sindy_discovery", "discover_sparse_dynamics", "optional", "synthetic_or_supplied_trajectory", True),
        ModuleEntry("pde_feature_planner", "core.structure_extensions", "build_pde_library_contract", "experimental", "contract_only", True, notes="规划态，不是完整 PDE 求解器"),
        ModuleEntry("pde_discovery_1d", "core.pde_discovery", "discover_1d_pde", "experimental", "regular_grid_holdout", True, notes="仅支持规则一维空间网格的有限差分候选"),
        ModuleEntry("pde_discovery_2d", "core.pde_discovery", "discover_2d_pde", "experimental", "regular_2d_grid_holdout", True, notes="二维规则网格有限差分候选；不支持任意边界/混合导数"),
        ModuleEntry("pde_discovery_3d", "core.pde_discovery", "discover_3d_pde", "experimental", "regular_3d_grid_holdout", True, notes="三维规则网格有限差分候选；不支持混合导数/不规则网格"),
        ModuleEntry("ude_linear_correction", "core.external_method_runtime", "execute_external_method", "experimental", "bounded_linear_correction", True, notes="当前为有界线性修正"),
        ModuleEntry("ude_neural_correction", "core.ude_neural", "fit_neural_ude", "experimental", "torch_temporal_holdout", True, optional_dependency=True, notes="神经残差修正，不是物理定律证明"),
        ModuleEntry("ude_neural_simulation", "core.ude_neural", "simulate_neural_ude_linear", "experimental", "bounded_closed_loop_rk4", True, optional_dependency=True, notes="线性已知 RHS 加逐状态神经残差；不是任意 UDE 求解器"),
        ModuleEntry("ude_joint_trajectory", "core.ude_neural", "fit_joint_neural_ude", "experimental", "joint_matrix_neural_trajectory_holdout", True, optional_dependency=True, notes="联合拟合线性机理矩阵与神经残差；Euler 小规模后端，不是任意 UDE"),
        ModuleEntry("ude_neural_stiff_solver", "core.ude_neural", "simulate_neural_ude_stiff", "experimental", "bounded_bdf_radau_execution", True, optional_dependency=True, notes="BDF/Radau 刚性意识执行；不等于自动刚性诊断或物理证明"),
        ModuleEntry("llm_symbolic_regression", "core.external_method_runtime", "execute_external_method", "experimental", "proposal_only", True, notes="生成源码仍为提议态；仅类型化 JSON 表达式树可进入本地留出拟合"),
        ModuleEntry("causal_dag_gate", "core.causal_dag", "validate_causal_dag", "optional", "role_and_evidence_gate"),
        ModuleEntry("causal_dag_discovery", "core.causal_dag", "discover_linear_causal_dag", "experimental", "order_constrained_bootstrap_screen", True, notes="线性顺序约束结构筛查；不等于干预因果识别"),
        ModuleEntry("latent_state_screen", "core.latent_state_discovery", "discover_delay_latent_states", "experimental", "mathematical_latent_candidate"),
        ModuleEntry("solver_worker", "core.solver_runtime", "SolverProcessRunner.execute", "active", "controlled_worker_limits"),
        ModuleEntry("os_sandbox_attestation", "core.os_sandbox", "require_strict_os_sandbox", "experimental", "external_supervisor_attestation", True, notes="严格模式无外部容器/隔离证明时拒绝回退；不是内置 Docker 管理器"),
        ModuleEntry("blind_statistics", "core.blind_statistics", "assess_blind_accuracy", "optional", "external_unseen_scores_only", notes="当前公开仓库没有解封评分"),
        ModuleEntry("independent_holdout_scorer", "core.independent_holdout", "score_regression_files", "active", "public_label_embargo", True, notes="独立评分进程；公开标签仍不等于真实未见题"),
        ModuleEntry("report_generator", "core.report_generator", "generate_html_report", "optional", "web_report_export"),
        ModuleEntry("model_deployer", "core.model_deployer", "generate_deploy_package", "optional", "deployment_package_export"),
        ModuleEntry("synthetic_data", "core.synthetic_data_generator", "generate_synthetic_data", "optional", "synthetic_only", notes="合成数据不能作为独立真值"),
        ModuleEntry("shap_explainability", "core.shap_explainer", "explain_model", "optional", "optional_dependency"),
        ModuleEntry("reinforcement_optimizer", "core.reinforcement_learning", "RLOptimizer", "experimental", "factory_registered"),
        ModuleEntry("time_series_forecaster", "core.time_series_forecaster", "TimeSeriesForecaster", "experimental", "legacy_path_review_required"),
        ModuleEntry("multiscale_features", "core.multiscale_features", "MultiScaleFeatureExtractor", "experimental", "no_main_entry_verified"),
        ModuleEntry("ot_reweighting", "core.ot_reweighting", "OTReweighter", "experimental", "no_main_entry_verified"),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "catalogued",
        "entries": [entry.public() for entry in entries],
        "policy": "module_presence_is_not_product_readiness; evidence_scope_is_required",
    }


def validate_module_catalog(catalog: Mapping[str, Any], *, check_imports: bool = False) -> dict[str, Any]:
    if not isinstance(catalog, Mapping) or catalog.get("schema_version") != SCHEMA_VERSION:
        raise ModuleCatalogError("module_catalog_schema_mismatch")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ModuleCatalogError("module_catalog_entries_required")
    seen: set[str] = set()
    checked = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ModuleCatalogError("module_catalog_entry_must_be_object")
        required = {"key", "module", "entrypoint", "status", "evidence_scope", "user_visible", "optional_dependency", "notes"}
        if set(item) != required:
            raise ModuleCatalogError("module_catalog_entry_fields_mismatch")
        key = item["key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ModuleCatalogError("module_catalog_duplicate_or_invalid_key")
        seen.add(key)
        if item["status"] not in _STATUSES:
            raise ModuleCatalogError("module_catalog_invalid_status")
        if not isinstance(item["module"], str) or not isinstance(item["entrypoint"], str) or not item["evidence_scope"]:
            raise ModuleCatalogError("module_catalog_invalid_entry")
        if type(item["user_visible"]) is not bool or type(item["optional_dependency"]) is not bool:
            raise ModuleCatalogError("module_catalog_boolean_field_required")
        import_status = "not_checked"
        if check_imports:
            try:
                importlib.import_module(item["module"])
                import_status = "importable"
            except Exception as exc:
                import_status = f"unavailable:{type(exc).__name__}"
        checked.append({**dict(item), "import_status": import_status})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "entry_count": len(checked),
        "entries": checked,
        "status_counts": {status: sum(item["status"] == status for item in checked) for status in sorted(_STATUSES)},
        "policy": catalog.get("policy"),
    }


def load_module_catalog(path: str | Path | None = None, *, check_imports: bool = False) -> dict[str, Any]:
    if path is None:
        return validate_module_catalog(default_module_catalog(), check_imports=check_imports)
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModuleCatalogError("module_catalog_read_failed") from exc
    return validate_module_catalog(payload, check_imports=check_imports)


def audit_module_usage(source_root: str | Path) -> dict[str, Any]:
    """Audit catalog entries against source/test references without importing them."""
    root = Path(source_root).resolve()
    if not root.is_dir() or root.anchor == root:
        raise ModuleCatalogError("module_audit_root_invalid")
    excluded_dirs = {".git", "workspace", "third_party", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    python_files = [path for path in root.rglob("*.py") if not any(part in excluded_dirs for part in path.parts)]
    if len(python_files) > 10_000:
        raise ModuleCatalogError("module_audit_file_budget_exceeded")
    entries = default_module_catalog()["entries"]
    audited = []
    for entry in entries:
        module = str(entry["module"])
        module_token = module.replace(".", "/")
        references = 0
        test_references = 0
        web_references = 0
        for path in python_files:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if module in text or module_token in text or str(entry["entrypoint"]) in text:
                references += 1
                if "tests" in path.parts:
                    test_references += 1
                if "web" in path.parts:
                    web_references += 1
        audited.append({**entry, "source_files_referencing": references,
                        "test_files_referencing": test_references,
                        "web_files_referencing": web_references,
                        "usage_status": "referenced" if references else "catalog_only"})
    return {"schema_version": SCHEMA_VERSION, "status": "audited",
            "source_root": str(root), "python_files_scanned": len(python_files),
            "entries": audited,
            "catalog_only_keys": [item["key"] for item in audited if item["usage_status"] == "catalog_only"],
            "policy": "text_reference_audit_not_runtime_coverage_or_product_quality_proof"}


__all__ = ["SCHEMA_VERSION", "ModuleCatalogError", "ModuleEntry", "default_module_catalog", "validate_module_catalog", "load_module_catalog", "audit_module_usage"]
