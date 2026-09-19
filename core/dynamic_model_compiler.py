"""Single guarded entry point for dynamically composed mathematical models.

The compiler accepts a typed JSON contract and dispatches to existing local
backends.  It is intentionally a dispatcher/contract checker, not an LLM
code executor: source strings, imports and filesystem paths are rejected.
"""
from __future__ import annotations

from typing import Any, Mapping


# Keep the wire version stable: the new problem_statement route only adds a
# kind and result envelope; existing typed-contract clients remain compatible.
SCHEMA_VERSION = "mathmodel.dynamic-compiler/v1"
_KINDS = {
    "problem_statement", "primitive_graph", "ode_cegis", "optimization_cegis",
    "multitable_cegis", "external_method", "dynamic_competition",
}


class DynamicCompilerError(ValueError):
    pass


def _compile_problem_statement(data: Mapping[str, Any]) -> dict[str, Any]:
    """Compile an arbitrary natural-language statement into executable IR.

    The statement is never evaluated as source.  The deterministic problem
    compiler extracts typed relations and sends only machine-verified relations
    to the existing numerical backends.  Unresolved prose is retained as
    bounded structure candidates and explicit clarification questions, so a
    novel problem has a useful next step instead of silently selecting a
    domain template.
    """
    problem = data.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise DynamicCompilerError("problem_statement_required")
    if len(problem) > 2_000_000:
        raise DynamicCompilerError("problem_statement_too_large")
    from .mechanistic_modeling import MechanisticModelingEngine
    from .problem_solver import analyze_problem
    from .structure_candidates import build_structure_candidates

    analysis = analyze_problem(problem)
    engine = MechanisticModelingEngine()
    ir_override = data.get("ir_override")
    if ir_override is not None and not isinstance(ir_override, Mapping):
        raise DynamicCompilerError("problem_ir_override_invalid")
    images = data.get("images", ())
    if images is None:
        images = ()
    if not isinstance(images, (list, tuple)) or len(images) > 8:
        raise DynamicCompilerError("problem_images_invalid")
    compiled = engine.analyze(problem, ir_override=ir_override, problem_images=images)
    has_observations = bool(data.get("has_observations", False))
    candidate_budget = data.get("candidate_budget", 3)
    if type(candidate_budget) is not int or not 1 <= candidate_budget <= 6:
        raise DynamicCompilerError("problem_candidate_budget_invalid")
    candidates = build_structure_candidates(
        analysis, has_observations=has_observations, max_candidates=candidate_budget,
    )

    # Raw record attachments may enter a bounded deterministic induction path.
    # This path infers a supported family and constructs the executable model;
    # callers do not provide candidate equations, matrices, or join plans.
    automatic_modeling = None
    if data.get("attachments") is not None:
        from .automatic_modeling import AutomaticModelingError, induce_and_solve_modeling_task_isolated
        modeling_payload = {key: data[key] for key in (
            "attachments", "problem", "query_inputs", "query_times", "response_column",
            "time_column", "state_column",
            "minimum_total", "required_total", "integer_decisions", "expression_max_depth",
            "expression_search_strategy", "expression_search_seed",
            "maximum_active_items", "activation_budget",
        ) if key in data}
        try:
            automatic_modeling = induce_and_solve_modeling_task_isolated(modeling_payload)
        except AutomaticModelingError as exc:
            automatic_modeling = {"status": "needs_input", "reason": str(exc),
                                  "policy": "bounded_schema_induction_failed_safe"}

    # Optional typed contracts let callers execute additional relations in the
    # same request.  Each contract is independently gated; one failure cannot
    # turn another verified relation into a fabricated result.
    executions: list[dict[str, Any]] = []
    contracts = data.get("contracts", [])
    if contracts is None:
        contracts = []
    if not isinstance(contracts, list) or len(contracts) > 16:
        raise DynamicCompilerError("problem_contracts_invalid")
    for item in contracts:
        if not isinstance(item, Mapping) or not isinstance(item.get("kind"), str) or not isinstance(item.get("payload"), Mapping):
            raise DynamicCompilerError("problem_contract_invalid")
        contract_kind = str(item["kind"])
        if contract_kind == "problem_statement":
            raise DynamicCompilerError("nested_problem_statement_forbidden")
        executions.append(compile_and_execute_model(contract_kind, item["payload"]))

    # A problem statement may carry a separately authored dynamic contract.
    # This is the bridge from arbitrary prose to executable model competition:
    # prose still only supplies semantic context, while numerical execution is
    # allowed only for the typed contract and remains independently gated.
    dynamic_contract = data.get("dynamic_contract")
    dynamic_execution = None
    if dynamic_contract is not None:
        if not isinstance(dynamic_contract, Mapping):
            raise DynamicCompilerError("problem_dynamic_contract_invalid")
        dynamic_execution = compile_and_execute_model("dynamic_competition", dynamic_contract)
        executions.append(dynamic_execution)

    execution = compiled.get("solver_execution", {})
    execution_status = str(compiled.get("execution_status", "needs_model_completion"))
    if execution_status in {"executed", "partially_executed"}:
        status = execution_status
    elif compiled.get("compiler_plan", {}).get("executable"):
        status = "compiled_with_unresolved_relations"
    else:
        status = "needs_input"
    if executions and any(item.get("status") in {"executed", "completed", "accepted", "verified"} for item in executions):
        status = "executed" if status == "needs_input" else status
    if isinstance(automatic_modeling, Mapping) and automatic_modeling.get("status") == "completed":
        status = "executed"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "kind": "problem_statement",
        "result": {
            "problem_analysis": analysis,
            "compiled_ir": compiled,
            "structure_candidates": candidates,
            "contract_executions": executions,
            "dynamic_competition": dynamic_execution,
            "automatic_modeling": automatic_modeling,
            "numerical_results": compiled.get("numerical_results", []),
            "execution_summary": {
                "status": execution_status,
                "result_count": len(execution.get("results", [])) if isinstance(execution, Mapping) else 0,
                "failure_count": len(execution.get("failures", [])) if isinstance(execution, Mapping) else 0,
            },
        },
        "policy": "typed_problem_compilation;bounded_raw_attachment_induction;prose_is_never_executed;unresolved_relations_require_input",
    }


def compile_and_execute_model(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise DynamicCompilerError("dynamic_model_kind_invalid")
    if not isinstance(payload, Mapping):
        raise DynamicCompilerError("dynamic_model_payload_invalid")
    data = dict(payload)
    # Never accept generated source or path-like execution controls.
    forbidden = {"source", "code", "python", "command", "path", "module", "import"}
    if forbidden.intersection(data):
        raise DynamicCompilerError("dynamic_model_source_or_path_forbidden")
    if kind == "problem_statement":
        return _compile_problem_statement(data)
    if kind == "primitive_graph":
        from .primitive_graph_runtime import execute_primitive_graph
        result = execute_primitive_graph(data.get("nodes"), data.get("bindings"), output_ids=data.get("output_ids", []))
    elif kind == "ode_cegis":
        from .ode_cegis import run_ode_cegis
        candidate = data.get("candidate")
        candidates = data.get("candidates", [candidate])
        if not isinstance(candidates, list) or not candidates or any(not isinstance(item, Mapping) for item in candidates):
            raise DynamicCompilerError("ode_candidates_invalid")
        result = run_ode_cegis(candidates, data.get("cases"), tolerance=data.get("tolerance", 1e-2))
    elif kind == "optimization_cegis":
        from .optimization_cegis import run_optimization_family_cegis
        candidate = data.get("candidate")
        if not isinstance(candidate, Mapping):
            raise DynamicCompilerError("optimization_candidate_invalid")
        result = run_optimization_family_cegis(str(data.get("optimization_kind", candidate.get("kind", "linear_program"))),
                                               [candidate], data.get("cases"), tolerance=data.get("tolerance", 1e-7))
    elif kind == "multitable_cegis":
        from .multitable_cegis import run_multitable_cegis
        candidate = data.get("candidate")
        candidates = data.get("candidates", [candidate])
        if not isinstance(candidates, list) or not candidates or any(not isinstance(item, Mapping) for item in candidates):
            raise DynamicCompilerError("multitable_candidates_invalid")
        result = run_multitable_cegis(candidates, data.get("cases"))
    elif kind == "dynamic_competition":
        from .cegis_controller import CEGISConfig
        from .model_competition import compete_dynamic_families
        from .model_family_adapters import ModelFamilyAdapter
        from .ode_cegis import build_ode_cegis_adapter
        from .optimization_cegis import build_optimization_family_adapter
        from .multitable_cegis import (
            compile_multitable_candidate, evaluate_multitable_candidate,
        )
        from .external_method_cegis import build_external_method_cegis_adapter

        raw_families = data.get("families")
        if not isinstance(raw_families, list) or not 1 <= len(raw_families) <= 8:
            raise DynamicCompilerError("dynamic_competition_families_invalid")
        family_specs = []
        for item in raw_families:
            if not isinstance(item, Mapping):
                raise DynamicCompilerError("dynamic_competition_family_invalid")
            family = str(item.get("family", "")).strip()
            candidates = item.get("candidates")
            cases = item.get("cases")
            if not family or not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, Mapping) for candidate in candidates):
                raise DynamicCompilerError("dynamic_competition_candidates_invalid")
            if not isinstance(cases, list) or any(not isinstance(case, Mapping) for case in cases):
                raise DynamicCompilerError("dynamic_competition_cases_invalid")
            if family == "ode":
                adapter = build_ode_cegis_adapter(tolerance=float(item.get("tolerance", 1e-2)))
            elif family in {"linear_program", "mixed_integer_linear_program", "quadratic_program"}:
                adapter = build_optimization_family_adapter(family, tolerance=float(item.get("tolerance", 1e-7)))
            elif family == "multitable":
                def compile_multitable(candidate):
                    return compile_multitable_candidate(candidate)
                adapter = ModelFamilyAdapter(
                    family="multitable", compile=compile_multitable,
                    evaluate=evaluate_multitable_candidate,
                    diagnose=lambda feedback: {"violations": feedback.get("violations", [])} if isinstance(feedback, Mapping) else {},
                    patch=lambda *_: (),
                )
            elif family in {"pde_find", "ude", "ude_neural", "ude_joint", "ude_stiff", "llm_sr"}:
                adapter = build_external_method_cegis_adapter(family)
            elif family == "gnn":
                from .gnn_model_cegis import build_gnn_cegis_adapter
                adapter = build_gnn_cegis_adapter()
            elif family == "external_method":
                method = str(item.get("method", "")).strip()
                if method not in {"pde_find", "ude", "ude_neural", "ude_joint", "ude_stiff", "llm_sr"}:
                    raise DynamicCompilerError("dynamic_competition_external_method_invalid")
                adapter = build_external_method_cegis_adapter(method)
            else:
                raise DynamicCompilerError("dynamic_competition_family_not_supported")
            family_specs.append({
                "family": family, "adapter": adapter,
                "initial_candidates": candidates, "cases": cases,
                "comparison_group": str(item.get("comparison_group", family)),
            })
        config_payload = data.get("cegis_config", {}) or {}
        if not isinstance(config_payload, Mapping):
            raise DynamicCompilerError("dynamic_competition_config_invalid")
        allowed_config = {"max_rounds", "max_candidates", "max_repairs", "max_counterexamples", "max_wall_seconds", "max_cost_units"}
        if set(config_payload) - allowed_config:
            raise DynamicCompilerError("dynamic_competition_config_field_invalid")
        try:
            config = CEGISConfig(**dict(config_payload))
        except Exception as exc:
            raise DynamicCompilerError("dynamic_competition_config_invalid") from exc
        uncertainty = data.get("uncertainty")
        if uncertainty is not None and not isinstance(uncertainty, Mapping):
            raise DynamicCompilerError("dynamic_competition_uncertainty_invalid")
        assumptions = data.get("assumptions", [])
        next_questions = data.get("next_questions", [])
        if not isinstance(assumptions, list) or not isinstance(next_questions, list):
            raise DynamicCompilerError("dynamic_competition_evidence_context_invalid")
        result = compete_dynamic_families(
            family_specs, config=config, uncertainty=uncertainty,
            assumptions=assumptions, next_questions=next_questions,
        )
    else:
        from .external_method_runtime import execute_external_method
        result = execute_external_method(str(data.get("method")), data.get("payload", {}))
    inner_status = result.get("status") if isinstance(result, Mapping) else None
    status = "executed" if inner_status in {None, "executed", "accepted", "accepted_candidates", "verified"} else str(inner_status)
    if inner_status in {"not_assessed", "candidate_set_inadequate", "budget_exhausted", "failed"}:
        status = inner_status
    return {"schema_version": SCHEMA_VERSION, "status": status, "kind": kind,
            "result": result, "policy": "typed_json_dispatch; no_generated_source_execution"}


__all__ = ["SCHEMA_VERSION", "DynamicCompilerError", "compile_and_execute_model"]
