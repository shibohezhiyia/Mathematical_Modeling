"""Training-only validation router for complementary symbolic solvers."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np

from .model_submission_evaluator import reexecute_submitted_model


class SymbolicPortfolioError(ValueError):
    pass


def _validation_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    mse = float(np.mean((prediction - reference) ** 2))
    variance = float(np.var(reference))
    nmse = mse / variance if variance > 0 else (0.0 if mse == 0 else float("inf"))
    relative = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1e-12)
    acc = float(np.mean(relative <= 0.1))
    return {"nmse": nmse, "acc_0.1": acc,
            "passes_primary_rule": bool(nmse <= 0.01 and acc >= 0.9)}


def _split_training_validation(payload: Mapping[str, Any], *, seed: int,
                               validation_fraction: float) -> tuple[dict[str, Any], list[list[float]], np.ndarray, int]:
    attachments = payload.get("attachments")
    if not isinstance(attachments, list) or len(attachments) != 1:
        raise SymbolicPortfolioError("portfolio_single_attachment_required")
    rows = attachments[0].get("rows") if isinstance(attachments[0], Mapping) else None
    if not isinstance(rows, list) or len(rows) < 64:
        raise SymbolicPortfolioError("portfolio_observations_invalid")
    columns = set(rows[0])
    if "response" not in columns or any(not isinstance(row, Mapping) or set(row) != columns for row in rows):
        raise SymbolicPortfolioError("portfolio_response_binding_required")
    inputs = sorted(columns - {"response"})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    validation_count = max(16, int(round(len(rows) * float(validation_fraction))))
    validation_indices, training_indices = order[:validation_count], order[validation_count:]
    validation_queries = [[float(rows[index][name]) for name in inputs]
                          for index in validation_indices]
    validation_reference = np.asarray([rows[index]["response"] for index in validation_indices], dtype=float)
    fit_payload = deepcopy(dict(payload))
    fit_payload["attachments"] = [{**dict(attachments[0]),
                                   "rows": [dict(rows[index]) for index in training_indices]}]
    fit_payload["query_inputs"] = validation_queries
    fit_payload["gplearn_seed"] = seed
    return fit_payload, validation_queries, validation_reference, validation_count


def _combined_supervision(outcomes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    supervision_rows = [item.get("execution_supervision") for item in outcomes.values()]
    supervised = [row for row in supervision_rows if isinstance(row, Mapping)]
    return {
        "protocol": "mathmodel.sequential-portfolio/v1",
        "process_isolated": bool(len(supervised) == len(outcomes)
                                 and all(row.get("process_isolated") is True for row in supervised)),
        "permission_isolated": bool(len(supervised) == len(outcomes)
                                    and all(row.get("permission_isolated") is True for row in supervised)),
        "elapsed_seconds": float(sum(float(row.get("elapsed_seconds", 0.0)) for row in supervised)),
        "limits": {"wall_seconds": 30.0 * len(outcomes), "memory_mb": 1024,
                   "execution_order": "sequential"},
        "memory_backend": (supervised[0].get("memory_backend") if supervised else None),
        "child_runs": len(supervised),
    }


def _observed_transition_ambiguity(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Flag observations that cannot distinguish a sharp smooth transition.

    Saturation, quantization, noise, and a true jump can yield the same sparse
    records. The gate therefore reports an identification gap and a local
    resampling interval; it does not claim that the generating structure lies
    outside the declared grammar.
    """
    attachments = payload.get("attachments")
    if not isinstance(attachments, list) or len(attachments) != 1:
        return None
    rows = attachments[0].get("rows") if isinstance(attachments[0], Mapping) else None
    if not isinstance(rows, list) or len(rows) < 64 or not rows or not isinstance(rows[0], Mapping):
        return None
    inputs = sorted(set(rows[0]) - {"response"})
    if len(inputs) != 1:
        return None
    try:
        x = np.asarray([row[inputs[0]] for row in rows], dtype=float)
        y = np.asarray([row["response"] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(np.unique(x)) != len(x):
        return None
    order = np.argsort(x)
    ordered_x, ordered_y = x[order], y[order]
    delta = np.abs(np.diff(ordered_y))
    response_range = float(np.ptp(y))
    if response_range <= 1e-12 or len(delta) == 0:
        return None
    flat_fraction = float(np.mean(delta <= max(1e-12, response_range * 1e-10)))
    maximum_index = int(np.argmax(delta))
    maximum_jump = float(delta[maximum_index])
    nonzero = delta[delta > max(1e-12, response_range * 1e-10)]
    typical_change = float(np.median(nonzero)) if len(nonzero) else 0.0
    jump_ratio = maximum_jump / max(typical_change, response_range * 1e-12)
    unique_levels = int(len(np.unique(ordered_y)))
    exact_plateau_jump = flat_fraction >= 0.8 and maximum_jump >= 0.5 * response_range
    quantized_transition = (flat_fraction >= 0.5 and unique_levels <= max(16, len(y) // 4)
                            and maximum_jump >= 0.05 * response_range)
    noisy_sharp_transition = maximum_jump >= 0.35 * response_range and jump_ratio >= 20.0
    if not (exact_plateau_jump or quantized_transition or noisy_sharp_transition):
        return None
    return {
        "observed_pattern": ("exact_plateau_jump" if exact_plateau_jump else
                             "quantized_transition" if quantized_transition else
                             "sharp_transition_within_plateau_noise"),
        "hypotheses_not_distinguished": [
            "steep_smooth_transition", "discontinuous_or_quantized_transition",
        ],
        "suggested_observation_interval": [
            float(ordered_x[maximum_index]), float(ordered_x[maximum_index + 1]),
        ],
        "maximum_adjacent_response_change_fraction": maximum_jump / response_range,
        "flat_adjacent_fraction": flat_fraction,
        "unique_response_levels": unique_levels,
    }


def run_validation_routed_portfolio(
    payload: Mapping[str, Any], *, seed: int = 20261012, validation_fraction: float = 0.2,
    enable_discontinuity_gate: bool = True,
    routing_policy: str = "evaluate_both",
    solver_arm_budget: int = 2,
    current_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    gplearn_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit both methods on a training prefix and route using only held-out observations."""
    if current_solver is None:
        from .automatic_modeling import induce_and_solve_modeling_task_isolated
        current_solver = induce_and_solve_modeling_task_isolated
    if gplearn_solver is None:
        from .gplearn_baseline import fit_gplearn_baseline_isolated
        gplearn_solver = fit_gplearn_baseline_isolated
    if (not isinstance(payload, Mapping) or type(seed) is not int
            or not 0 <= seed <= 2**32 - 1
            or type(enable_discontinuity_gate) is not bool
            or routing_policy not in {"evaluate_both", "current_then_fallback"}
            or type(solver_arm_budget) is not int or solver_arm_budget not in {1, 2}
            or (routing_policy == "evaluate_both" and solver_arm_budget != 2)
            or type(validation_fraction) not in (int, float)
            or not 0.1 <= float(validation_fraction) <= 0.4):
        raise SymbolicPortfolioError("portfolio_configuration_invalid")
    transition_ambiguity = (_observed_transition_ambiguity(payload)
                            if enable_discontinuity_gate else None)
    if transition_ambiguity is not None:
        return {
            "status": "needs_input",
            "reason": "portfolio_transition_region_underobserved",
            "result_grade": "abstain",
            "recommended_action": "collect_observations_within_suggested_transition_interval",
            "routing_evidence": {
                "selection_rule": "abstain_when_transition_mechanism_is_not_identified",
                "identification_status": "smooth_vs_discontinuous_not_distinguished",
                **transition_ambiguity,
            },
            "execution_supervision": {
                "protocol": "mathmodel.sequential-portfolio/v1", "process_isolated": True,
                "permission_isolated": True, "elapsed_seconds": 0.0,
                "limits": {"wall_seconds": 60.0, "memory_mb": 1024,
                           "execution_order": "not_started_out_of_scope"},
                "child_runs": 0,
            },
            "usage": {"model_api_calls": 0, "numerical_solver_calls": 0,
                      "manual_interventions": 0},
            "policy": "transition_ambiguity_requires_local_resampling;no_structure_exclusion_claim",
        }
    fit_payload, validation_queries, validation_reference, validation_count = _split_training_validation(
        payload, seed=seed, validation_fraction=float(validation_fraction))
    training_count = len(fit_payload["attachments"][0]["rows"])

    solvers = {"current_bounded_grammar": current_solver, "official_gplearn": gplearn_solver}
    outcomes: dict[str, dict[str, Any]] = {}
    for name, solver in solvers.items():
        output: Mapping[str, Any] = {}
        try:
            output = solver(deepcopy(fit_payload))
            prediction = reexecute_submitted_model(output.get("model"), validation_queries)
            if prediction.shape != validation_reference.shape or not np.isfinite(prediction).all():
                raise SymbolicPortfolioError("portfolio_validation_prediction_invalid")
            metrics = _validation_metrics(prediction, validation_reference)
            rank = (0 if metrics["passes_primary_rule"] else 1,
                    float(metrics["nmse"]), -float(metrics["acc_0.1"]), name)
            outcomes[name] = {"output": output, "metrics": metrics, "rank": rank}
        except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
            outcomes[name] = {"output": output if isinstance(output, Mapping) else {},
                              "metrics": {"status": "failed", "reason": str(exc)},
                              "rank": (2, float("inf"), 0.0, name)}
        if (name == "current_bounded_grammar" and
                (solver_arm_budget == 1 or
                 (routing_policy == "current_then_fallback" and outcomes[name]["rank"][0] == 0))):
            break
    routing_evidence = {name: item["metrics"] for name, item in outcomes.items()}
    if "official_gplearn" not in outcomes:
        routing_evidence["official_gplearn"] = {
            "status": "not_run", "reason": ("current_arm_passed_validation"
                                               if outcomes["current_bounded_grammar"]["rank"][0] == 0
                                               else "solver_arm_budget_exhausted"),
        }
    supervision = _combined_supervision({name: item["output"] for name, item in outcomes.items()})
    selected_name = min(outcomes, key=lambda name: outcomes[name]["rank"])
    selected = outcomes[selected_name]
    if solver_arm_budget == 1 and selected["rank"][0] != 0:
        technical = selected["rank"][0] == 2
        return {
            "status": "not_assessed" if technical else "needs_input",
            "reason": "portfolio_second_arm_budget_unavailable",
            "result_grade": "not_assessed" if technical else "abstain",
            "recommended_action": ("increase_solver_budget_or_resolve_solver_failure" if technical else
                                   "increase_solver_budget_or_collect_more_observations"),
            "routing_evidence": {"seed": seed, "training_row_count": training_count,
                                 "validation_row_count": validation_count,
                                 "validation_metrics": routing_evidence,
                                 "selection_rule": "validated_first_arm_then_budget_limited_abstention"},
            "execution_supervision": supervision,
            "usage": {"model_api_calls": 0, "numerical_solver_calls": 1,
                      "manual_interventions": 0},
            "policy": "training_only_validation;one_arm_budget;no_unvalidated_prediction",
        }
    if selected["rank"][0] == 2:
        return {"status": "not_assessed", "reason": "portfolio_all_arms_failed",
                "result_grade": "not_assessed", "recommended_action": "resolve_solver_failure",
                "routing_evidence": routing_evidence, "execution_supervision": supervision,
                "usage": {"model_api_calls": 0, "numerical_solver_calls": len(outcomes),
                          "manual_interventions": 0}}
    if not bool(selected["metrics"].get("passes_primary_rule")):
        return {"status": "needs_input", "reason": "portfolio_no_validated_candidate",
                "result_grade": "abstain",
                "recommended_action": "collect_more_observations_or_expand_declared_model_scope",
                "routing_evidence": {"seed": seed, "training_row_count": training_count,
                                     "validation_row_count": validation_count,
                                     "validation_metrics": routing_evidence,
                                     "selection_rule": "abstain_when_no_arm_passes_primary_rule"},
                "execution_supervision": supervision,
                "usage": {"model_api_calls": 0, "numerical_solver_calls": len(outcomes),
                          "manual_interventions": 0},
                "policy": "training_only_validation_route;no_hidden_test_reference;safe_abstention"}
    model = deepcopy(dict(selected["output"]["model"]))
    model["portfolio_selection"] = {
        "selected_arm": selected_name, "seed": seed,
        "training_row_count": training_count,
        "validation_row_count": validation_count,
        "validation_metrics": routing_evidence,
        "selection_rule": ("current_then_fallback_primary_rule" if routing_policy == "current_then_fallback"
                           else "primary_rule_then_nmse_then_acc_then_name"),
    }
    predictions = reexecute_submitted_model(model, payload.get("query_inputs"))
    return {
        "status": "completed", "family": "modeling_algebra", "model": model,
        "predictions": predictions.tolist(),
        "result_grade": "validated_candidate", "recommended_action": "use_with_stated_scope",
        "routing_evidence": model["portfolio_selection"],
        "execution_supervision": supervision,
        "usage": {"model_api_calls": 0, "numerical_solver_calls": len(outcomes),
                  "manual_interventions": 0},
        "policy": ("training_only_validation_route;no_hidden_test_reference;second_solver_on_demand"
                   if routing_policy == "current_then_fallback" else
                   "training_only_validation_route;no_hidden_test_reference;double_solver_cost"),
    }


def run_same_split_portfolio_arm(
    payload: Mapping[str, Any], *, arm: str, seed: int = 20261012,
    validation_fraction: float = 0.2,
    current_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    gplearn_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one arm on exactly the portfolio's training split for attribution."""
    if arm not in {"current_bounded_grammar", "official_gplearn"}:
        raise SymbolicPortfolioError("portfolio_arm_invalid")
    if current_solver is None:
        from .automatic_modeling import induce_and_solve_modeling_task_isolated
        current_solver = induce_and_solve_modeling_task_isolated
    if gplearn_solver is None:
        from .gplearn_baseline import fit_gplearn_baseline_isolated
        gplearn_solver = fit_gplearn_baseline_isolated
    fit_payload, validation_queries, validation_reference, validation_count = _split_training_validation(
        payload, seed=seed, validation_fraction=float(validation_fraction))
    solver = current_solver if arm == "current_bounded_grammar" else gplearn_solver
    validation_output = solver(fit_payload)
    validation_prediction = reexecute_submitted_model(validation_output.get("model"), validation_queries)
    metrics = _validation_metrics(validation_prediction, validation_reference)
    model = deepcopy(dict(validation_output["model"]))
    model["same_split_attribution"] = {
        "arm": arm, "seed": seed,
        "training_row_count": len(fit_payload["attachments"][0]["rows"]),
        "validation_row_count": validation_count, "validation_metrics": metrics,
    }
    prediction = reexecute_submitted_model(model, payload.get("query_inputs"))
    return {"status": "completed", "family": "modeling_algebra", "model": model,
            "predictions": prediction.tolist(), "result_grade": "ungated_attribution_arm",
            "recommended_action": "comparison_only",
            "execution_supervision": validation_output.get("execution_supervision"),
            "usage": validation_output.get("usage", {"model_api_calls": 0,
                                                       "numerical_solver_calls": 1,
                                                       "manual_interventions": 0}),
            "policy": "same_training_split_as_portfolio;comparison_only"}


__all__ = ["SymbolicPortfolioError", "run_same_split_portfolio_arm",
           "run_validation_routed_portfolio"]
