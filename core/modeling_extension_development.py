"""Development-only cases derived from consumed structure-challenge failures."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Sequence

from .automated_benchmark import AutomatedBenchmarkCase
from .modeling_structure_challenge import build_modeling_structure_challenge


def build_modeling_extension_development_suite(
    *, seed: int = 20260918,
) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("extension_development_seed_invalid")
    rng = random.Random(seed)
    result = []
    for original in build_modeling_structure_challenge(seed=seed):
        public = deepcopy(original.public_input)
        reference = deepcopy(original.hidden_reference)
        for index, attachment in enumerate(public["attachments"]):
            rng.shuffle(attachment["rows"])
            attachment["name"] = f"development_raw_{index + 1}"

        if original.case_id == "challenge-algebra-three-way":
            coefficient = float(reference["coefficients"][0])
            reference["coefficients"] = [coefficient, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 2.0 * coefficient]
        elif original.case_id == "challenge-ode-external-drive":
            factor = 1.4
            for row in public["attachments"][0]["rows"]:
                row["state"] *= factor
            reference["trajectory"] = [value * factor for value in reference["trajectory"]]
            reference["parameters"]["forcing"] *= factor
        elif original.case_id == "challenge-ode-coupled-state":
            for row in public["attachments"][0]["rows"]:
                row["time"] *= 2.0
            public["query_times"] = [value * 2.0 for value in public["query_times"]]
            reference["parameters"]["decay"] /= 2.0
            reference["parameters"]["coupling"] /= 2.0
        elif original.case_id == "challenge-optimization-integer":
            objective_field = "profit"
            items = next(attachment["rows"] for attachment in public["attachments"]
                         if objective_field in attachment["rows"][0])
            by_name = {row["item"]: row for row in items}
            by_name["A"][objective_field], by_name["B"][objective_field] = 7.0 * 1.3, 3.0 * 1.3
            reference["objective_coefficients"] = [7.0 * 1.3, 3.0 * 1.3]
            reference["objective"] = 10.0 * 1.3
            reference["known_solution"] = [1.0, 1.0]
        elif original.family == "modeling_optimization":
            objective_field = "profit" if reference["direction"] == "maximize" else "cost"
            for attachment in public["attachments"]:
                for row in attachment["rows"]:
                    if objective_field in row:
                        row[objective_field] *= 1.3
            reference["objective_coefficients"] = [value * 1.3 for value in reference["objective_coefficients"]]
            reference["objective"] *= 1.3
        elif original.family == "modeling_multi_table" and "group_totals" in reference:
            for attachment in public["attachments"]:
                for row in attachment["rows"]:
                    if "amount" in row:
                        row["amount"] *= 1.25
            reference["group_totals"] = {key: value * 1.25 for key, value in reference["group_totals"].items()}

        case = AutomatedBenchmarkCase(
            case_id=original.case_id.replace("challenge-", "extension-development-"),
            family=original.family,
            statement=original.statement,
            public_input=public,
            hidden_reference=reference,
            structure_group=original.structure_group,
            source_group="post-challenge-development-v1",
        )
        case.validate()
        result.append(case)
    return tuple(result)


def run_modeling_extension_ablation(
    *, cases: Sequence[AutomatedBenchmarkCase] | None = None,
) -> dict[str, Any]:
    """Attribute development successes to four bounded structure extensions."""
    from .automatic_modeling import induce_and_solve_modeling_task
    from .modeling_benchmark_suite import score_modeling_benchmark_case

    suite = tuple(cases or build_modeling_extension_development_suite())
    variants = ("full", "without_higher_order_algebra", "without_extended_ode",
                "without_extended_optimization", "without_three_table_reasoning")
    family_gate = {
        "without_higher_order_algebra": "modeling_algebra",
        "without_extended_ode": "modeling_ode",
        "without_extended_optimization": "modeling_optimization",
        "without_three_table_reasoning": "modeling_multi_table",
    }
    rows = []
    for case in suite:
        full = induce_and_solve_modeling_task(case.public_input)
        for variant in variants:
            output = deepcopy(full)
            if family_gate.get(variant) == case.family:
                output = {"status": "incomplete", "reason": f"{variant}_disabled"}
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            rows.append({"task_id": case.case_id, "family": case.family, "variant": variant,
                         "valid": bool(scored["valid"]), "score": scored.get("score"),
                         "reason": scored.get("reason")})
    summary = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {"case_count": len(selected),
                            "valid_count": sum(row["valid"] for row in selected),
                            "valid_rate": sum(row["valid"] for row in selected) / len(selected)}
    return {"schema_version": "mathmodel.modeling-extension-ablation/v1",
            "status": "descriptive_post_challenge_development_only",
            "rows": rows, "summary": summary,
            "policy": "consumed_challenge_used_only_for_failure_taxonomy;separate_development_parameters;output_gate_ablation;not_confirmation"}


def build_modeling_extension_confirmation_suite(
    *, seed: int = 20260919, variants_per_structure: int = 2,
) -> tuple[AutomatedBenchmarkCase, ...]:
    """Create new parameter/representation cases for the extended structures."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("extension_confirmation_seed_invalid")
    if type(variants_per_structure) is not int or not 1 <= variants_per_structure <= 4:
        raise ValueError("extension_confirmation_variant_budget_invalid")
    rng = random.Random(seed)
    result = []
    for base in build_modeling_extension_development_suite(seed=seed):
        for variant in range(variants_per_structure):
            public = deepcopy(base.public_input)
            reference = deepcopy(base.hidden_reference)
            factor = 0.72 + 0.19 * variant + 0.02 * rng.randint(0, 6)
            for attachment in public["attachments"]:
                rng.shuffle(attachment["rows"])
            public["attachments"].reverse()
            for index, attachment in enumerate(public["attachments"]):
                attachment["name"] = f"sealed_raw_{variant + 1}_{index + 1}"

            if base.family == "modeling_algebra":
                for attachment in public["attachments"]:
                    for row in attachment["rows"]:
                        if "response" in row:
                            row["response"] *= factor
                reference["coefficients"] = [value * factor for value in reference["coefficients"]]
                reference["predictions"] = [value * factor for value in reference["predictions"]]
            elif base.family == "modeling_ode":
                for attachment in public["attachments"]:
                    for row in attachment["rows"]:
                        for key in tuple(row):
                            if key == "state" or key.startswith("state_"):
                                row[key] *= factor
                reference["trajectory"] = [
                    [value * factor for value in row] if isinstance(row, list) else row * factor
                    for row in reference["trajectory"]
                ]
                if "forcing" in reference.get("parameters", {}):
                    reference["parameters"]["forcing"] *= factor
            elif base.family == "modeling_optimization":
                objective_field = "profit" if reference["direction"] == "maximize" else "cost"
                for attachment in public["attachments"]:
                    for row in attachment["rows"]:
                        if objective_field in row:
                            row[objective_field] *= factor
                reference["objective_coefficients"] = [value * factor for value in reference["objective_coefficients"]]
                reference["objective"] *= factor
            elif "group_totals" in reference:
                for attachment in public["attachments"]:
                    for row in attachment["rows"]:
                        if "amount" in row:
                            row["amount"] *= factor
                reference["group_totals"] = {key: value * factor for key, value in reference["group_totals"].items()}

            case = AutomatedBenchmarkCase(
                case_id=f"extension-confirm-{base.structure_group}-v{variant + 1}",
                family=base.family, statement=base.statement, public_input=public,
                hidden_reference=reference, structure_group=base.structure_group,
                source_group="generated-extension-confirmation-v1",
            )
            case.validate()
            result.append(case)
    return tuple(result)


__all__ = ["build_modeling_extension_development_suite", "run_modeling_extension_ablation",
           "build_modeling_extension_confirmation_suite"]
