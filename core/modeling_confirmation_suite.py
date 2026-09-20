"""Prospective internal confirmation cases for bounded automatic modeling.

The generator transforms the development structures without changing their
mathematical meaning.  It is useful for a freeze-then-run check of parameter,
row-order, attachment-order, attachment-name, and numerical-scale robustness.
It is explicitly not an unseen-structure or external-domain benchmark.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

from .automated_benchmark import AutomatedBenchmarkCase
from .modeling_benchmark_suite import build_modeling_benchmark_suite


class ModelingConfirmationAlreadyConsumed(RuntimeError):
    pass


def reserve_modeling_confirmation(
    directory: str | Path, *, protocol_id: str, seed: int,
) -> Path:
    """Atomically consume a protocol/seed before any hidden cases are built."""
    if protocol_id not in {"modeling-internal-confirmation-v1", "modeling-structure-challenge-v1",
                           "modeling-extension-confirmation-v1", "modeling-open-structure-challenge-v2",
                           "modeling-unseen-structure-confirmation-v3", "modeling-compositional-holdout-v4",
                           "modeling-equivalence-holdout-v5", "modeling-search-strategy-holdout-v6",
                           "modeling-delay-holdout-v7", "modeling-logic-temporal-holdout-v8",
                           "modeling-beam-holdout-v9", "modeling-depth-three-holdout-v10",
                           "llm-srbench-mirror-confirmation-v11",
                           "llm-srbench-trig-confirmation-v12",
                           "llm-srbench-extended-confirmation-v13",
                           "llm-srbench-transformed-power-confirmation-v14",
                           "llm-srbench-harmonic-confirmation-v15",
                           "llm-srbench-angular-confirmation-v16",
                           "llm-srbench-product-angle-confirmation-v17",
                           "srsd-official-final-confirmation-v18",
                           "srsd-gplearn-confirmation-v19",
                           "srsd-portfolio-confirmation-v20",
                           "product-workflow-confirmation-v21",
                           "product-discontinuity-confirmation-v22",
                           "product-routing-confirmation-v23",
                           "product-fallback-confirmation-v24",
                           "product-rescue-confirmation-v25",
                           "mixed-risk-confirmation-v26",
                           "budget-decision-confirmation-v27",
                           "sparse-operator-topology-confirmation-v28",
                           "scipy-function-confirmation-v29",
                           "sparse-robustness-confirmation-v30",
                           "validation-gated-sparse-confirmation-v31",
                           "airfoil-raw-source-confirmation-v32",
                           "sparse-irregular-noise-confirmation-v33",
                           "sparse-structure-selection-confirmation-v34",
                           "sparse-multistart-search-confirmation-v35",
                           "yacht-group-holdout-confirmation-v36"}:
        raise ValueError("confirmation_protocol_invalid")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("confirmation_seed_invalid")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f"{protocol_id}-{seed}.json"
    payload = {"protocol_id": protocol_id, "seed": seed, "status": "reserved",
               "policy": "attempt_is_consumed_before_case_generation;failure_does_not_release_seed"}
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    except FileExistsError as exc:
        raise ModelingConfirmationAlreadyConsumed("confirmation_seed_already_consumed") from exc
    return marker


def complete_modeling_confirmation(marker: str | Path, *, outcome: str, report_path: str) -> None:
    if outcome not in {"completed", "failed"}:
        raise ValueError("confirmation_outcome_invalid")
    path = Path(marker)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "reserved":
        raise ModelingConfirmationAlreadyConsumed("confirmation_attempt_not_reserved")
    payload.update(status=outcome, report_path=str(report_path))
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _scale_case(public: dict[str, Any], reference: dict[str, Any], family: str, scale: float) -> None:
    attachments = public["attachments"]
    if family == "modeling_algebra":
        for attachment in attachments:
            for row in attachment["rows"]:
                if "response" in row:
                    row["response"] = float(row["response"]) * scale
        reference["coefficients"] = [float(value) * scale for value in reference["coefficients"]]
        reference["predictions"] = [float(value) * scale for value in reference["predictions"]]
    elif family == "modeling_ode":
        for attachment in attachments:
            for row in attachment["rows"]:
                if "state" in row:
                    row["state"] = float(row["state"]) * scale
        reference["trajectory"] = [float(value) * scale for value in reference["trajectory"]]
        if reference["structure"] == "logistic_growth":
            reference["parameters"]["capacity"] = float(reference["parameters"]["capacity"]) * scale
    elif family == "modeling_optimization":
        objective_field = "profit" if reference["direction"] == "maximize" else "cost"
        for attachment in attachments:
            for row in attachment["rows"]:
                if objective_field in row:
                    row[objective_field] = float(row[objective_field]) * scale
        reference["objective"] = float(reference["objective"]) * scale
        reference["objective_coefficients"] = [
            float(value) * scale for value in reference["objective_coefficients"]
        ]
    elif family == "modeling_multi_table":
        for attachment in attachments:
            for row in attachment["rows"]:
                if "amount" in row:
                    row["amount"] = float(row["amount"]) * scale
        if "group_totals" in reference:
            reference["group_totals"] = {
                key: float(value) * scale for key, value in reference["group_totals"].items()
            }


def build_modeling_confirmation_suite(
    *, seed: int = 20260916, variants_per_structure: int = 3,
) -> tuple[AutomatedBenchmarkCase, ...]:
    """Create a deterministic post-freeze invariance/parameter confirmation pack."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("confirmation_seed_invalid")
    if type(variants_per_structure) is not int or not 1 <= variants_per_structure <= 8:
        raise ValueError("confirmation_variant_budget_invalid")
    rng = random.Random(seed)
    result: list[AutomatedBenchmarkCase] = []
    for base in build_modeling_benchmark_suite():
        for variant in range(variants_per_structure):
            public = deepcopy(base.public_input)
            reference = deepcopy(base.hidden_reference)
            scale = 0.65 + 0.17 * variant + 0.03 * rng.randint(0, 7)
            _scale_case(public, reference, base.family, scale)

            for attachment in public["attachments"]:
                rng.shuffle(attachment["rows"])
            if variant == 1:
                public["attachments"].reverse()
                for index, attachment in enumerate(public["attachments"]):
                    attachment["name"] = f"raw_{index + 1}"
            if variant == 2:
                if "query_inputs" in public:
                    public["query_inputs"].reverse()
                    reference["predictions"].reverse()
                if "query_times" in public:
                    public["query_times"].reverse()
                    reference["trajectory"].reverse()

            case = AutomatedBenchmarkCase(
                case_id=f"confirm-{base.case_id}-v{variant + 1}",
                family=base.family,
                statement=base.statement,
                public_input=public,
                hidden_reference=reference,
                structure_group=base.structure_group,
                source_group="generated-modeling-confirmation-v1",
            )
            case.validate()
            result.append(case)
    return tuple(result)


__all__ = ["ModelingConfirmationAlreadyConsumed", "reserve_modeling_confirmation",
           "complete_modeling_confirmation", "build_modeling_confirmation_suite"]
