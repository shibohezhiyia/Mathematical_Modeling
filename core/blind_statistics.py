"""Statistical claims for independently scored real unseen problems.

The ordinary benchmark reports in this project are deliberately descriptive.
This module is the stricter, opt-in layer: it can report accuracy and a paired
significance test only after a sealed manifest, external-real cases, unlocked
reference scores, complete score coverage (or an explicit failure policy), and
an independent-evaluation attestation have all been supplied.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .blind_benchmark import (
    BLIND_SCHEMA, RUN_SCHEMA, SCORE_SCHEMA, RUN_STATUSES, BlindBenchmarkError,
    BlindManifest,
)


BLIND_STATISTICS_SCHEMA = "mathmodel.blind-statistics/v1"
_PRIMARY_SCORES = frozenset({
    "contract_correct", "numerically_correct", "constraint_validity",
    "evidence_completeness", "stability", "human_judgement",
})


class BlindStatisticsError(ValueError):
    """Raised when a statistical report request is malformed."""


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired outcomes."""
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1))
    p = 2.0 * tail / (2.0 ** discordant)
    return min(1.0, float(p))


def _not_assessed(reason: str, *, case_count: int, baseline: str, treatment: str) -> dict[str, Any]:
    return {
        "schema_version": BLIND_STATISTICS_SCHEMA,
        "status": "not_assessed",
        "accuracy_claim": "not_assessed",
        "statistical_claim": "not_assessed",
        "reason": reason,
        "case_count": case_count,
        "baseline_system": baseline,
        "treatment_system": treatment,
        "policy": "real_unseen_accuracy_requires_sealed_external_cases_and_independent_unlocked_scores",
    }


def assess_blind_accuracy(
    manifest: BlindManifest | Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    *,
    baseline_system: str,
    treatment_system: str,
    primary_score: str = "numerically_correct",
    success_threshold: float = 0.5,
    min_cases: int = 20,
    alpha: float = 0.05,
    bootstrap_replicates: int = 2000,
    seed: int = 20260910,
    independent_evaluation_attested: bool = False,
    failure_as_incorrect: bool = True,
) -> dict[str, Any]:
    """Assess two systems on a sealed real-unseen case set.

    Completed runs are scored from unlocked reference records.  Failed or
    missing runs count as incorrect only when ``failure_as_incorrect`` is true;
    otherwise the report is not assessed unless every system/case cell is
    complete.  The paired exact McNemar test compares binary successes and a
    deterministic bootstrap interval describes the paired accuracy difference.
    A significant result is conditional on the supplied attestation and cannot
    certify that the external case sample represents every future problem.
    """
    if not isinstance(manifest, BlindManifest):
        if not isinstance(manifest, Mapping):
            raise BlindStatisticsError("manifest_required")
        try:
            manifest = BlindManifest.from_payload(dict(manifest))
        except BlindBenchmarkError as exc:
            raise BlindStatisticsError(str(exc)) from exc
    if manifest.public().get("schema_version") != BLIND_SCHEMA:
        raise BlindStatisticsError("manifest_schema_mismatch")
    if not isinstance(baseline_system, str) or not baseline_system.strip():
        raise BlindStatisticsError("baseline_system_required")
    if not isinstance(treatment_system, str) or not treatment_system.strip() or baseline_system == treatment_system:
        raise BlindStatisticsError("treatment_system_invalid")
    baseline_system, treatment_system = baseline_system.strip(), treatment_system.strip()
    if primary_score not in _PRIMARY_SCORES:
        raise BlindStatisticsError("primary_score_invalid")
    if type(success_threshold) not in (int, float) or isinstance(success_threshold, bool) or not 0 <= float(success_threshold) <= 1:
        raise BlindStatisticsError("success_threshold_invalid")
    if type(min_cases) is not int or not 1 <= min_cases <= 100_000:
        raise BlindStatisticsError("min_cases_invalid")
    if type(alpha) not in (int, float) or isinstance(alpha, bool) or not 0 < float(alpha) < 1:
        raise BlindStatisticsError("alpha_invalid")
    if type(bootstrap_replicates) is not int or not 100 <= bootstrap_replicates <= 20_000:
        raise BlindStatisticsError("bootstrap_replicates_invalid")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise BlindStatisticsError("seed_invalid")
    if type(independent_evaluation_attested) is not bool or type(failure_as_incorrect) is not bool:
        raise BlindStatisticsError("attestation_flags_must_be_boolean")
    cases = manifest.cases(split="unseen")
    all_cases = manifest.cases()
    if len(cases) != len(all_cases):
        return _not_assessed("manifest_contains_non_unseen_cases", case_count=len(all_cases),
                             baseline=baseline_system, treatment=treatment_system)
    if not cases:
        return _not_assessed("no_unseen_cases", case_count=0, baseline=baseline_system, treatment=treatment_system)
    if any(case.provenance != "external_real" for case in cases):
        return _not_assessed("cases_are_not_all_external_real", case_count=len(cases),
                             baseline=baseline_system, treatment=treatment_system)
    if any(case.answer_sha256 is None for case in cases):
        return _not_assessed("independent_reference_commitment_missing", case_count=len(cases),
                             baseline=baseline_system, treatment=treatment_system)
    if not independent_evaluation_attested:
        return _not_assessed("independent_evaluation_attestation_required", case_count=len(cases),
                             baseline=baseline_system, treatment=treatment_system)
    if len(cases) < min_cases:
        return _not_assessed("case_count_below_predeclared_minimum", case_count=len(cases),
                             baseline=baseline_system, treatment=treatment_system)
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise BlindStatisticsError("runs_required")
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise BlindStatisticsError("scores_required")
    case_ids = {case.case_id for case in cases}
    run_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    run_by_id: dict[str, Mapping[str, Any]] = {}
    for row in runs:
        if not isinstance(row, Mapping) or row.get("schema_version") != RUN_SCHEMA:
            raise BlindStatisticsError("run_schema_mismatch")
        if row.get("manifest_digest") != manifest.digest or row.get("case_id") not in case_ids:
            raise BlindStatisticsError("run_scope_mismatch")
        system = row.get("system_version")
        run_id = row.get("run_id")
        if not isinstance(system, str) or not system.strip() or not isinstance(run_id, str) or not run_id.strip():
            raise BlindStatisticsError("run_identity_invalid")
        if row.get("status") not in RUN_STATUSES:
            raise BlindStatisticsError("run_status_invalid")
        if row.get("budget") != manifest.budget:
            raise BlindStatisticsError("run_budget_mismatch")
        key = (system.strip(), row["case_id"])
        if key in run_by_key or run_id in run_by_id:
            raise BlindStatisticsError("duplicate_run_identity")
        run_by_key[key] = row
        run_by_id[run_id] = row
    score_by_run: dict[str, Mapping[str, Any]] = {}
    evaluators: set[str] = set()
    for row in scores:
        if not isinstance(row, Mapping) or row.get("schema_version") != SCORE_SCHEMA or row.get("manifest_digest") != manifest.digest or row.get("unlocked") is not True:
            raise BlindStatisticsError("score_scope_mismatch")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id not in run_by_id:
            raise BlindStatisticsError("score_run_unknown")
        if row.get("case_id") != run_by_id[run_id].get("case_id"):
            raise BlindStatisticsError("score_case_mismatch")
        evaluator = row.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator.strip():
            raise BlindStatisticsError("evaluator_required")
        evaluator = evaluator.strip()
        if evaluator == run_by_id[run_id].get("system_version"):
            return _not_assessed("evaluator_matches_system", case_count=len(cases),
                                 baseline=baseline_system, treatment=treatment_system)
        if run_id in score_by_run:
            raise BlindStatisticsError("duplicate_score_run")
        score_map = row.get("scores")
        if not isinstance(score_map, Mapping) or primary_score not in score_map:
            raise BlindStatisticsError("primary_score_missing")
        value = score_map[primary_score]
        if primary_score == "contract_correct":
            if type(value) is not bool:
                raise BlindStatisticsError("contract_score_must_be_boolean")
        elif type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise BlindStatisticsError("score_value_invalid")
        score_by_run[run_id] = row
        evaluators.add(evaluator)
    if not evaluators:
        return _not_assessed("unlocked_scores_required", case_count=len(cases),
                             baseline=baseline_system, treatment=treatment_system)

    outcomes: dict[str, list[int]] = {}
    missing_cells = 0
    failed_cells = 0
    for system in (baseline_system, treatment_system):
        values: list[int] = []
        for case in cases:
            run = run_by_key.get((system, case.case_id))
            if run is None or run.get("status") != "completed":
                failed_cells += 1
                values.append(0)
                continue
            score = score_by_run.get(run["run_id"])
            if score is None:
                missing_cells += 1
                values.append(0)
                continue
            raw = score["scores"][primary_score]
            value = raw if primary_score == "contract_correct" else float(raw) >= float(success_threshold)
            values.append(int(bool(value)))
        outcomes[system] = values
    if not failure_as_incorrect and (failed_cells or missing_cells):
        return _not_assessed("incomplete_cells_and_failure_policy_disallows_imputation", case_count=len(cases),
                             baseline=baseline_system, treatment=treatment_system)
    n = len(cases)
    base = np.asarray(outcomes[baseline_system], dtype=int)
    treatment = np.asarray(outcomes[treatment_system], dtype=int)
    differences = treatment - base
    b = int(np.sum((base == 0) & (treatment == 1)))
    c = int(np.sum((base == 1) & (treatment == 0)))
    p_value = _exact_mcnemar_pvalue(b, c)
    rng = np.random.default_rng(seed)
    if bootstrap_replicates * n > 20_000_000:
        raise BlindStatisticsError("bootstrap_memory_budget_exceeded")
    sampled = differences[rng.integers(0, n, size=(bootstrap_replicates, n))].mean(axis=1)
    interval = [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]
    base_success, treatment_success = int(base.sum()), int(treatment.sum())
    base_accuracy = base_success / n
    treatment_accuracy = treatment_success / n
    claimable = bool(len(cases) >= min_cases and independent_evaluation_attested and not (not failure_as_incorrect and (failed_cells or missing_cells)))
    statistically_significant = bool(claimable and p_value < float(alpha) and interval[0] > 0)
    return {
        "schema_version": BLIND_STATISTICS_SCHEMA,
        "status": "statistically_significant" if statistically_significant else ("assessed_not_significant" if claimable else "not_assessed"),
        "accuracy_claim": "conditional_real_unseen" if claimable else "not_assessed",
        "statistical_claim": "paired_mcnemar_exact" if claimable else "not_assessed",
        "manifest_digest": manifest.digest,
        "case_count": n, "baseline_system": baseline_system, "treatment_system": treatment_system,
        "primary_score": primary_score, "success_threshold": float(success_threshold),
        "baseline_accuracy": base_accuracy, "treatment_accuracy": treatment_accuracy,
        "baseline_accuracy_interval": list(_wilson(base_success, n)),
        "treatment_accuracy_interval": list(_wilson(treatment_success, n)),
        "paired_accuracy_difference": treatment_accuracy - base_accuracy,
        "paired_difference_interval": interval,
        "mcnemar": {"treatment_only": b, "baseline_only": c, "discordant": b + c,
                    "p_value": p_value, "alpha": float(alpha)},
        "coverage": {"failed_cells": failed_cells, "missing_score_cells": missing_cells,
                     "failure_as_incorrect": failure_as_incorrect},
        "evaluators": sorted(evaluators),
        "bootstrap_seed": seed, "bootstrap_replicates": bootstrap_replicates,
        "policy": "sealed_external_real_unseen_cases;_unlocked_reference_scores;independent_evaluation_attested;exact_paired_mcnemar;bootstrap_ci;conditional_not_universal_generalization_proof",
    }


__all__ = ["BLIND_STATISTICS_SCHEMA", "BlindStatisticsError", "assess_blind_accuracy"]
