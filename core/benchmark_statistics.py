"""Paired benchmark summaries with conservative uncertainty reporting."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


class BenchmarkStatisticsError(ValueError):
    pass


def paired_benchmark_effect(samples: Sequence[Mapping[str, Any]], *, baseline: str = "baseline",
                            treatment: str = "treatment", seed: int = 20260908,
                            bootstrap_replicates: int = 1000, min_samples: int = 5,
                            cluster: str | None = None) -> dict[str, Any]:
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise BenchmarkStatisticsError("samples_required")
    if (not isinstance(baseline, str) or not baseline.strip() or
            not isinstance(treatment, str) or not treatment.strip() or baseline == treatment):
        raise BenchmarkStatisticsError("baseline_treatment_names_invalid")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise BenchmarkStatisticsError("invalid_bootstrap_seed")
    if type(min_samples) is not int or not 1 <= min_samples <= 100_000:
        raise BenchmarkStatisticsError("invalid_minimum_sample_count")
    if type(bootstrap_replicates) is not int or not 100 <= bootstrap_replicates <= 20_000:
        raise BenchmarkStatisticsError("invalid_bootstrap_replicates")
    if len(samples) > 100_000:
        raise BenchmarkStatisticsError("sample_budget_exceeded")
    if cluster is not None and (not isinstance(cluster, str) or not cluster.strip()
                                or cluster in {baseline, treatment}):
        raise BenchmarkStatisticsError("cluster_name_invalid")
    rows = []
    cluster_ids: list[str] = []
    for item in samples:
        if not isinstance(item, Mapping):
            raise BenchmarkStatisticsError("sample_must_be_mapping")
        try:
            left, right = float(item[baseline]), float(item[treatment])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkStatisticsError("paired_values_required") from exc
        if not math.isfinite(left) or not math.isfinite(right):
            raise BenchmarkStatisticsError("paired_values_must_be_finite")
        rows.append(right - left)
        if cluster is not None:
            value = item.get(cluster)
            if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
                raise BenchmarkStatisticsError("cluster_id_required")
            cluster_ids.append(str(value))
    if not rows:
        raise BenchmarkStatisticsError("samples_must_not_be_empty")
    raw_values = np.asarray(rows, dtype=float)
    if cluster is None:
        values = raw_values
    else:
        ordered_ids = list(dict.fromkeys(cluster_ids))
        values = np.asarray([raw_values[np.asarray(cluster_ids) == identifier].mean()
                             for identifier in ordered_ids], dtype=float)
    mean = float(values.mean())
    result = {"schema_version": "mathmodel.benchmark-statistics/v1", "sample_count": len(raw_values),
              "independent_unit_count": len(values),
              "resampling_unit": "cluster" if cluster is not None else "sample",
              "cluster_field": cluster,
              "paired_effect": mean, "baseline": baseline, "treatment": treatment,
              "status": "descriptive_only" if len(values) < min_samples else "assessed",
              "policy": "paired_effect_is_not_significance_or_zero_risk_proof;dependent_variants_resampled_by_cluster"}
    if len(values) < min_samples:
        result["confidence_interval"] = None
        result["reason"] = "independent_unit_count_below_predeclared_minimum"
        return result
    if bootstrap_replicates * len(values) > 20_000_000:
        raise BenchmarkStatisticsError("bootstrap_memory_budget_exceeded")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
    means = values[indices].mean(axis=1)
    result["confidence_interval"] = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    result["bootstrap_seed"] = int(seed)
    result["bootstrap_replicates"] = int(bootstrap_replicates)
    return result


__all__ = ["BenchmarkStatisticsError", "paired_benchmark_effect"]
