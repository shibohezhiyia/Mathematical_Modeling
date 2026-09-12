"""Empirical observation/preprocessing uncertainty accounting.

Rows must identify the same observation across repeated measurements or
preprocessing variants.  The result is descriptive variance accounting, not a
measurement-error model or confidence interval.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


class ObservationUncertaintyError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def assess_observation_uncertainty(
    records: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
    observation_key: str = "observation_id",
    processing_key: str = "processing_id",
    unit_signature: str | None = None,
    max_records: int = 100_000,
) -> dict[str, Any]:
    if type(max_records) is not int or not 2 <= max_records <= 1_000_000:
        raise ObservationUncertaintyError("invalid_record_budget")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not 2 <= len(records) <= max_records:
        raise ObservationUncertaintyError("record_count_out_of_bounds")
    groups: dict[str, list[float]] = defaultdict(list)
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    declared_noise: list[float] = []
    units: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ObservationUncertaintyError(f"record_{index}_must_be_object")
        observation = row.get(observation_key)
        processing = row.get(processing_key)
        if type(observation) not in (str, int) or isinstance(observation, bool) or type(processing) not in (str, int) or isinstance(processing, bool):
            raise ObservationUncertaintyError("observation_and_processing_ids_required")
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ObservationUncertaintyError("value_must_be_finite") from exc
        if not math.isfinite(value):
            raise ObservationUncertaintyError("value_must_be_finite")
        if "observation_noise_variance" in row:
            noise = row["observation_noise_variance"]
            if (isinstance(noise, bool) or not isinstance(noise, (int, float))
                    or not math.isfinite(float(noise)) or float(noise) < 0):
                raise ObservationUncertaintyError("observation_noise_variance_must_be_finite_nonnegative")
            declared_noise.append(float(noise))
        row_unit = row.get("unit_signature", unit_signature)
        if not isinstance(row_unit, str) or not row_unit.strip():
            raise ObservationUncertaintyError("unit_signature_required")
        units.add(row_unit.strip())
        obs_key, process_key = str(observation), str(processing)
        groups[obs_key].append(value)
        cells[(obs_key, process_key)].append(value)
    if len(units) != 1:
        raise ObservationUncertaintyError("unit_signatures_must_match")
    all_values = [value for values in groups.values() for value in values]
    mean = sum(all_values) / len(all_values)
    total_variance = sum((value - mean) ** 2 for value in all_values) / len(all_values)
    group_means = {key: sum(values) / len(values) for key, values in groups.items()}
    between = sum(len(values) * (group_means[key] - mean) ** 2
                  for key, values in groups.items()) / len(all_values)
    within = sum(sum((value - group_means[key]) ** 2 for value in values)
                 for key, values in groups.items()) / len(all_values)
    process_values: dict[str, list[float]] = defaultdict(list)
    cell_means: dict[tuple[str, str], float] = {}
    for (obs_key, process_key), values in cells.items():
        cell_mean = sum(values) / len(values)
        cell_means[(obs_key, process_key)] = cell_mean
        process_values[process_key].extend(values)
    process_means = {key: sum(values) / len(values) for key, values in process_values.items()}
    between_processing = sum(
        len(values) * (process_means[key] - mean) ** 2
        for key, values in process_values.items()
    ) / len(all_values)
    interaction = 0.0
    for (obs_key, process_key), values in cells.items():
        effect = (cell_means[(obs_key, process_key)] - group_means[obs_key]
                  - process_means[process_key] + mean)
        interaction += len(values) * effect * effect
    interaction /= len(all_values)
    within_cell = sum(
        sum((value - cell_means[(obs_key, process_key)]) ** 2 for value in values)
        for (obs_key, process_key), values in cells.items()
    ) / len(all_values)
    # Nested/interaction decomposition is exact for balanced designs.  For an
    # unbalanced table retain the residual instead of silently adding correlated
    # components as if they were independent.
    residual = total_variance - (between + between_processing + interaction + within_cell)
    return {
        "schema_version": "mathmodel.observation-uncertainty/v1",
        "status": "assessed" if all(len(values) >= 2 for values in groups.values()) else "partial",
        "unit_signature": next(iter(units)),
        "record_count": len(all_values),
        "observation_count": len(groups),
        "processing_variants": len({str(row[processing_key]) for row in records}),
        "mean": mean,
        "total_variance": max(0.0, total_variance),
        "between_observation_variance": max(0.0, between),
        "within_observation_variance": max(0.0, within),
        "between_processing_variance": max(0.0, between_processing),
        "observation_processing_interaction_variance": max(0.0, interaction),
        "within_processing_variance": max(0.0, within_cell),
        "declared_observation_noise_variance": (
            sum(declared_noise) / len(declared_noise) if declared_noise else None
        ),
        "declared_noise_record_count": len(declared_noise),
        "variance_reconstruction_residual": residual,
        "decomposition": "observation + processing + observation_processing_interaction + within_processing",
        "policy": "empirical_observation_and_preprocessing_variance; declared_noise_is_metadata_not_automatically_subtracted",
    }


__all__ = ["ObservationUncertaintyError", "assess_observation_uncertainty"]
