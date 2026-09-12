"""Deterministic preview sampling that preserves rare and anomalous rows."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


class PreviewSamplingError(ValueError):
    pass


def sample_preview_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = 500,
    group_key: str | None = None,
    anomaly_key: str | None = None,
    rare_group_limit: int = 3,
) -> dict[str, Any]:
    """Select a bounded preview without treating it as final evaluation data."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise PreviewSamplingError("rows_must_be_sequence")
    if type(max_rows) is not int or not 1 <= max_rows <= 100_000:
        raise PreviewSamplingError("invalid_max_rows")
    if type(rare_group_limit) is not int or not 0 <= rare_group_limit <= 100:
        raise PreviewSamplingError("invalid_rare_group_limit")
    if group_key is not None and (not isinstance(group_key, str) or not group_key.strip()):
        raise PreviewSamplingError("invalid_group_key")
    if anomaly_key is not None and (not isinstance(anomaly_key, str) or not anomaly_key.strip()):
        raise PreviewSamplingError("invalid_anomaly_key")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PreviewSamplingError("rows_must_contain_objects")
    if not rows:
        return {"schema_version": "mathmodel.preview-sampling/v1", "status": "empty", "rows": [], "selected_indices": []}
    group_values = [str(row.get(group_key, "__missing__")) if group_key else "__all__" for row in rows]
    counts = Counter(group_values)
    rare_groups = {group for group, count in counts.items() if count <= rare_group_limit} if group_key else set()
    selected: set[int] = set()
    selected.update(index for index, group in enumerate(group_values) if group in rare_groups)
    if anomaly_key:
        scored = []
        for index, row in enumerate(rows):
            value = row.get(anomaly_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            scored.append((abs(float(value)), index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected.update(index for _, index in scored[:max_rows])
    if len(selected) < max_rows:
        # Evenly spaced indices preserve temporal/row-order coverage.  The
        # stable tie break makes cached previews reproducible.
        stride = max(1, len(rows) // max_rows)
        selected.update(range(0, len(rows), stride))
    if len(selected) > max_rows:
        # Priority: rare groups, then anomaly magnitude, then earliest rows.
        def priority(index: int) -> tuple[int, float, int]:
            anomaly = rows[index].get(anomaly_key) if anomaly_key else None
            score = abs(float(anomaly)) if isinstance(anomaly, (int, float)) and not isinstance(anomaly, bool) and math.isfinite(float(anomaly)) else 0.0
            return (1 if group_values[index] in rare_groups else 0, score, -index)
        selected = set(sorted(selected, key=priority, reverse=True)[:max_rows])
    selected_indices = sorted(selected)
    return {
        "schema_version": "mathmodel.preview-sampling/v1",
        "status": "complete" if len(rows) <= max_rows else "sampled",
        "source_rows": len(rows),
        "selected_indices": selected_indices,
        "rows": [dict(rows[index]) for index in selected_indices],
        "preserved_rare_groups": sorted(rare_groups),
        "anomaly_key": anomaly_key,
        "policy": "preview_only; never substitutes for final evaluation",
    }


__all__ = ["PreviewSamplingError", "sample_preview_rows"]
