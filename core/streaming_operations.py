"""Memory-bounded operations over chunk iterators.

The functions here are intentionally explicit about what they materialize.
They reject an over-budget join instead of sampling or silently truncating a
many-to-many result, which is critical for mathematical-model evidence.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd


class StreamingOperationError(ValueError):
    pass


def _keys(keys: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence) or not keys:
        raise StreamingOperationError(f"{label}_must_be_nonempty_sequence")
    result = tuple(item.strip() for item in keys if type(item) is str and item.strip())
    if len(result) != len(keys) or len(set(result)) != len(result):
        raise StreamingOperationError(f"invalid_{label}")
    return result


def bounded_stream_hash_join(
    left_chunks: Iterable[pd.DataFrame],
    right_chunks: Iterable[pd.DataFrame],
    *,
    left_on: Sequence[str],
    right_on: Sequence[str] | None = None,
    right_columns: Sequence[str] | None = None,
    max_right_rows: int = 500_000,
    max_output_rows: int = 2_000_000,
) -> pd.DataFrame:
    """Join chunks with a bounded right-side hash table.

    The right side is deliberately retained in memory because a portable
    external sort/SQLite backend is not assumed.  If the bound is exceeded,
    the operation fails explicitly; it never drops rows to make the join fit.
    """
    if not callable(getattr(left_chunks, "__iter__", None)) or not callable(getattr(right_chunks, "__iter__", None)):
        raise StreamingOperationError("chunk_sources_must_be_iterable")
    left_keys = _keys(left_on, "left_keys")
    right_keys = _keys(right_on if right_on is not None else left_on, "right_keys")
    if len(left_keys) != len(right_keys):
        raise StreamingOperationError("join_key_arity_mismatch")
    for name, value, upper in (("max_right_rows", max_right_rows, 5_000_000),
                               ("max_output_rows", max_output_rows, 10_000_000)):
        if type(value) is not int or not 1 <= value <= upper:
            raise StreamingOperationError(f"invalid_{name}")

    right_parts: list[pd.DataFrame] = []
    right_count = 0
    for chunk in right_chunks:
        if not isinstance(chunk, pd.DataFrame):
            raise StreamingOperationError("right_chunks_must_contain_dataframes")
        missing = [key for key in right_keys if key not in chunk.columns]
        if missing:
            raise StreamingOperationError("right_join_key_missing")
        if right_columns is not None:
            columns = _keys(right_columns, "right_columns")
            missing_columns = [key for key in columns if key not in chunk.columns]
            if missing_columns:
                raise StreamingOperationError("right_projection_column_missing")
            chunk = chunk[list(dict.fromkeys((*right_keys, *columns)))]
        right_count += len(chunk)
        if right_count > max_right_rows:
            raise StreamingOperationError("right_materialization_budget_exceeded")
        right_parts.append(chunk.copy(deep=False))
    if not right_parts:
        return pd.DataFrame()
    right = pd.concat(right_parts, ignore_index=True, copy=False)
    output_parts: list[pd.DataFrame] = []
    output_count = 0
    for chunk in left_chunks:
        if not isinstance(chunk, pd.DataFrame):
            raise StreamingOperationError("left_chunks_must_contain_dataframes")
        missing = [key for key in left_keys if key not in chunk.columns]
        if missing:
            raise StreamingOperationError("left_join_key_missing")
        joined = chunk.merge(right, how="inner", left_on=list(left_keys), right_on=list(right_keys),
                             suffixes=("", "_right"), sort=False)
        output_count += len(joined)
        if output_count > max_output_rows:
            raise StreamingOperationError("join_output_budget_exceeded")
        output_parts.append(joined)
    return pd.concat(output_parts, ignore_index=True, copy=False) if output_parts else pd.DataFrame()


def stream_numeric_group_stats(
    chunks: Iterable[pd.DataFrame], *, group_by: Sequence[str], value: str,
    max_groups: int = 500_000,
) -> dict[tuple[Any, ...], dict[str, float | int]]:
    """Compute count/mean/variance/sum per group without retaining rows."""
    keys = _keys(group_by, "group_by")
    if type(value) is not str or not value.strip() or value in keys:
        raise StreamingOperationError("invalid_value_column")
    if type(max_groups) is not int or not 1 <= max_groups <= 5_000_000:
        raise StreamingOperationError("invalid_max_groups")
    state: dict[tuple[Any, ...], list[float]] = {}
    for chunk in chunks:
        if not isinstance(chunk, pd.DataFrame):
            raise StreamingOperationError("chunks_must_contain_dataframes")
        missing = [name for name in (*keys, value) if name not in chunk.columns]
        if missing:
            raise StreamingOperationError("group_column_missing")
        numeric = pd.to_numeric(chunk[value], errors="coerce")
        for group, raw in zip(chunk[list(keys)].itertuples(index=False, name=None), numeric):
            if pd.isna(raw):
                continue
            key = tuple(group)
            try:
                hash(key)
            except TypeError as exc:
                raise StreamingOperationError("group_key_must_be_hashable") from exc
            if key not in state and len(state) >= max_groups:
                raise StreamingOperationError("group_cardinality_budget_exceeded")
            current = state.setdefault(key, [0.0, 0.0, 0.0])  # count, mean, M2
            count, mean, m2 = current
            count += 1
            delta = float(raw) - mean
            mean += delta / count
            m2 += delta * (float(raw) - mean)
            current[:] = [count, mean, m2]
    return {
        key: {"count": int(values[0]), "mean": values[1], "variance": values[2] / (values[0] - 1) if values[0] > 1 else 0.0,
              "sum": values[0] * values[1]}
        for key, values in state.items()
    }


__all__ = ["StreamingOperationError", "bounded_stream_hash_join", "stream_numeric_group_stats"]
