"""Exact dimensional-analysis primitives for unit-aware model search."""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.buckingham-pi/v1"
BASE_DIMENSIONS = ("M", "L", "T", "I", "Theta", "N", "J")


class DimensionalAnalysisError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fraction(value: Any) -> Fraction:
    if type(value) is bool:
        raise DimensionalAnalysisError("dimension_exponent_must_be_numeric")
    try:
        result = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise DimensionalAnalysisError("invalid_dimension_exponent") from exc
    if abs(result) > 32 or result.denominator > 16:
        raise DimensionalAnalysisError("dimension_exponent_limit")
    return result


def _format(value: Fraction) -> int | str:
    return int(value) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _nullspace(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """Exact RREF nullspace of a rows-by-columns matrix."""
    rows, columns = len(matrix), len(matrix[0]) if matrix else 0
    work = [row[:] for row in matrix]
    pivot_columns, pivot_row = [], 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(rows):
            if row != pivot_row and work[row][column]:
                factor = work[row][column]
                work[row] = [left - factor * right for left, right in zip(work[row], work[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    basis = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(columns)]
        vector[free] = Fraction(1)
        for row, pivot in reversed(list(enumerate(pivot_columns))):
            vector[pivot] = -sum(work[row][column] * vector[column] for column in free_columns)
        basis.append(vector)
    return basis


def buckingham_pi_groups(dimensions: Mapping[str, Any], *, base_dimensions: tuple[str, ...] = BASE_DIMENSIONS) -> dict[str, Any]:
    """Generate exact null-space dimensionless groups.

    ``dimensions`` maps quantity names to either a mapping of base-dimension
    exponents or a sequence in ``base_dimensions`` order.  The result is a
    candidate basis, not a claim that every group is physically useful.
    """
    if not isinstance(dimensions, Mapping) or not 1 <= len(dimensions) <= 128:
        raise DimensionalAnalysisError("quantity_count_must_be_between_1_and_128")
    if (not isinstance(base_dimensions, tuple) or not 1 <= len(base_dimensions) <= 16 or
            any(type(item) is not str or not item.strip() or len(item) > 32 for item in base_dimensions) or
            len(set(base_dimensions)) != len(base_dimensions)):
        raise DimensionalAnalysisError("invalid_base_dimensions")
    names = list(dimensions)
    if (any(type(name) is not str or not name.strip() or len(name) > 128 for name in names) or
            len(set(names)) != len(names)):
        raise DimensionalAnalysisError("invalid_quantity_names")
    matrix = []
    for name in names:
        raw = dimensions[name]
        if isinstance(raw, Mapping):
            if set(raw) - set(base_dimensions):
                raise DimensionalAnalysisError("unknown_base_dimension")
            vector = [_fraction(raw.get(key, 0)) for key in base_dimensions]
        elif isinstance(raw, (list, tuple)) and len(raw) == len(base_dimensions):
            vector = [_fraction(value) for value in raw]
        else:
            raise DimensionalAnalysisError(f"invalid_dimensions:{name}")
        matrix.append(vector)
    # Columns are quantities, rows are base dimensions.
    transposed = [[matrix[column][row] for column in range(len(names))]
                  for row in range(len(base_dimensions))]
    basis = _nullspace(transposed)
    groups = []
    for index, vector in enumerate(basis, 1):
        exponents = {name: _format(value) for name, value in zip(names, vector) if value}
        groups.append({"id": f"pi_{index}", "exponents": exponents,
                       "expression": " ".join(f"{name}^{_format(value)}" for name, value in exponents.items()),
                       "dimensionless": True, "status": "candidate"})
    rank = len(names) - len(basis)
    public_matrix = [[_format(value) for value in row] for row in matrix]
    digest = sha256(_canonical({"base_dimensions": list(base_dimensions), "quantities": names,
                                "matrix": public_matrix}).encode("utf-8")).hexdigest()
    return {"schema_version": SCHEMA_VERSION, "base_dimensions": list(base_dimensions),
            "quantities": names, "rank": rank, "group_count": len(groups),
            "groups": groups, "dimension_matrix": public_matrix,
            "dimension_matrix_sha256": digest,
            "policy": "exact_nullspace_candidate_not_physical_validation"}


__all__ = ["BASE_DIMENSIONS", "SCHEMA_VERSION", "DimensionalAnalysisError", "buckingham_pi_groups"]
