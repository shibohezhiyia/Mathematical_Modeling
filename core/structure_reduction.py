"""Conservative structural analysis for solver dimension reduction.

The analyzer proposes sparse, separable, block, and symmetry opportunities
without deleting constraints or weak couplings.  A caller must explicitly
accept a plan and compare it with a cold full problem; the returned metadata
is therefore a reduction *plan*, not an error-free approximation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


class StructureReductionError(ValueError):
    pass


def analyze_design_structure(
    matrix: Sequence[Sequence[float]],
    *,
    zero_tolerance: float = 1e-12,
    symmetry_tolerance: float = 1e-10,
    max_cells: int = 2_000_000,
) -> dict[str, Any]:
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)) or not matrix:
        raise StructureReductionError("matrix_must_be_nonempty")
    rows = [list(row) for row in matrix]
    if any(not row for row in rows):
        raise StructureReductionError("matrix_rows_must_be_nonempty")
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise StructureReductionError("matrix_must_be_rectangular")
    if len(rows) * columns > max_cells:
        raise StructureReductionError("matrix_exceeds_budget")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise StructureReductionError("matrix_values_must_be_finite_numeric")
    if isinstance(zero_tolerance, bool) or not isinstance(zero_tolerance, (int, float)) or float(zero_tolerance) < 0:
        raise StructureReductionError("invalid_zero_tolerance")
    if isinstance(symmetry_tolerance, bool) or not isinstance(symmetry_tolerance, (int, float)) or float(symmetry_tolerance) < 0:
        raise StructureReductionError("invalid_symmetry_tolerance")
    ztol, stol = float(zero_tolerance), float(symmetry_tolerance)
    total = len(rows) * columns
    zero_count = sum(abs(float(value)) <= ztol for row in rows for value in row)
    supports = [
        {index for index, row in enumerate(rows) if abs(float(row[column])) > ztol}
        for column in range(columns)
    ]
    symmetry_groups: list[list[int]] = []
    unassigned = set(range(columns))
    while unassigned:
        column = min(unassigned)
        group = [other for other in sorted(unassigned)
                 if all(abs(float(rows[row][column]) - float(rows[row][other])) <= stol
                        for row in range(len(rows)))]
        symmetry_groups.append(group)
        unassigned.difference_update(group)
    disjoint_pairs = sum(
        supports[left].isdisjoint(supports[right])
        for left in range(columns) for right in range(left + 1, columns)
    )
    sparse = zero_count / total >= 0.5
    separable = disjoint_pairs >= max(1, columns * (columns - 1) // 4)
    symmetric = any(len(group) > 1 for group in symmetry_groups)
    return {
        "schema_version": "mathmodel.structure-reduction/v1",
        "status": "assessed",
        "shape": [len(rows), columns],
        "zero_fraction": zero_count / total,
        "sparse_candidate": sparse,
        "separable_candidate": separable,
        "symmetry_groups": symmetry_groups,
        "symmetry_candidate": symmetric,
        "disjoint_support_pairs": disjoint_pairs,
        "plan": [
            item for item, enabled in (
                ("sparse_storage", sparse), ("blockwise_evaluation", separable),
                ("symmetry_reuse_after_certificate", symmetric),
            ) if enabled
        ],
        "policy": "plans_do_not_delete_constraints_or_weak_couplings;_full_cold_comparison_required",
    }


__all__ = ["StructureReductionError", "analyze_design_structure"]
