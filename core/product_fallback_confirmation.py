"""Prospective raw-table tasks testing whether a second solver rescues failures.

The four-variable product-sine topology appeared in prior SRSD work. These
cases are fresh observations/parameter settings, not unseen topologies.
"""
from __future__ import annotations

import numpy as np

from .product_workflow_confirmation import ProductWorkflowCase


def build_product_fallback_confirmation(*, seed: int = 20261017) -> tuple[ProductWorkflowCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("product_fallback_seed_invalid")
    rng = np.random.default_rng(seed)
    cases = []
    for case_id, factor, angle_scale in (
        ("product-sine-a", 1.0, 1.0),
        ("product-sine-b", 0.8, 1.25),
    ):
        x = rng.uniform([0.25, 0.35, 0.45, -1.2], [1.8, 1.6, 1.75, 1.2], size=(256, 4))
        columns = ("flow_in", "pressure", "exposure", "phase")
        values = factor * x[:, 0] * x[:, 1] * x[:, 2] * np.sin(angle_scale * x[:, 3])
        query = (1.1, 0.9, 1.25, 0.55)
        expected = factor * query[0] * query[1] * query[2] * np.sin(angle_scale * query[3])
        records = tuple({**{name: float(value) for name, value in zip(columns, row)},
                         "signal": float(response)} for row, response in zip(x, values))
        assignments = "、".join(f"{name} = {value}" for name, value in zip(columns, query))
        cases.append(ProductWorkflowCase(
            case_id=case_id, structure_group="four-way-product-sine",
            dataset_name=f"raw_{case_id}", input_column=columns[0], target_column="signal",
            problem=f"由原始观测建模；预测 {assignments} 时的 signal。",
            records=records, expected_status="completed", expected_prediction=float(expected),
            tolerance=0.04,
        ))
    for case_id, slope, intercept in (
        ("affine-a", 1.4, -0.2), ("affine-b", -0.75, 0.8),
    ):
        x = rng.uniform(-2.5, 2.5, size=96)
        query = 1.15
        cases.append(ProductWorkflowCase(
            case_id=case_id, structure_group="affine-control",
            dataset_name=f"raw_{case_id}", input_column="driver", target_column="output",
            problem=f"由原始观测建模；预测 driver = {query} 时的 output。",
            records=tuple({"driver": float(value), "output": float(slope * value + intercept)}
                          for value in x),
            expected_status="completed", expected_prediction=slope * query + intercept,
            tolerance=1e-6,
        ))
    return tuple(cases)


__all__ = ["build_product_fallback_confirmation"]
