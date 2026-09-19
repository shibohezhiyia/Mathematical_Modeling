"""Fresh raw-table tasks for a predeclared sqrt-absolute second-arm rescue check.

The topology was selected on a separate development probe. This suite tests
new numeric domains and records, not discovery of an unknown structure.
"""
from __future__ import annotations

import numpy as np

from .product_workflow_confirmation import ProductWorkflowCase


def build_product_rescue_confirmation(*, seed: int = 20261018) -> tuple[ProductWorkflowCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("product_rescue_seed_invalid")
    rng = np.random.default_rng(seed)
    cases = []
    for case_id, width, count, query in (
        ("sqrt-abs-wide", 5.2, 128, 0.42),
        ("sqrt-abs-irregular", 3.7, 144, -0.51),
    ):
        x = (np.linspace(-width, width, count) if case_id.endswith("wide")
             else rng.uniform(-width, width, count))
        order = rng.permutation(len(x))
        cases.append(ProductWorkflowCase(
            case_id=case_id, structure_group="sqrt-absolute-transition",
            dataset_name=f"raw_{case_id}", input_column="actuator", target_column="sensor_value",
            problem=f"根据原始观测建立关系；预测 actuator = {query} 时的 sensor_value。",
            records=tuple({"actuator": float(x[index]),
                           "sensor_value": float(np.sqrt(abs(x[index])))} for index in order),
            expected_status="completed", expected_prediction=float(np.sqrt(abs(query))),
            tolerance=0.03,
        ))
    for case_id, slope, intercept in (
        ("linear-a", 1.2, 0.3), ("linear-b", -0.9, 0.4),
    ):
        x = rng.uniform(-2.0, 2.0, 96)
        query = 0.65
        cases.append(ProductWorkflowCase(
            case_id=case_id, structure_group="affine-control",
            dataset_name=f"raw_{case_id}", input_column="actuator", target_column="sensor_value",
            problem=f"根据原始观测建立关系；预测 actuator = {query} 时的 sensor_value。",
            records=tuple({"actuator": float(value),
                           "sensor_value": float(slope * value + intercept)} for value in x),
            expected_status="completed", expected_prediction=slope * query + intercept,
            tolerance=1e-6,
        ))
    return tuple(cases)


__all__ = ["build_product_rescue_confirmation"]
