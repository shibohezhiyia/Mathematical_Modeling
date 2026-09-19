"""Prospective confirmation cases for the bounded discontinuity abstention gate."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .product_workflow_confirmation import ProductWorkflowCase


def build_product_discontinuity_confirmation(
    *, seed: int = 20261015,
) -> tuple[ProductWorkflowCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("product_discontinuity_seed_invalid")
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.6, 2.6, 96)
    specs: list[tuple[str, str, Callable[[np.ndarray], np.ndarray], float, str]] = [
        ("unseen-threshold-up", "discontinuity-threshold-up",
         lambda z: np.where(z >= -0.65, 3.2, -1.4), 0.4, "abstain"),
        ("unseen-threshold-down", "discontinuity-threshold-down",
         lambda z: np.where(z >= 0.55, -2.0, 4.0), 1.1, "abstain"),
        ("unseen-three-level", "discontinuity-three-level",
         lambda z: np.where(z < -0.7, -2.0, np.where(z < 0.8, 0.5, 3.0)), 1.2, "abstain"),
        ("unseen-window-pulse", "discontinuity-window-pulse",
         lambda z: np.where(np.abs(z - 0.2) < 0.7, 4.0, -1.0), 0.35, "abstain"),
        ("smooth-tanh-control", "smooth-tanh-control",
         lambda z: 1.1 + 2.3 * np.tanh(1.4 * z - 0.2), 1.15, "completed"),
        ("smooth-nested-control", "smooth-nested-control",
         lambda z: -0.2 + 1.5 * np.tanh(np.sin(0.9 * z + 0.1)), 1.3, "completed"),
        ("smooth-sine-control", "smooth-sine-control",
         lambda z: 0.4 + 1.8 * np.sin(1.2 * z - 0.3), 1.05, "completed"),
        ("smooth-rational-control", "smooth-rational-control",
         lambda z: (0.8 + 1.3 * z) / (1.0 + 0.15 * z), 1.25, "completed"),
    ]
    result = []
    for case_id, group, function, query, expected_status in specs:
        order = rng.permutation(len(x))
        y = function(x)
        input_name = f"driver_{case_id.replace('-', '_')}"
        target_name = f"metric_{case_id.replace('-', '_')}"
        records = tuple({input_name: float(x[index]), target_name: float(y[index])}
                        for index in order)
        result.append(ProductWorkflowCase(
            case_id=case_id, structure_group=group, dataset_name=f"raw_{case_id}",
            input_column=input_name, target_column=target_name,
            problem=f"根据原始观测建立关系，并预测 {input_name} = {query} 时的结果。",
            records=records, expected_status=expected_status,
            expected_prediction=(float(function(np.asarray([query]))[0])
                                 if expected_status == "completed" else None),
            tolerance=2e-3,
        ))
    return tuple(result)


__all__ = ["build_product_discontinuity_confirmation"]
