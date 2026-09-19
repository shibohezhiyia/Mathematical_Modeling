"""Prospective product tasks for transition uncertainty and on-demand routing."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .product_workflow_confirmation import ProductWorkflowCase


def build_product_routing_confirmation(*, seed: int = 20261016) -> tuple[ProductWorkflowCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("product_routing_seed_invalid")
    rng = np.random.default_rng(seed)
    specifications: list[tuple[str, str, np.ndarray, Callable[[np.ndarray], np.ndarray], float, str]] = [
        ("wide-saturation", "smooth-transition-wide", np.linspace(-16.0, 16.0, 96),
         lambda z: np.tanh(4.4 * z), 0.1, "needs_input"),
        ("shifted-saturation", "smooth-transition-shifted", np.linspace(-12.0, 12.0, 96),
         lambda z: 0.5 + 1.4 * np.tanh(4.8 * (z - 0.35)), 0.35, "needs_input"),
        ("quantized-saturation", "quantized-transition", np.linspace(-8.0, 8.0, 96),
         lambda z: np.round(1.7 * np.tanh(1.3 * z), 1), 0.2, "needs_input"),
        ("noisy-threshold", "noisy-transition", np.linspace(-4.0, 4.0, 96),
         lambda z: np.where(z >= 0.4, 2.5, -0.7)
         + rng.normal(0.0, 0.008, len(z)), 0.7, "needs_input"),
        ("affine-control", "smooth-affine", np.linspace(-3.0, 3.0, 96),
         lambda z: 1.3 * z + 0.7, 1.25, "completed"),
        ("sine-control", "smooth-sine-new", np.linspace(-2.0, 2.0, 96),
         lambda z: 0.4 + 1.7 * np.sin(0.9 * z + 0.3), 1.15, "completed"),
        ("rational-control", "smooth-rational-new", np.linspace(-2.0, 2.0, 96),
         lambda z: (1.2 + 1.1 * z) / (1.0 + 0.17 * z), 1.1, "completed"),
        ("nested-control", "smooth-nested-new", np.linspace(-2.0, 2.0, 96),
         lambda z: -0.1 + 1.2 * np.tanh(np.sin(0.8 * z - 0.15)), 1.35, "completed"),
    ]
    cases = []
    for case_id, group, x, function, query, expected_status in specifications:
        y = function(x)
        order = rng.permutation(len(x))
        input_name = f"driver_{case_id.replace('-', '_')}"
        target_name = f"metric_{case_id.replace('-', '_')}"
        cases.append(ProductWorkflowCase(
            case_id=case_id, structure_group=group, dataset_name=f"raw_{case_id}",
            input_column=input_name, target_column=target_name,
            problem=f"根据原始记录建模，并预测 {input_name} = {query} 时的结果。",
            records=tuple({input_name: float(x[index]), target_name: float(y[index])}
                          for index in order),
            expected_status=expected_status,
            expected_prediction=(float(function(np.asarray([query]))[0])
                                 if expected_status == "completed" else None),
            tolerance=2e-3,
        ))
    return tuple(cases)


__all__ = ["build_product_routing_confirmation"]
