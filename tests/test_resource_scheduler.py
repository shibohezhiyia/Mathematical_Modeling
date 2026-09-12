import pytest

from core.resource_scheduler import ResourceBudget, ResourceSchedulerError, admit_workload, estimate_workload


def test_pairwise_work_is_rejected_before_allocation():
    workload = estimate_workload(rows=10_000, columns=8, pairwise=True)
    decision = admit_workload(workload, ResourceBudget(max_pairwise_cells=1_000_000))
    assert decision["status"] == "rejected_resource_budget"
    assert "pairwise_cells" in decision["failed_checks"]


def test_admission_is_explicit_and_does_not_sample():
    workload = estimate_workload(rows=100, columns=3, candidates=2)
    decision = admit_workload(workload)
    assert decision["status"] == "admitted"
    assert decision["workload"]["rows"] == 100


def test_resource_scheduler_rejects_coercion_and_negative_estimates():
    with pytest.raises(ResourceSchedulerError, match="pairwise_must_be_bool"):
        estimate_workload(rows=10, columns=2, pairwise="yes")
    with pytest.raises(ResourceSchedulerError, match="workload_estimate_values"):
        admit_workload({"rows": -1, "columns": 2, "pairwise_cells": 0,
                        "estimated_memory_mb": 1, "candidates": 1})
