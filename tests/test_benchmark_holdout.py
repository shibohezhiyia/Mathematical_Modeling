import pytest

from core.automated_benchmark_suite import build_automated_benchmark_suite
from core.benchmark_holdout import BenchmarkHoldoutError, split_structure_holdout


def test_structure_holdout_keeps_groups_atomic_and_covers_all_cases():
    cases = build_automated_benchmark_suite(cases_per_family=2)
    split = split_structure_holdout(cases)
    assert split["case_counts"] == {"development": 4, "confirmation": 2, "final": 2}
    observed = [case.case_id for name in ("development", "confirmation", "final") for case in split[name]]
    assert set(observed) == {case.case_id for case in cases}
    for name, group_names in split["groups"].items():
        assert {case.structure_group for case in split[name]} == set(group_names)


def test_structure_holdout_rejects_overlap_or_unknown_groups():
    cases = build_automated_benchmark_suite(cases_per_family=1)
    with pytest.raises(BenchmarkHoldoutError, match="disjoint"):
        split_structure_holdout(cases, confirmation_groups=["algebra-linear"], final_groups=["algebra-linear"])
    with pytest.raises(BenchmarkHoldoutError, match="unknown"):
        split_structure_holdout(cases, confirmation_groups=["missing"], final_groups=["algebra-linear"])
