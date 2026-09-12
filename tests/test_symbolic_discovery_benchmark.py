from core.sindy_discovery import discover_sparse_dynamics
from core.symbolic_discovery_benchmark import build_unseen_equation_cases, evaluate_unseen_equation_benchmark


def test_unseen_equation_cases_hide_support_labels_from_public_case_rows():
    cases = build_unseen_equation_cases(sample_count=64)
    assert len(cases) == 3
    assert all(item.public()["expected_support_hidden"] for item in cases)
    report = evaluate_unseen_equation_benchmark(discover_sparse_dynamics, cases=cases)
    assert report["case_count"] == 3
    assert report["support_recovery_rate"] >= 2 / 3
    assert all("expected" not in row for row in report["rows"])


def test_unseen_benchmark_can_compare_paired_fit_only_split_without_leaking_labels():
    report = evaluate_unseen_equation_benchmark(
        discover_sparse_dynamics, cases=build_unseen_equation_cases(sample_count=64),
        fit_only_discover=discover_sparse_dynamics,
    )
    baseline = report["fit_only_baseline"]
    assert baseline["case_count"] == report["case_count"] == 3
    assert "expected_support" not in str(baseline)
    assert "paired_validation_split" in baseline["policy"]
