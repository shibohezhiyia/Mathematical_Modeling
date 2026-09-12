from core.synthetic_bench_cases import build_diagnostic_case_catalog


def test_diagnostic_catalog_covers_failure_attribution_fixtures():
    cases = build_diagnostic_case_catalog()
    assert len(cases) == 7
    expected = {"hidden_state", "noise_present", "observation_bias", "regime_switch",
                "nonidentifiable", "contradictory_constraints", "data_absent"}
    assert {next(iter(case.expected)) for case in cases} == expected
