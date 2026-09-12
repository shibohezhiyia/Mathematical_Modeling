import numpy as np
import pytest

from core.structure_diagnostics import StructureDiagnosticsError, diagnose_series_structure


def test_structure_diagnostics_detects_monotone_periodic_and_change_signals():
    times = np.arange(80, dtype=float)
    periodic = np.sin(2 * np.pi * times / 8)
    shifted = np.r_[np.zeros(40), np.full(40, 4.0)]
    result = diagnose_series_structure(times, np.column_stack((times, periodic, shifted)), ["trend", "cycle", "shift"])

    assert result["diagnostics"]["trend"]["monotone"]["direction"] == "increasing"
    assert "monotone_candidate" in result["diagnostics"]["trend"]["signals"]
    assert "periodic_candidate" in result["diagnostics"]["cycle"]["signals"]
    assert "change_point_candidate" in result["diagnostics"]["shift"]["signals"]
    assert result["policy"] == "signals_generate_search_hints_only"


def test_structure_diagnostics_is_deterministic_and_accepts_one_series():
    times = np.linspace(0, 1, 12)
    result_a = diagnose_series_structure(times, np.exp(times))
    result_b = diagnose_series_structure(times, np.exp(times))
    assert result_a["trajectory_sha256"] == result_b["trajectory_sha256"]
    assert result_a["series_count"] == 1


def test_structure_diagnostics_rejects_bad_contract():
    with pytest.raises(StructureDiagnosticsError, match="strictly_increasing"):
        diagnose_series_structure([0, 1, 1, 2, 3, 4, 5, 6], np.ones((8, 1)))
    with pytest.raises(StructureDiagnosticsError, match="lag_budget"):
        diagnose_series_structure(np.arange(8), np.ones(8), max_lag=0)
