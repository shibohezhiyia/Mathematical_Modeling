import pytest

from core.probability_variance import ProbabilityVarianceError, propagate_probability_variance


def _row(s, m, p, n, value, probability):
    return {"semantic_id": s, "structure_id": m, "parameter_id": p,
            "numerical_id": n, "value": value, "probability": probability,
            "unit_signature": "m"}


def test_probability_variance_requires_mass_and_reports_total_variance_ledger():
    report = propagate_probability_variance([
        _row("s1", "m1", "p1", "n1", 1, .25),
        _row("s1", "m1", "p1", "n2", 3, .25),
        _row("s2", "m2", "p2", "n1", 5, .25),
        _row("s2", "m2", "p2", "n2", 7, .25),
    ])
    assert report["law_total_variance"] is True
    assert report["probability_mass"] == pytest.approx(1)
    assert report["decomposition_residual"] == pytest.approx(0)
    assert "not_automatic_posterior" in report["policy"]


def test_probability_variance_rejects_non_normalized_mass():
    with pytest.raises(ProbabilityVarianceError, match="sum_to_one"):
        propagate_probability_variance([_row("s", "m", "p", "n", 1, .5)])
