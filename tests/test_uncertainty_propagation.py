import pytest

from core.uncertainty_propagation import (
    UncertaintyPropagationError,
    propagate_layered_uncertainty,
)


def _row(semantic, structure, parameter, numerical, value, **extra):
    return {
        "semantic_id": semantic,
        "structure_id": structure,
        "parameter_id": parameter,
        "numerical_id": numerical,
        "value": value,
        "unit_signature": "m",
        **extra,
    }


def test_layered_variance_decomposes_without_double_counting():
    rows = [
        _row("s1", "m1", "p1", "n1", 1.0),
        _row("s1", "m1", "p1", "n2", 3.0),
        _row("s1", "m1", "p2", "n1", 5.0),
        _row("s1", "m1", "p2", "n2", 7.0),
        _row("s1", "m2", "p1", "n1", 9.0),
        _row("s1", "m2", "p1", "n2", 11.0),
        _row("s2", "m3", "p1", "n1", 13.0),
        _row("s2", "m3", "p1", "n2", 15.0),
    ]
    result = propagate_layered_uncertainty(rows)
    assert result["status"] == "ok"
    assert result["unit_signature"] == "m"
    assert all(result["variance_components"][name]["status"] == "assessed"
               for name in ("semantic", "structure", "parameter", "numerical"))
    assert result["reconstructed_variance"] == pytest.approx(result["output_summary"]["variance"])
    assert result["decomposition_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["output_summary"]["q50"] == pytest.approx(7.0)


def test_weights_change_mean_and_effective_sample_size():
    result = propagate_layered_uncertainty([
        _row("s", "m", "p", "n1", 0.0, weight=1.0),
        _row("s", "m", "p", "n2", 10.0, weight=3.0),
    ])
    assert result["output_summary"]["mean"] == pytest.approx(7.5)
    assert result["effective_sample_size"] == pytest.approx(16 / 10)
    assert result["variance_components"]["semantic"]["variance"] == pytest.approx(0.0)


def test_missing_deeper_layer_is_explicitly_partial():
    rows = [_row("s", "m", None, "n1", 1.0), _row("s", "m", None, "n2", 2.0)]
    result = propagate_layered_uncertainty(rows)
    assert result["status"] == "partial"
    assert result["variance_components"]["semantic"]["status"] == "assessed"
    assert result["variance_components"]["structure"]["status"] == "assessed"
    assert result["variance_components"]["parameter"]["status"] == "not_assessed"
    assert result["variance_components"]["numerical"]["status"] == "not_assessed"
    assert result["reconstructed_variance"] is None


def test_unit_mismatch_and_invalid_weights_are_rejected():
    with pytest.raises(UncertaintyPropagationError, match="unit_signatures_must_match"):
        propagate_layered_uncertainty([_row("s", "m", "p", "n1", 1.0),
                                       {**_row("s", "m", "p", "n2", 2.0), "unit_signature": "s"}])
    with pytest.raises(UncertaintyPropagationError, match="weight_must_be_positive"):
        propagate_layered_uncertainty([_row("s", "m", "p", "n1", 1.0, weight=0)])


def test_unit_is_required_before_any_variance_is_reported():
    row = _row("s", "m", "p", "n", 1.0)
    row.pop("unit_signature")
    result = propagate_layered_uncertainty([row])
    assert result["status"] == "not_assessed"
    assert result["reason"] == "unit_signature_required"


def test_invalid_record_budget_is_rejected_before_sequence_comparison():
    with pytest.raises(UncertaintyPropagationError, match="invalid_record_budget"):
        propagate_layered_uncertainty([], max_records="100")
