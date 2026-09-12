import pytest

from core.uncertainty_audit import UncertaintyAuditError, build_uncertainty_audit


def _row(index, value):
    return {"semantic_id": "s", "structure_id": "m", "parameter_id": f"p{index % 2}",
            "numerical_id": f"n{index}", "value": value, "unit_signature": "m"}


def test_uncertainty_audit_keeps_layers_and_independent_interval_calibration_separate():
    rows = [_row(i, float(i)) for i in range(10)]
    result = build_uncertainty_audit(rows, interval={
        "actual": list(range(10)), "lower": [value - 0.5 for value in range(10)],
        "upper": [value + 0.5 for value in range(10)],
    })
    assert result["status"] == "assessed"
    assert result["propagation"]["status"] == "ok"
    assert result["calibration"]["sample_count"] == 10


def test_uncertainty_audit_does_not_infer_calibration_or_accept_two_kinds():
    result = build_uncertainty_audit([_row(0, 1.0), _row(1, 2.0)])
    assert result["status"] == "partial"
    assert result["calibration"]["status"] == "not_assessed"
    with pytest.raises(UncertaintyAuditError, match="choose_one_calibration_kind"):
        build_uncertainty_audit([_row(0, 1.0)], interval={"actual": [1], "lower": [0], "upper": [2]},
                                probability={"outcomes": [1], "probabilities": [0.5]})
