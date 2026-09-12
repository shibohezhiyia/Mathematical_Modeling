import pytest

from core.reversible_scaling import ReversibleScalingError, fit_reversible_scaler


def test_reversible_scaler_roundtrips_and_keeps_units():
    scaler = fit_reversible_scaler([[1000, 2], [2000, 4], [3000, 6]], unit_signature=["m", "s"])
    scaled = scaler.transform([[1500, 3]])
    assert scaled[0][0] == pytest.approx(-0.6123724357)
    assert scaler.audit_roundtrip([[1000, 2], [3000, 6]])["status"] == "pass"
    assert scaler.unit_signature == ("m", "s")


def test_constant_and_minmax_columns_are_bounded_and_reversible():
    scaler = fit_reversible_scaler([[1, 5], [1, 10]], unit_signature="u", method="minmax")
    assert scaler.constant_columns == (0,)
    recovered = scaler.inverse_transform(scaler.transform([[1, 7.5]]))
    assert recovered[0] == pytest.approx([1, 7.5])


@pytest.mark.parametrize("values, units, method", [
    ([[1, 2]], ["m"], "standard"), ([[float("nan")]], "m", "standard"),
    ([[1]], "m", "unknown"),
])
def test_scaler_rejects_invalid_contract(values, units, method):
    with pytest.raises(ReversibleScalingError):
        fit_reversible_scaler(values, unit_signature=units, method=method)
