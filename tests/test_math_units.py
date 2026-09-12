import numpy as np
import pytest

from core.mathematical_reasoning import (
    UnitConversionError,
    convert_values,
    unit_conversion_factor,
)


def test_linear_unit_conversion_is_dimension_checked_and_non_mutating():
    values = np.array([1.0, 2.0])
    converted = convert_values(values, "km", "m")
    assert np.allclose(converted, [1000.0, 2000.0])
    assert np.array_equal(values, [1.0, 2.0])
    assert unit_conversion_factor("h", "s") == 3600.0


def test_unit_conversion_rejects_incompatible_affine_and_nonfinite_inputs():
    with pytest.raises(UnitConversionError, match="incompatible_units"):
        unit_conversion_factor("m", "s")
    with pytest.raises(UnitConversionError, match="affine_temperature_conversion_not_supported"):
        unit_conversion_factor("°C", "K")
    with pytest.raises(UnitConversionError, match="values_must_be_finite"):
        convert_values([float("nan")], "m", "cm")
