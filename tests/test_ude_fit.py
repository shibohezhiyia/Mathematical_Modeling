import numpy as np
import pytest

from core.structure_extensions import StructureExtensionError, fit_ude_correction


def test_ude_correction_fits_known_residual_and_keeps_holdout_audit():
    x = np.linspace(-2, 2, 20)
    features = np.column_stack([np.ones_like(x), x])
    result = fit_ude_correction(2.0 * x + 0.5, 1.5 * x, features)
    assert result["status"] == "fitted"
    assert result["holdout_rows"] > 0
    assert result["holdout_rmse"] < 0.01
    assert "not_neural" in result["policy"]


def test_ude_correction_rejects_nonfinite_or_tiny_inputs():
    with pytest.raises(StructureExtensionError, match="ude_fit_size_invalid"):
        fit_ude_correction([1, 2], [1, 2], [[1], [1]])
    values = np.ones(8)
    values[0] = np.nan
    with pytest.raises(StructureExtensionError, match="ude_fit_values_must_be_finite"):
        fit_ude_correction(values, np.ones(8), np.ones((8, 1)))
