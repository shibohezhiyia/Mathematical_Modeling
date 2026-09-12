import pandas as pd
import pytest

from core.modeling_assistant import MathModelingAssistant


def test_large_materialized_input_is_bounded_and_keeps_source_provenance(tmp_path):
    frame = pd.DataFrame({"time": range(2501), "value": [i * 2 for i in range(2501)]})
    first = MathModelingAssistant(output_dir=str(tmp_path / "a"), max_input_rows=1000)
    bounded = first._validate_datasets({"series": frame})["series"]
    assert len(bounded) == 1000
    assert bounded.attrs["source_rows"] == 2501
    assert bounded.attrs["sampling_complete"] is False
    assert bounded.attrs["sampling_policy"] == "deterministic_even_row_span"
    assert any("超过输入上限" in warning for warning in first._input_warnings)

    second = MathModelingAssistant(output_dir=str(tmp_path / "b"), max_input_rows=1000)
    repeated = second._validate_datasets({"series": frame})["series"]
    assert bounded["time"].tolist() == repeated["time"].tolist()


def test_input_budget_parameters_are_rejected_outside_safe_bounds(tmp_path):
    with pytest.raises(ValueError, match="max_input_rows"):
        MathModelingAssistant(output_dir=str(tmp_path), max_input_rows=999)
    with pytest.raises(ValueError, match="max_input_memory_mb"):
        MathModelingAssistant(output_dir=str(tmp_path), max_input_memory_mb=32)
