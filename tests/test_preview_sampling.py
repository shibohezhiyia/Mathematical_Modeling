import pytest

from core.preview_sampling import PreviewSamplingError, sample_preview_rows


def test_preview_sampling_preserves_rare_groups_and_high_anomalies():
    rows = [{"group": "common", "score": 0.0, "i": i} for i in range(20)]
    rows.extend([{"group": "rare", "score": 0.1, "i": 20}, {"group": "common", "score": 99.0, "i": 21}])
    result = sample_preview_rows(rows, max_rows=5, group_key="group", anomaly_key="score", rare_group_limit=1)
    assert len(result["rows"]) == 5
    assert "rare" in result["preserved_rare_groups"]
    assert any(row["i"] == 21 for row in result["rows"])


def test_preview_sampling_rejects_invalid_rows_and_is_deterministic():
    rows = [{"i": i} for i in range(10)]
    assert sample_preview_rows(rows, max_rows=3)["selected_indices"] == sample_preview_rows(rows, max_rows=3)["selected_indices"]
    with pytest.raises(PreviewSamplingError):
        sample_preview_rows(rows, max_rows=0)
