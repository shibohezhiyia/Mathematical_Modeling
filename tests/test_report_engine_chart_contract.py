import pandas as pd

from extensions.report_engine import ChartBuilder, ChartConfig


def test_exploration_count_contract_is_replayed_in_report():
    frame = pd.DataFrame(
        {
            "类别": ["甲", "甲", "乙", "乙"],
            "数值": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = ChartConfig(
        chart_type="bar",
        x_field="类别",
        y_field="__count__",
        agg="count",
        filters=[{"field": "数值", "kind": "range", "min": 2, "max": 4}],
    )

    prepared = ChartBuilder.prepare_frame(frame, config)
    aggregated = ChartBuilder.aggregate_frame(prepared, config)

    assert aggregated["__count__"].to_dict() == {"甲": 1, "乙": 2}


def test_exploration_time_grain_is_replayed_in_report():
    frame = pd.DataFrame(
        {
            "日期": ["2026-01-01", "2026-01-15", "2026-02-01"],
            "数值": [1.0, 3.0, 8.0],
        }
    )
    config = ChartConfig(
        chart_type="line",
        x_field="日期",
        y_field="数值",
        agg="mean",
        time_unit="month",
    )

    prepared = ChartBuilder.prepare_frame(frame, config)
    aggregated = ChartBuilder.aggregate_frame(prepared, config)

    assert list(aggregated.index.astype(str)) == ["2026-01-01", "2026-02-01"]
    assert aggregated["数值"].tolist() == [2.0, 8.0]
