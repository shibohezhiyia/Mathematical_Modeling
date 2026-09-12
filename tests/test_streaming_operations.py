import pandas as pd
import pytest

from core.streaming_operations import (
    StreamingOperationError, bounded_stream_hash_join, stream_numeric_group_stats,
)


def test_bounded_stream_hash_join_preserves_many_to_many_rows():
    left = [pd.DataFrame({"id": [1, 2], "x": [10, 20]}), pd.DataFrame({"id": [1], "x": [30]})]
    right = [pd.DataFrame({"key": [1, 1, 3], "v": [2, 4, 9]})]
    result = bounded_stream_hash_join(left, right, left_on=["id"], right_on=["key"])
    assert len(result) == 4
    assert sorted(result["v"].tolist()) == [2, 2, 4, 4]


def test_stream_join_rejects_over_budget_instead_of_sampling():
    with pytest.raises(StreamingOperationError, match="materialization"):
        bounded_stream_hash_join(
            [pd.DataFrame({"id": [1]})],
            [pd.DataFrame({"id": [1, 2]})],
            left_on=["id"], max_right_rows=1,
        )
    with pytest.raises(StreamingOperationError, match="output"):
        bounded_stream_hash_join(
            [pd.DataFrame({"id": [1, 1]})],
            [pd.DataFrame({"id": [1, 1]})],
            left_on=["id"], max_output_rows=2,
        )


def test_stream_numeric_group_stats_is_chunk_order_invariant_and_bounded():
    chunks = [pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}),
              pd.DataFrame({"g": ["a", "b"], "v": [3, 4]})]
    result = stream_numeric_group_stats(chunks, group_by=["g"], value="v")
    assert result[("a",)]["mean"] == 2.0
    assert result[("a",)]["variance"] == 2.0
    with pytest.raises(StreamingOperationError, match="cardinality"):
        stream_numeric_group_stats([pd.DataFrame({"g": [1, 2], "v": [1, 2]})],
                                   group_by=["g"], value="v", max_groups=1)
