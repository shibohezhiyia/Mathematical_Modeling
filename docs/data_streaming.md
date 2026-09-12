# 大数据流式读取

## 什么时候使用

`DataLoader.load()` 和 `load_chunked()` 仍然返回完整 `DataFrame`，适合后续算法确实需要全量数据的任务。超大 CSV、TSV、TXT 或 Parquet 的探索、画像和质量预检应使用惰性接口，避免先把所有块拼接到内存。

```python
from core.data_module import DataLoader, DataModule

loader = DataLoader()
for chunk in loader.iter_chunks("data.csv", chunk_size=50_000,
                                columns=["time", "value"]):
    # 在这里完成单块计算；不要把 chunk 追加到列表后再 concat
    process(chunk)

# 也可以用旧的 load 入口显式要求惰性返回；materialize=True 仍是默认行为。
for chunk in loader.load("data.csv", materialize=False, chunk_size=50_000,
                         columns=["time", "value"]):
    process(chunk)

profile = DataModule().stream_profile(
    "data.parquet", chunk_size=50_000, columns=["time", "value"]
)

for chunk in DataModule().stream_chunks("data.parquet", chunk_size=50_000,
                                        columns=["time", "value"]):
    process(chunk)

reduced = DataModule().stream_reduce(
    "data.parquet",
    lambda state, chunk: state + float(chunk["value"].sum()),
    0.0,
    chunk_size=50_000,
    max_rows=2_000_000,
)

from core.data_quality import generate_streaming_data_quality_report
quality = generate_streaming_data_quality_report(
    "data.csv", chunk_size=50_000, target_col="value"
)
```

## 统计边界

画像中的行数、缺失数和数值列 `count/mean/std/min/max` 是跨块合并得到的精确统计。唯一值默认使用有界集合：`unique_exact=true` 时是精确值；超出 `max_unique_per_column` 后只返回 `unique_lower_bound`，不会假装是精确基数。

分位数、相关系数、全局重复行数以及依赖全局排序的算法没有被隐式近似，结果会在 `requires_full_scan` 中列出。需要这些统计时，应显式选择外存算法、近似算法并保存误差界，或确认有足够内存后调用完整加载接口。

目前流式接口支持 CSV/TSV/TXT、XLSX 和 Parquet；旧式 XLS 仍需先转换为 XLSX 或 Parquet。XLSX 使用 `openpyxl` 的只读工作簿，并可通过 `sheet_name` 和 `columns` 限定工作表与列。

注意：`load_chunked()` 的默认 `materialize=True` 为兼容旧代码仍会合并 DataFrame；处理大数据时必须显式传 `materialize=False` 或使用 `iter_chunks()`，否则最终内存峰值仍可能接近全量数据。

`stream_reduce()` 的返回值包含 `state`、实际处理行数、块数、耗时和状态。触发行数或时间上限时状态分别为 `row_budget_exhausted` / `time_budget_exhausted`，不能当作完整扫描。
