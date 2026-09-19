# 表格到数学图搜索

`core.data_graph_bridge` 把一个有界的数值表转换为 `mathmodel.graph-search-bundle/v1`，再交给受控图搜索 worker。它不是自动因果发现器，也不会把搜索候选当成最终答案。

## Python

```python
from pathlib import Path
import pandas as pd
from core.data_graph_bridge import run_scalar_graph_search

result, run_dir, audit = run_scalar_graph_search(
    pd.read_csv("data.csv"), "temperature", "output",
    output_root=Path("workspace/graph_search_runs"),
)
```

多特征交互使用 `build_tabular_graph_bundle(frame, ["x1", "x2", "x3"], "y")`；当前最多四列。图搜索会按每一步的显式量纲推导二元折叠：无量纲结构、同量纲的加/减/最小/最大，以及最终量纲恰好匹配目标的乘/除可以进入候选；无法证明中间量纲的结构仍会拒绝，避免猜测或静默丢列。

主研究入口在 `enable_graph_search=true` 且显式提供目标时，先建立互斥的特征选择、图搜索执行和最终留出行分区。选择分区计算有限值覆盖率、Spearman 关联和标准化成对交互的增量误差收益，以列名作单位尺度不敏感的平局键；方差只记录。最多四列和图搜索执行分区交给本桥，最终留出行不会传入。成对筛查最多覆盖按名称稳定排序的 32 列，审计会说明是否完整覆盖；更高阶交互不在当前保证内。选择审计保存在 `specialized_results.data_graph_search.feature_selection`，它只用于控制探索预算，不代表因果重要性。桥接器在执行分区内部再建立按特征元组互斥的训练/搜索分区。

## Web

`POST /api/research/data-graph-search` 的 JSON 至少包含 `rows`、`features`（一至四列）和 `target`。可选 `feature_dimensions` 与 `target_dimensions` 显式绑定量纲，例如：

```json
{
  "features": ["time"],
  "target": "distance",
  "feature_dimensions": {"time": {"T": 1}},
  "target_dimensions": {"distance": {"L": 1}},
  "rows": [{"time": 0, "distance": 0}, {"time": 1, "distance": 2}]
}
```

量纲未提供时返回的 `assumed_dimensionless_requires_confirmation` 只能用于探索；`causal_status=not_assessed` 也必须保留在报告中。运行产物按 `workspace/graph_search_runs/<run_id>/` 隔离，删除单次目录即可清理缓存。
