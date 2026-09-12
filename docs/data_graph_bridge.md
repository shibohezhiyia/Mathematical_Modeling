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
