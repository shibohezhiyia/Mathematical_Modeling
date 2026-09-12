# 四层不确定性传播

`core.uncertainty_propagation.propagate_layered_uncertainty` 对一组已经执行过的输出做有限的方差分解，层级固定为：

`semantic → structure → parameter → numerical replicate`

每条记录必须包含 `semantic_id`、`structure_id`、`parameter_id`、`numerical_id`、数值字段（默认是 `value`）和统一的 `unit_signature`。可以附带正的 `weight`，但系统不会把权重自动解释为后验概率。

```python
from core import propagate_layered_uncertainty

report = propagate_layered_uncertainty([
    {"semantic_id": "s1", "structure_id": "m1", "parameter_id": "p1",
     "numerical_id": "n1", "value": 2.1, "unit_signature": "m"},
    {"semantic_id": "s1", "structure_id": "m1", "parameter_id": "p1",
     "numerical_id": "n2", "value": 2.3, "unit_signature": "m"},
])
```

报告包含整体均值、方差、经验分位数，以及四个条件方差分量和重构残差。分量的含义是当前样本层级下的经验方差贡献，不是置信区间、后验方差或现实机制证明。

## 重要边界

- 没有统一单位时直接返回 `not_assessed`；不同单位不会被静默相加。
- 缺少某一层标识时返回 `partial`，该层及更深层显示 `not_assessed`，不会填零伪装成“没有不确定性”。
- 有限样本的分解只能说明这些候选和运行的差异来源，不能证明候选集合覆盖真实机制。
- 需要向量输出、相关误差、概率模型或校准集时，应先在外部定义联合观测模型，再把可比的标量决策量送入此接口。
