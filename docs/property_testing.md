# 有界性质检查

`core.property_testing.check_scalar_property` 为通用数学原语提供有限探针检查，支持：

- 输出上下界；
- 单变量非减/非增；
- 关于零点的偶对称/奇对称。

调用方提供已经受控的标量 `evaluate(point)`，检查器只负责生成确定性探针、限制评估预算和保存最小反例。返回状态为 `pass`、`fail` 或 `not_assessed`；通过时 `proof_status` 是 `tested_not_falsified`，绝不表示连续域上的数学证明。

```python
from core.property_testing import PropertyTestBudget, check_scalar_property

result = check_scalar_property(
    lambda point: point["x"] ** 2,
    domain={"x": [-2, 2]},
    kind="bounded", lower=0, upper=4,
    budget=PropertyTestBudget(probe_count=32, seed=42),
)
```

多变量单调性必须明确传入 `variable`；单变量场景可以省略。评估器异常、非有限输出或预算不足会返回 `not_assessed`，不会被折算为通过。该内核目前是独立的可复用证据组件，图搜索和各专用求解器接入时仍需保留自己的定义域、单位与约束语义。
