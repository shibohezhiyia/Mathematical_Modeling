# 模型集合评估

`assess_model_set` 用验证损失、复杂度、约束违反和不稳定性四个低值轴计算 Pareto 候选，并比较候选预测包络和决策值。它不会把排序分数归一化成后验概率，也不会因为某个模型分数最高而删除仍有证据支持的结构。

```python
from core.model_set_assessment import assess_model_set

result = assess_model_set([
    {
        "id": "mechanistic",
        "predictions": [1.0, 2.0],
        "decision": "accept",
        "metrics": {
            "validation_loss": 0.2,
            "complexity": 3,
            "constraint_violation": 0,
            "instability": 0.1,
        },
    }
])
```

`decision_consensus` 只说明当前 Pareto 候选在给定决策字段上相同；`prediction_envelope` 是候选包络，不是置信区间。参数和数值不确定性没有输入证据时会明确返回 `not_assessed`。
