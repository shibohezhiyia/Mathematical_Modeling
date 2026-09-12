# 机制删减评估

`core.model_simplification.assess_simplifications` 对变量或机制逐个做消融比较。它要求后端评价器返回主目标损失、可行性，并可选返回压力域损失；只有删减后没有超过声明容差且仍可行时，才标记为 `tested_not_falsified`。

评估器异常、缺失指标和压力域恶化会保留为 `not_assessed` 或 `counterexample_found`。模块不会修改原模型、不固定删除比例，也不把“当前压力域没有反例”解释为全局误差界或现实正确性。
