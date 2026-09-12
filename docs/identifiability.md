# 局部可辨识性检查

`assess_local_identifiability` 对参数化预测函数做有限差分，构造响应 Jacobian，报告数值秩、奇异值、Fisher 信息矩阵的特征值和条件数。边界参数使用单侧差分，并且所有预测调用都计入预算。

```python
from core.identifiability import assess_local_identifiability

result = assess_local_identifiability(
    predict, {"decay": 0.2, "gain": 1.0},
    bounds={"decay": [0, 5], "gain": [0, 10]},
)
```

`locally_identified` 只表示当前参数点附近的响应 Jacobian 数值满秩；它不证明全局唯一、因果关系或参数在噪声下可稳定估计。`weakly_identified` 和条件数应进入候选模型比较与报告，而不是被隐藏在单一拟合分数中。
