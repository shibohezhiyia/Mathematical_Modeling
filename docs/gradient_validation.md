# 梯度交叉校验

`check_gradient` 在自动微分、代理模型或用户提供梯度进入优化前，用有限差分独立复核每个坐标。内部点使用中心差分，边界点自动使用可行的前向/后向差分；所有函数调用计入预算。

```python
from core.gradient_validation import check_gradient

result = check_gradient(
    lambda p: p["x"] ** 2,
    lambda p: {"x": 2 * p["x"]},
    {"x": 1.0}, bounds={"x": [0.0, 2.0]},
)
```

`pass` 只表示当前点和容差内的交叉检查通过；`fail` 表示发现梯度不一致；`not_assessed` 表示预算、边界或函数值问题阻止检查。它不证明全局可微、凸性或最优性。
