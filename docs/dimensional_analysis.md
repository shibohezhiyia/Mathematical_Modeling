# Buckingham π 无量纲候选

`buckingham_pi_groups` 对量纲矩阵做精确有理数零空间计算，输出无量纲候选群及矩阵指纹。它不依赖浮点 SVD，因此不会因接近零的数值阈值改变群的维数。

```python
from core.dimensional_analysis import buckingham_pi_groups

result = buckingham_pi_groups({
    "length": {"L": 1},
    "time": {"T": 1},
    "speed": {"L": 1, "T": -1},
})
```

返回的群标记为 `candidate`。无量纲只是代数必要条件，不证明变量之间存在该关系，也不代表选择了正确的特征尺度；后续仍需由题意、数据、边界和反例检验筛选。该模块可作为符号回归候选库的前置约束。
## 单位换算

除表达式量纲检查外，`core.mathematical_reasoning` 现在提供受限的线性单位换算：

```python
from core.mathematical_reasoning import convert_values

meters = convert_values([1, 2], "km", "m")
```

换算要求源单位和目标单位量纲兼容、输入有限且不超过规模上限。摄氏/华氏等需要偏移量的仿射温度转换会被拒绝，不会被错误地当成乘法比例；单位字符串替换也不会自动修改数据数值。
