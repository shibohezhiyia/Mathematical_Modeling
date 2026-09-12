# 守恒律候选发现

`discover_linear_conservation` 从多变量时间序列的有限差分导数矩阵中寻找近似线性不变量。它先在训练时间段计算导数矩阵的零空间，再用末段数据验证候选，返回系数、残差、奇异值谱和轨迹指纹。

```python
from core.conservation_discovery import discover_linear_conservation

result = discover_linear_conservation(times, states, ["x", "y"])
```

输出中的 `candidate` 只表示在有限差分、有限样本和当前容差下未被反驳；`proof_status= tested_not_falsified` 不等于已经证明存在真实守恒律。非均匀时间间隔会按实际时间差计算，状态数大于导数行数时保留完整右奇异向量，避免漏掉短序列中的零空间。

当前版本只搜索线性不变量。积分守恒、非线性不变量、噪声稳健导数和跨轨迹验证属于后续扩展，不应把本模块结果直接当作物理定律。
