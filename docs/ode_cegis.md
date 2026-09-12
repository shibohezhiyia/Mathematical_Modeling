# ODE CEGIS 适配器

`core.ode_cegis` 将一个有限的自治 ODE 候选族接入统一 CEGIS：

```text
JSON 多项式 RHS
  → 结构/系数契约
  → scipy.solve_ivp
  → 轨迹误差与反例 witness
  → 单系数有界变异
  → 下一轮评估
```

候选只允许 `constant`、`linear`、`quadratic` 基底，状态维度最多 3，且有积分评估预算。除单系数变异外，CEGIS 可以一次加入一个尚未启用的基底（系数初始化为零），再由同一受限搜索调参；这是真正的有限结构变异，不是任意代码生成。编译失败、积分超时或数值爆炸属于 `not_assessed`，不会被当作反例；只有观测轨迹误差超过阈值才会进入 CEGIS 反馈。

示例：

```python
from core.ode_cegis import run_ode_cegis

result = run_ode_cegis(
    [{"state_dim": 1, "basis": ["linear"], "coefficients": [[0.0]]}],
    [{"id": "case", "times": [0, 0.5, 1], "initial": [1],
      "observations": [[1], [0.7788], [0.6065]]}],
)
```

这是一个可执行的动力学适配器，不代表任意 ODE、隐变量、参数可辨识性或现实机理已经自动发现。完整 PDE、非自治输入、刚性/事件 ODE 仍需各自契约和独立验证。
