# ODE CEGIS 适配器

`core.ode_cegis` 将一个有限的低维 ODE 候选族接入统一 CEGIS：

```text
JSON 多项式 RHS
  → 结构/系数契约
  → scipy.solve_ivp
  → 轨迹误差与反例 witness
  → 单系数有界变异
  → 下一轮评估
```

候选允许 `constant`、`linear`、`quadratic`、`cross`、`time` 和最多四个显式 `driver:<name>` 基底，状态维度最多 3，且有积分评估预算。`cross` 是所有状态两两乘积的有界耦合项，`time` 是显式时间驱动；外部驱动通过案例中的同长度时间序列提供，并在积分器内部线性插值。候选还可以声明最多四个阈值事件 `{state_index, threshold, direction, reset_delta, priority, max_occurrences}`，事件触发后分段重新积分并把事件证据写入结果；同一时刻按优先级确定顺序，事件次数有硬上限，且样本端点会记录重置后的状态。所有新增项都通过逐方程系数控制，避免生成代码字符串。

候选也支持受限时滞项 `delay:<index>`：每个时滞指定状态和 `tau`，案例必须提供历史状态；运行时使用有界显式 Euler 方法的 method-of-steps 插值，现在允许非均匀但严格递增的采样网格。时滞系统也可使用采样阈值事件层：在步端检测穿越、按优先级处理同时规则，并每步至多执行一次有界原子复位（这不是连续事件根细化）。`solver` 可选择 `auto/RK45/RK23/DOP853/Radau/BDF/LSODA`；`auto` 用初值处有限差分 Jacobian 的谱半径/特征值展宽在 `Radau` 和 `RK45` 间路由，并把诊断写入 `integration_evidence`，这不是全局刚性或稳定性证明。事件可用 `reset: [..]` 对多个状态做一次原子复位；`requires` 仍只允许依赖较早规则，形成有界无环事件网络。可选 `units` 契约声明状态、时间、驱动和系数维度；编译期会逐基底传播量纲并拒绝不相容的系数。若只给出 `states/time/drivers`，系数单位会自动推导并标记 `auto_propagated=true`，不再把单位缺失伪装成已确认。

除单系数变异外，CEGIS 可以一次加入一个尚未启用的基底（系数初始化为零），再由同一受限搜索调参；这是真正的有限结构变异，不是任意代码生成。编译失败、积分超时或数值爆炸属于 `not_assessed`，不会被当作反例；只有观测轨迹误差超过阈值才会进入 CEGIS 反馈。

示例：

```python
from core.ode_cegis import run_ode_cegis

result = run_ode_cegis(
    [{"state_dim": 1, "basis": ["linear"], "coefficients": [[0.0]]}],
    [{"id": "case", "times": [0, 0.5, 1], "initial": [1],
      "observations": [[1], [0.7788], [0.6065]]}],
)
```

这是一个可执行的动力学适配器，不代表任意 ODE、隐变量、参数可辨识性或现实机理已经自动发现。驱动外推、连续事件面/随机事件、连续状态相关重置、守恒/非负性证明、单位自动从自然语言题面绑定，以及完整 PDE 仍需各自契约和独立验证；时滞与事件组合仅支持采样步端的受限近似。`instability` 指标只是数值守护边界代理，不是 Lyapunov 稳定性结论；`auto` 刚性结果也只能作为求解器选择证据。
