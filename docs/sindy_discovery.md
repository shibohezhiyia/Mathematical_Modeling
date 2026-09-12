# 稀疏动力学候选

`discover_sparse_dynamics` 是一个不依赖专用题型的 SINDy 风格切片：从有限差分导数构造受限多项式库，在训练段用顺序阈值最小二乘保留稀疏项，再用末段验证导数残差。

```python
from core.sindy_discovery import discover_sparse_dynamics

candidate = discover_sparse_dynamics(times, states, ["x", "y"], polynomial_degree=2)
```

若能绑定状态和时间单位，可在候选生成阶段启用量纲剪枝：

```python
candidate = discover_sparse_dynamics(
    times, states, ["x", "y"], polynomial_degree=2,
    state_dimensions=[{"T": 1}, {"T": 2}],
    time_dimension={"T": 1},
)
```

此时每个方程只拟合与 `d(state)/dt` 维度相容的基底，结果会保存 `allowed_library_terms` 和单位过滤状态。单位绑定错误会导致候选库为空并明确失败，不会静默放宽约束。

输出的方程是当前数据、有限差分和多项式库下的候选。`candidate_found` 只表示留出导数误差通过当前阈值；它不证明 ODE 真实、噪声稳健、外推有效或状态空间已经闭合。复杂库、弱形式积分、噪声导数估计和单位约束仍需后续接入。

验证失败时结果还会保留最多 32 个确定性反例（时间点、方程和绝对残差），供后续 CEGIS/结构变异器重放；反例记录是验证证据，不代表系统已经自动完成修复。

对导数噪声较大的序列，可以调用 `discover_weak_form_dynamics`。它使用端点为零的局部测试函数和分部积分，把 `w·x'` 改写为 `-w'·x`，再按窗口留出验证。该路线减少直接差分放大的影响，但仍受窗口、库、采样和观测噪声影响，不能称为噪声稳健证明。
