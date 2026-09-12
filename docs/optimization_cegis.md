# 优化族 CEGIS 适配器

`core.optimization_cegis` 把现有 `linear_program/v1`、`mixed_integer_linear_program/v1` 和
`quadratic_program/v1` 合同接入统一的
`compile → evaluate → diagnose → patch → replay` 流程。候选仍然是经过
`UniversalRelationValidator` 校验的 JSON 数学合同，不执行生成的 Python。

适配器支持两类有限证据：

- 每个案例可提供 `overrides`，对目标系数、边界和线性约束做受限场景扰动；
- 每个案例可提供 `expected_objective`，用于检查求解结果是否偏离已知参考值。

求解由 `scipy.optimize.linprog(method="highs")` 完成。求解失败、合同错误或资源问题返回
`not_assessed`，不会被伪装成反例；只有目标偏差超过容差才进入 CEGIS 反例档案。
目标系数（QP 为线性项）的单坐标小步变异仅用于证明闭环可执行，不能替代完整结构搜索、非凸
优化、多目标权衡或现实约束建模。

```python
from core.optimization_cegis import run_optimization_family_cegis

result = run_optimization_family_cegis("linear_program", [candidate], cases)
```

该模块当前成熟度为 `experimental`，需要在独立优化题和固定预算下与基线比较后才能升级。
