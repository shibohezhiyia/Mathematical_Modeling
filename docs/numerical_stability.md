# 求解容差稳定性

`assess_solver_tolerance_stability` 用同一个受控求解器在多个固定停止容差下运行，逐个输出比较绝对差异和允许阈值：

- `stable_on_tested_tolerances`：记录的容差点之间未发现超阈值差异；
- `unstable`：至少一个输出在容差变化下发生超阈值变化，并保留具体反例；
- `not_assessed`/`partial`：求解器失败或输出 schema 不一致，不能把失败当成不稳定或稳定。

这是有限数值敏感性证据，不是离散化误差界、条件数定理或全局数值正确性证明。求解器仍应运行在资源监督环境中。

`assess_numerical_error_budget` 另外记录求解器提供的离散化误差、条件数和误差界元数据；它要求单位一致、值有限且非负，但不会把声明性估计升级成普适误差定理。
