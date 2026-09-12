# 类型化符号回归执行切片

`core.typed_symbolic_regression.fit_typed_symbolic_expression` 是 LLM-SR 的窄执行入口。它只接受 JSON 表达式树，例如 `var`、`param`、`const` 与白名单算子 `add/multiply/exp/...`，并要求每个参数有有限上下界。表达式由本地 `scipy.optimize.least_squares` 拟合，末段样本作为时间留出。

`method=llm_sr` 时，源码字符串、Python 代码和仓库路径不会执行；只有符合契约的表达式树才进入本地拟合。缺少 SciPy 返回 `unavailable`，数值域错误返回结构化拒绝。结果包含训练/留出 RMSE、参数、表达式复杂度和基线误差。

这不是符号等价证明、单位证明、因果发现或现实正确性证明。最终确认仍须经过候选门、独立留出和适用边界证书。

同一入口也接受最多 8 个 `{expression, parameter_bounds}` 候选组成的池。候选共享同一时间留出，系统只按留出误差、复杂度和失败状态排序，返回 `candidates_need_confirmation`；`best_id` 不是自动批准，独立确认仍是必需步骤。

若同时提供 `feature_dimensions`、`parameter_dimensions` 和 `target_dimensions`，编译器会验证树内的加减、乘除、平方根和超越函数量纲；缺少任一部分时不伪造量纲结论，结果的 `unit_status` 为 `not_assessed`。

候选还可以声明 `allowed_operators` 和 `allowed_variables`，用于把题面/诊断先验
编译成白名单。`search_typed_symbolic_candidates` 在共享时间留出上进行有限的
常数邻域搜索，记录每代报告和候选哈希谱系；它不是 LLM-SR 官方实现，也不代表
结构发现或最终测试通过。通过 `method=llm_sr` 增加 `search` 配置即可启用该窄切片。
