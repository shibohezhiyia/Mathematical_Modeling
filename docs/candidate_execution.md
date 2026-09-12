# 候选结构执行门

`core.candidate_execution.execute_structure_candidate` 是结构提议到数值执行之间
的窄接口。候选必须包含完整 `primitive_graph`，图先经过类型/量纲检查，再由
`primitive_graph_runtime` 执行有限代数子图。变量只能由调用者通过 `bindings` 显式
提供；`unknown_mechanism`、微分、事件、优化和潜变量节点不会被猜测填充。

返回值同时保留 `proposal_status` 和 `evidence.independent_validation`，因此“执行成功”
不会被渲染为真实正确或跨题泛化证明。Web 端可通过
`POST /api/research/structure-candidate/execute` 调用。

`evaluate_structure_candidate` 还可作为 `run_cegis` 的确定性 evaluator：它在一组
显式绑定/期望值上返回 RMSE、有限反例 witness 和成本，不修改候选。变异器仍需由
上层提供并受 CEGIS 预算约束；数值通过不等于现实正确。

`run_arithmetic_candidate_cegis` 提供一个更窄的默认变异器，只修改有限标量常数并
重新执行全部案例。它不是 ODE、优化或多表的自动修复器，且达到容差只表示该案例
集合未发现反例。Web 端可通过 `POST /api/research/structure-candidate/cegis`
提交候选和显式案例；配置预算由服务端校验，旧候选、反例档案和未决状态都会保留。
