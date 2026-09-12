# 语义绑定契约

`core.binding_contract` 将表、变量角色、维度、目标、约束、初边值、有效域、来源和
未决项保存为版本化 JSON。缺少单位不会自动按无量纲处理；`compile_gate=blocked`
时只返回缺失项，不进入数值求解。

`plan_bound_subgraphs` 只根据已绑定角色规划 ODE/动力学、约束优化或类型化代数后端，
不返回数值答案。Web 入口为 `POST /api/research/binding-contract`。
