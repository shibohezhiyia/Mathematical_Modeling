# 因果 DAG 结构门

`build_causal_dag_contract` 先验证变量角色、边、无环性和证据状态；`restrict_interactions_to_dag` 只把已声明边作为符号搜索优先范围，其余交互延后而不是判定为“不存在”。DAG 从不自动产生因果证据，未达到 `verified` 时不会授权因果执行。

新增 `discover_linear_causal_dag` 与 `/api/research/causal-discovery`：在有界数值表上选择一个
顺序约束线性 DAG，并通过 bootstrap 报告边稳定性，再生成 `evidence_status=partial` 的 DAG 合同。
它是真正可执行的结构筛查，可用于限制后续符号回归候选边；顺序假设、观测混淆和非干预数据仍使
结果不能升级为因果识别或干预效应证明。
