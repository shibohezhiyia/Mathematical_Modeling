# 因果 DAG 结构门

`build_causal_dag_contract` 先验证变量角色、边、无环性和证据状态；`restrict_interactions_to_dag` 只把已声明边作为符号搜索优先范围，其余交互延后而不是判定为“不存在”。DAG 从不自动产生因果证据，未达到 `verified` 时不会授权因果执行。

新增 `discover_linear_causal_dag` 与 `/api/research/causal-discovery`：在有界数值表上选择一个
顺序约束线性 DAG，并通过 bootstrap 报告边稳定性，再生成 `evidence_status=partial` 的 DAG 合同。
它是真正可执行的结构筛查，可用于限制后续符号回归候选边；顺序假设、观测混淆和非干预数据仍使
结果不能升级为因果识别或干预效应证明。

同一接口提供 `time_lags` 分支（`discover_temporal_causal_graph`）：对按时间排序的
序列建立目标自回归基线，逐项加入来源滞后，使用时间尾部留出、验证 RMSE 增益和块
bootstrap 符号稳定性筛查有向滞后交互。它适合为动态图/GNN 或动力学候选缩小搜索空间，
但输出仍标记为 `screened_temporal_predictive`，不构成 Granger 因果、干预识别或全局
动态系统证明。
