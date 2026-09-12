# 文档中心

本目录只保留面向使用者的文档，自动生成的大型 API 清单不再提交到仓库。

## 文档

- [未完成能力实施方案](implementation_backlog.md)：当前缺口、代码接入点、分阶段实现、进一步优化和验收条件；对应 ROADMAP 的 R0–R9。
- [模块能力目录](../core/module_catalog.py)：查看模块成熟度、用户入口和证据范围；也可运行 `python scripts/audit_module_catalog.py --check-imports`。
- [模块引用审计](module_usage_audit.md)：静态扫描源码、测试和 Web 引用，识别只登记未接入的模块；不冒充运行覆盖率。
- [Worker 权限与磁盘配额](worker_permissions.md)：允许目录、网络/子进程开关和有界磁盘写入检查；不是 OS 级沙箱。
- [模型族 CEGIS 适配契约](model_family_adapters.md)：统一编译、评价、诊断、修复和反例重放边界；不冒充各后端已完成。
- [ODE CEGIS 适配器](ode_cegis.md)：低维多项式 RHS 的受控积分、轨迹反例和系数修复；不是通用微分方程求解器。
- [线性规划 CEGIS 适配器](optimization_cegis.md)：复用线性规划合同的有限扰动、目标偏差检查和反例修复；不是任意优化器。
- [USAGE.md](USAGE.md)：Python、Web 界面、多数据集和多模态调用示例。
- [项目主页](../README.md)：安装、功能概览和最小运行流程。
- [安全说明](../SECURITY.md)：API Key、公开部署和隐私文件处理规则。
- [数学图搜索](graph_search.md)：受限原语组合、反例重放和候选审计。
- [表格到数学图搜索](data_graph_bridge.md)：将一至四列数值特征绑定到未知机制图并执行受控候选搜索。
- [轻量图消息传递交互筛查](gnn_interaction_screen.md)：可选 PyTorch 边门消息传递和验证集证据边界。
- [规则网格 PDE 候选发现](pde_discovery.md)：一维/二维时空场的有限差分候选、留出残差和边界审计。
- [模型引导的受限数学图变异](model_guided_graph_mutation.md)：模型只提交窄范围 IR patch，仍由确定性校验器与反例裁决。
- [规模感知执行路线](backend_planner.md)：按数学结构和规模生成保守执行计划。
- [模型竞争与判决](model_competition.md)：比较候选并保留未决、反例和结构分歧。
- [开放模型基准](open_model_bench.md)：公开基准清单和结果记录边界。
- [真实未见题封存基准](blind_benchmark.md)：公开哈希清单、泄漏扫描、固定预算运行和解封计分。
- [真实未见题统计评估](blind_statistics.md)：独立解封评分、准确率区间和配对显著性检验门槛。
- [评估来源审计](evaluation_sources.md)：区分公开回归、私有防污染基准和可发表的真实未见题证据。
- [独立未见题登记](holdout_intake.md)：封存前检查题型分层、哈希承诺和出题/评审角色隔离。
- [外部未公开题与评审合作](external_holdout_outreach.md)：可联系的建模竞赛渠道、所需材料和不可声称边界。
- [数学工具](dimensional_analysis.md)：无量纲分析、性质检查和梯度校验。
- [守恒律候选发现](conservation_discovery.md)：有限差分零空间、验证段和反证状态。
- [结构诊断信号](structure_diagnostics.md)：有界单调/周期/变点筛查和搜索提示。
- [延迟嵌入潜状态](latent_state_discovery.md)：有界延迟嵌入和留出重建候选。
- [未闭合系统竞争解释](unclosed_state_competition.md)：并列筛查潜状态、外部输入、异方差、非平稳与函数库/数值失配。
- [局部可辨识性](identifiability.md)：局部 Jacobian 秩、条件数和参数退化检查。
- [稀疏动力学候选](sindy_discovery.md)：有限差分、多项式库和留出验证。
- [模型集合评估](model_set_assessment.md)：不压成伪后验的模型分歧与共同结论。
- [诊断到搜索提示](diagnostic_router.md)：把结构/残差信号转成可撤销的原语搜索方向。
- [域外反证](domain_extrapolation.md)：声明域与扩展域的有限压力测试和反例记录。
- [全局敏感性](sensitivity_analysis.md)：参数置换总效应和搜索优先级筛查。
- [参数不确定性](parameter_uncertainty.md)：条件于当前数据的 bootstrap 稳定性报告。
- [可逆尺度变换](reversible_scaling.md)：保留单位签名、回变换和原始尺度复核。
- [输入快照隔离](input_snapshot.md)：防止用户更新输入后旧任务覆盖新结果。
- [变量投影](variable_projection.md)：在有界搜索中消去线性参数块并记录秩亏。
- [数值稳定性](numerical_stability.md)：不同求解停止容差下的结果一致性检查。
- [通用原语运行时](primitive_runtime.md)：受限几何、事件和统计原语的契约、预算和 worker 路由。
- [类型化原语图运行时](primitive_graph_runtime.md)：不解析生成源代码，按依赖拓扑执行通过量纲检查的有限代数子图。
- [候选结构执行门](candidate_execution.md)：将已绑定的候选图送入受限运行时，并保留提议、验证和独立确认的边界。
- [语义绑定契约](binding_contract.md)：固定变量角色、单位、目标、约束和未决项，再规划可执行后端。
- [跨域类比映射](analogy_mapping.md)：变量/单位/交互/守恒/边界映射，类比只作为可撤销提议。
- [批量候选评价](batch_candidate_evaluation.md)：共享特征视图、最小诊断反馈和有界重试计划。
- [四层不确定性传播](uncertainty_propagation.md)：按语义、结构、参数和数值层分解经验方差。
- [不确定性与校准审计](uncertainty_audit.md)：将四层经验变化和独立留出校准分开报告。
- [主动澄清的信息价值](information_value.md)：按候选分歧与声明成本排序澄清问题；结果不是后验概率。
- [结论证据证书](conclusion_certificate.md)：强制保留边界、残差、约束和反例字段。
- [搜索控制消融](search_ablation.md)：预登记缓存、诊断、先验和反例重放的单因素消融。
- [跨进程资源配额](shared_resource_quota.md)：SQLite 原子租约和过期回收。
- [外部未见题基准来源](benchmark_sources.md)：15 个真实 MCM/ICM 封存案例、官方来源和解封评测边界。
- [外部验证实跑记录](external_validation_20260911.md)：5 个 UCI 数据源、6 个案例的固定留出与独立评分结果（描述性证据，不冒充竞赛金标准）。
- [评测冻结](evaluation_freeze.md)：冻结源码、依赖、提示摘要、随机种子和预算，防止外部评测期间修改系统。
- [当前系统与旧版基线对照](system_benchmark.md)：在同一冻结切分上运行完整 ModelingEngine 与 Ridge 基线。
- [多表结构 CEGIS](multitable_cegis.md)：多表键、粒度和聚合的受限编译、执行与反例修复。
- [外部来源注册表](../examples/benchmark_sources.json)：题面、评审量规、研究基准和参考候选的机器可读目录；公开题解不会被当作金标准。可用 `python scripts/list_benchmark_sources.py` 筛选。
- [外部方法来源与接入边界](external_methods.md)：PySINDy、UDE、LLM-SR、LLM-SRBench 及 2023/2024 COMAP 未见题的可复用入口和严格评测边界。
- `data/benchmarks/`：公开题源索引与外部研究基准元数据；仅登记来源，不把公开题解/论文分数当作独立金标准。
- [公开数据标签隔离替代评测](public_holdout_fallback.md)：无法取得封存题和独立评分时的可复现回退方案及证据边界。
- [类型化符号回归](typed_symbolic_regression.md)：JSON 表达式树、参数边界和时间留出执行边界。
- [外部方法候选路由](external_method_router.md)：根据题目证据规划 SINDy、weak-form、PDE、UDE 和 LLM-SR 候选，不默认假设依赖已安装。
- [外部方法 CEGIS 执行桥](external_method_cegis.md)：将类型化 PDE/UDE/神经 UDE/LLM-SR 候选接入受限反例修复循环。
- [预测校准检查](calibration.md)：独立评估预测区间覆盖率、区间得分和概率校准误差。
- [合成校准](synthetic_calibration.md)：在显式生成器上复核有限样本覆盖率，不冒充现实覆盖证明。
- [概率方差传播](../core/probability_variance.py)：显式概率质量下的四层全方差账本。
- [统一候选指标](candidate_metrics.md)：七目标同构校验与 Pareto 保留。
- [残差解释比较](diagnostic_comparison.md)：在增加隐变量前比较数值、噪声、可辨识性和结构解释。
- [隐变量误报控制](latent_state_benchmark.md)：高噪声无隐变量对照的运行特征。
- [观测等价模型](observational_equivalence.md)：有限评估点等价集合，不升级为代数证明。
- [重写证书](rewrite_certificate.md)：对已登记等价变换生成条件化证书，缺条件保持未证明。
- [对抗案例生成](adversarial_cases.md)：端点、退化、极端和扰动输入生成，不替代模型评价证据。
- [数据/公理候选筛选](axiom_candidate_filter.md)：显式证据引用和域外反例门。
- [同预算配对基准](paired_budget_benchmark.md)：固定种子、预算、失败分母和资源指标。
- [冷/热缓存基准](cache_performance_benchmark.md)：cold/warm/incremental 一致性与时延记录。
- [worker 权限策略](worker_permissions.md)：Python audit-hook 防御层及 OS 隔离边界。
- [OS 级隔离证明](os_sandbox.md)：严格模式的外部 supervisor/容器证明与失败关闭策略。
- [观测与数据处理不确定性](observation_uncertainty.md)：分解观测、处理、交互与重复测量波动。
- [数值后端策略提示](numerical_strategy.md)：根据条件数、刚性、事件和梯度证据选择保守路由。
- [结构感知降维计划](structure_reduction.md)：识别稀疏、块结构和对称性，但保留完整问题对照。
- [流式任务与联表计划](stream_plan.md)：列投影、跨块状态协议和多对多基数上界准入。
- [有界工作队列](work_queue.md)：同时限制内存、原生线程和 API 调用窗口。
- [交互预览抽样](preview_sampling.md)：有界、确定性地保留稀有组和异常记录。
- [缓存复用政策](cache_policy.md)：等价证书、失败缓存分类、来源检查和 warm-start 门。
- [数据切分与预处理隔离](split_integrity.md)：锁定最终测试并审计训练摘要不被其改变。
- [交互、UDE 与 PDE 候选空间](structure_extensions.md)：非因果交互筛选和受限微分方程候选契约。
- [神经 UDE 残差后端](ude_neural.md)：可选 PyTorch 残差修正、时间留出与基线比较边界。
- [对比与消融预注册](ablation_protocol.md)：固定任务/预算/评分，保留失败分母。
- [配对系统比较执行](comparison_protocol.md)：相同任务网格运行多 arm，并保留失败与配对效果边界。
- [逐轮保真度晋级](progressive_budget.md)：低保真仅引导搜索，保留探索与未决候选。
- [多保真候选运行器](multifidelity_runner.md)：统一低保真排序、高保真确认和反例预算。
- [最终确认预算](confirmation_budget.md)：同时预留最终确认和反例搜索额度。
- [假设滑块与局部重算](hypothesis_controls.md)：有限控件 schema、下游节点闭包计划和类型化预览执行。
- [建模选择政策门](modeling_policy.md)：禁止无证据强制定律、固定删减和停用基线。
- [调度压力协议](stress_protocol.md)：记录有界队列的完成、取消、超时和准入拒绝。
- [受限 SymPy 后端](sympy_backend.md)：从安全 AST 构造符号表达式，不调用 `sympify`。
- [可选后端运行时](optional_backend_runtime.md)：SciPy/SymPy/CVXPY/Pyomo/JAX 的结构化、显式不可用边界。
- [中间量复用计划](reuse_plan.md)：按视图、特征、稀疏结构和键指纹复用并保留冷启动对照。
- [搜索节省量审计](efficiency_accounting.md)：统计静态/低保真跳过，同时锁定独立、稀有事件和慢收敛检查分母。
- [持久中间量缓存](persistent_compile_cache.md)：JSON-only 跨运行编译缓存，损坏时冷启动且不缓存判决。
- [多任务调度压力协议](multitask_stress.md)：记录并发任务的完成、异常、取消、超时和准入恢复。
- [交互式数学结果曲面](interactive_surface.md)：约束状态、有限敏感性曲面和无权重 Pareto 数据契约。
- [数据来源与污染审计](data_provenance.md)：来源授权、内容指纹和开发/提示/最终集重叠检查。
- [未见题独立评分](rubric_review.md)：固定评分量表、双评审和分歧记录。
- [离散外微分契约](discrete_exterior_calculus.md)：带方向复形、形式次数和 `d²=0` 拓扑义务。
- [因果 DAG 结构门](causal_dag.md)：先验证角色与无环性，再限制搜索边，不把结构假设冒充因果证明。
- [验证后经验库](experience_store.md)：脱敏、内容寻址、结构检索和安全清理。
- [建模概念库](concept_library.md)：保存可检验概念、实现方式、适用条件和历史反例。
- [反例引导修复](cegis_repair.md)：把已记录反例转成受控原语优先级，不绕过验证。
- [分级候选评估](staged_evaluation.md)：在预留确认预算的前提下逐级筛选候选，区分反例与资源未决。
- [增量编译缓存](incremental_compile_cache.md)：按版本、来源和上游子图键复用中间产物。
- [模型判决书](model_verdict.md)：保守区分获准、条件、未决和被反证状态。
- [执行准入门](execution_readiness.md)：统一类型、单位、来源、资源和安全证据，不以缺失字段默认放行。
- 代码 API：`core.counterexample_protocol`（反例分类、端点优先测试与重放）、`core.proposal_router`（本地/外部提议路由）、`core.repair_router`（确定性修复优先级）、`core.batch_candidate_evaluation`（共享特征与最小反馈）、`core.analogy_mapping`（跨域映射契约）、`core.safe_code`（AST 白名单）和 `core.proof_certificate`（证明条件证书）。

## 本地运行

```bash
python web/app.py
```

打开 <http://127.0.0.1:5000>。

## 测试

```bash
python -m pytest -q
```

如果需要重新生成开发者 API 索引，可运行 `generate_api_docs.py`；生成文件属于本地产物，不应提交到公开仓库。
