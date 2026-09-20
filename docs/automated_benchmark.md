# 自动可判定基准

项目现在把外部盲评降为可选补充，默认优先使用有真值、可自动判分的冻结任务。
`core.automated_benchmark` 实现三方分离：

1. 题目/数据生成器保存隐藏参考；
2. 被测系统只收到题面和公开输入；
3. 独立评分器才读取参考结果并返回 `valid`、误差或约束指标。

这类设计有公开研究先例：[LLM-SRBench](https://proceedings.mlr.press/v267/shojaee25a.html)
用表示变换与合成任务评价科学方程发现；[OptiBench](https://arxiv.org/abs/2407.09887)
覆盖从人类可读输入（包括表格）到求解器数值答案的端到端优化建模。它们说明自动真值任务
可以形成正式研究证据，但不能把特定基准上的结果外推为全部开放式建模质量。

仓库还提供 `core.automated_benchmark_suite`：默认生成 32 个任务（代数、ODE、优化、
多表各 8 个），共用参数变体但按 7 个结构组计数。它是开发/确认流程的可运行起点，
不是现实题库，并且只考查已给出模型后的类型化执行。典型调用：

```python
from core.automated_benchmark_suite import build_automated_benchmark_suite, score_automated_benchmark_case
from core.automated_benchmark import run_automated_benchmark

cases = build_automated_benchmark_suite(cases_per_family=8)
report = run_automated_benchmark(cases, my_system_adapter,
                                 score_automated_benchmark_case)
```

`my_system_adapter` 只能读取每个案例的公开 `statement` 与 `input`；不能把
`hidden_reference` 作为参数传进去。若要做最终确认，先复制结构组并变换单位、变量名、
列顺序或噪声，再冻结代码和预算，而不是把同一套 32 个任务重复运行当成独立证据。

每个任务必须带 `structure_group`。同一个方程改变参数、同一数据生成机制改变随机种子，
只能算同一结构组，不能把重复变体当作独立题目。结果保留失败、超时和不完整行，
`valid_rate` 只在有自动真值的任务上生成，不能外推为现实建模准确率。

仓库内置的 `typed_execution_baseline` 可直接验证四类类型化执行后端是否连通：

```powershell
python scripts/run_automated_benchmark_suite.py
```

默认报告写入 `artifacts/automated-benchmark/latest.json`。当前基线将代数图交给受限
primitive graph runtime，ODE 交给 `scipy.solve_ivp`，优化交给已验证的
`UniversalSolverRegistry` 线性规划后端，多表交给 `multitable_cegis`。这条基线的目的
是回归执行契约；它不包含题面语义解析，也不代表任意新题泛化。完整 32 个默认任务通过
只能说明这些固定结构的生成器、执行器和独立评分器已连通。

## 从题意和原始附件开始的建模层基准

`core.modeling_benchmark_suite` 另行提供 13 个结构不同的小型端到端任务。公开输入只有题意、原始观测或
原始表格，不含计算图、正确 ODE 基底/系数、优化矩阵或连接计划；评分器同时要求模型描述和
独立可复算答案。缺少决策向量、变量/约束、结构或数据关系时，即使标量答案碰巧正确也不通过。

```powershell
python scripts/run_modeling_benchmark_suite.py
```

若已按数据提供方条款在本地准备 LLM-SRBench Parquet 快照，可运行外部格式适配器；路径和源码修订必须
显式传入，程序会记录每个 Parquet 文件的 SHA-256。社区镜像结果不会自动升级为官方数据结果：

```powershell
python scripts/run_llm_srbench_pilot.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_trig_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_extended_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_transformed_power_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_harmonic_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_angular_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
python scripts/run_llm_srbench_product_angle_confirmation.py --dataset-dir <parquet目录> --source-revision <40或64位修订>
```

当前外部符号评分不仅比较隐藏测试输出，还会用 `core.model_submission_evaluator` 独立执行提交的模型描述，
并要求重算值与提交预测一致。模型缺字段、执行失败或二者不一致时，数值预测即使碰巧正确也不计成功。
v11--v17 生成时尚无这项检查，故历史报告只能支持预测指标，不能单独证明提交的符号模型正确。

若本地准备了官方 SRSD-Feynman easy 快照，可先运行开发评分管线；正式确认脚本会排除固定开发 ID、
原子消费协议种子并冻结源码，同一协议/种子不能重跑：

```powershell
python scripts/run_srsd_pilot.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_srsd_final_confirmation.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_srsd_gplearn_development.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_srsd_gplearn_confirmation.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_srsd_portfolio_development.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_srsd_portfolio_confirmation.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_srsd_same_split_attribution.py --dataset-dir <SRSD目录> --source-revision <40或64位修订>
python scripts/run_product_workflow_confirmation.py
python scripts/run_product_discontinuity_confirmation.py
python scripts/run_product_routing_confirmation.py
```

v18 正式结果为当前有限语法 `7/8`、关闭多变量非线性的诊断消融 `2/8`；全部提交模型均通过独立重算，
运行中 API 调用和人工介入为零。该消融共享一元分支且不是成熟外部搜索器；PySR、gplearn、Operon 和
SymbolicRegression 在运行环境中均不可用，所以报告把强基线明确记为 `not_executed`，不据此声称
搜索策略先进性。完整结果在 `artifacts/srsd-confirmation/official-easy-20261009.json`。

gplearn 对照依赖单独固定在 `requirements-symbolic-baseline.txt`，不会把可选研究依赖伪装成普通运行的
必需组件。v19 使用官方提交 `0390aea8639ce5f6c0b388400e07b58c05acad6a`，1000 个体、20 代、单线程、
固定随机种子和每题 20,000 次程序评估上限。新 8 题上当前系统 `5/8`、gplearn `3/8`，区间跨零；
完整记录在 `artifacts/srsd-confirmation/official-gplearn-20261011.json`。两臂共享数据、采样、墙钟、内存、
API 与人工介入预算，但不共享完整候选表示和先验，因此不能把结果写成纯搜索算法优势。

训练内组合路由固定 80% 拟合、20% 验证，不读取隐藏测试答案。v20 新 8 题为组合 `5/8`、当前系统
`2/8`，但组合顺序运行两个求解器，累计 worker 时间 `96.343 s` 对 `14.58 s`，平均 NMSE 还更差。
报告必须同时给出有效解增量、错误路由、墙钟倍率和连续误差，不能只展示 `5/8`。完整模型和路由依据
保存在 `artifacts/srsd-confirmation/official-portfolio-20261013.json`。

确认后，同一批已消费 v20 任务用于开发归因：当前语法和 gplearn 严格共享 80% 拟合行与 20% 验证行。
当前语法为 `4/8`、gplearn 为 `1/8`；新拒答路由覆盖 `5/8`，覆盖内 `5/5` 有效，`3/8` 拒答，
相对当前语法只增加 1 个有效解。完整报告在
`artifacts/srsd-pilot/same-split-attribution-v20.json`。该文件不能登记为新确认集成绩；它只说明原 v20
的三个增量混合了训练切分效应和一题求解器互补。统计汇总必须同时报告总任务数、覆盖率、覆盖内错误
和拒答率，不能通过移除拒答提高分母上的成功率。

生产路由现在把“两臂都未通过验证”映射为 `needs_input` / `abstain`，并建议补充观测或扩大声明模型
范围。研究主入口和网页默认开启该路由，但只在自动绑定为单表代数、观测不少于 64 条且题意提供显式
查询时触发；否则保留原受限自动建模行为。gplearn 是可选研究依赖，缺失或技术失败不会伪装成验证通过。

v21 进一步从普通研究入口执行 8 个原始单表任务，响应字段使用业务名称并由用户明确选择 `target`，查询
只出现在题面。冻结结果为组合 `7/8`、单求解器 `6/8`，耗时 `108.86 s` 对 `25.562 s`。组合仍把
一个语法外阶跃接受为平滑双 `tanh`，因此错误接受数为 1；该失败不能被总通过数掩盖。报告位于
`artifacts/product-workflow-confirmation/confirmation-20261014.json`。确认后加入的精确分段常值跳变拒答门
不能追溯修改这份冻结报告。独立 v22 随后预登记 4 个新跳变和 4 个光滑对照：开启门 `8/8`、关闭门
`7/8`，开启门没有错误接受或光滑误拒，累计耗时 `129.063 s` 对 `235.093 s`。完整报告在
`artifacts/product-workflow-confirmation/discontinuity-20261015.json`。它只确认精确、占主导的分段常值跳变
模式，不支持对含噪或一般不连续结构的结论。

v22 后发现稀疏采样的光滑 `tanh(5x)` 也触发旧门，旧原因码对真实结构作出了过强判断。现将门控语义
改为“过渡区观测不足以区分陡峭光滑、跳变或量化”，并给出局部补点区间。v23 在新 8 个原始任务上
冻结比较按需第二臂与总跑两臂：两者均 `8/8`，按需顶层求解臂启动 `4` 次、耗时 `16.921 s`，
总跑两臂启动 `8` 次、耗时 `78.719 s`；该口径不含臂内部候选评估次数。四个过渡区任务均正确报告辨识不足，四个光滑对照均完成。
完整报告为 `artifacts/product-workflow-confirmation/routing-20261016.json`。本批没有当前臂失败而第二臂
补救的案例，因此成本收益成立于这批简单任务，当时互补恢复仍需新的独立对照。

v24 在新参数的四变量乘积正弦和仿射任务上冻结比较，按需和总跑两臂均 `4/4`，但第一臂全部
验证通过、第二臂没有救回；按需省 `4` 次顶层臂，耗时 `16.703 s` 对 `99.453 s`。
随后先以已消费开发探针确定 `sqrt(|x|)` 为互补结构，再在新域/新行序上冻结 v25：两组平方根题
均由 gplearn 第二臂救回，两组仿射题不调用第二臂；两策略均 `4/4`，启动 `6` 对 `8` 次顶层臂，
耗时 `29.578 s` 对 `65.094 s`。完整报告依次位于
`artifacts/product-workflow-confirmation/fallback-20261017.json` 与
`artifacts/product-workflow-confirmation/rescue-20261018.json`。平方根两题共用一个拓扑，不能把
它们当作两项独立结构，也不能把开发挑选的拓扑称为未见结构确认。

后续风险口径将范围外接受、错误预测接受、错误拒答和接受后正确率分开，不能再用单个
`unsafe_accept=0` 代替所有正确性判断。v26 以一个冻结代码版本运行 39 个四族组件任务和 10 个
产品入口任务，分别报告；产品按需策略 `10/10`、接受后 `6/6` 正确，单求解器仅 `5/10`，
并有 4 次范围外接受和 1 次错误预测接受。原始报告的两项汇总口径错误保留在
`artifacts/product-workflow-confirmation/mixed-risk-20261020.json`，逐题重算勘误见
`artifacts/product-workflow-confirmation/mixed-risk-20261020-erratum.json`，不可忽略勘误引用原报告。

v27 对同一批预声明的新任务运行一臂、按需两臂和总跑两臂：有效处置分别 `8/10`、`10/10`、
`10/10`，顶层臂启动 `6/8/12` 次，耗时 `25.594/36.344/117.156 s`。一臂对两组共享拓扑的
平方根题拒答，展示预算约束的覆盖代价；其余错误接受为零。报告见
`artifacts/product-workflow-confirmation/budget-decision-20261021.json`。这些仍是已知生成结构的
小规模任务，不应写成目标用户群体上的普适决策收益。

后四个确认脚本依次消费不同种子并排除全部先前已查看案例。v14 的新增三角幂律/Laurent 根式机制相对
消融为 `7/8` 对 `7/8`，配对效应为零；该负归因结果与此前正向结果同样保留。
v15 的谐波有理组合消融也为 `6/8` 对 `6/8`；虽然连续指标改善，主成功数未变，报告不把它改写成
有效解率提升。
v16 的角差位置组合消融为 `5/8` 对 `5/8`，连续指标也相同；这项零效应同样是正式结果。
v17 的乘积角/半角稀疏位置组合为 `8/8` 对 `8/8`；基线饱和，不能把无回归写成新增机制收益。

运行器为兼容历史产物保留 `frozen_old` 标签，但该实现只是“缺少类型化合同”接口探针，报告明确标记
`comparison_eligible: false`，不再计算它与新版之间的方法效应。可执行方法对照只有
`candidate_new` 与 `simple_tool_baseline`，失败仍保留在分母。
仓库内置的 `candidate_new` 连接当前受限原始记录归纳入口：它自行识别支持的模型族、拟合候选并
形成模型和答案；无法安全确定连接口径时返回 `needs_input`。正式实验必须把三个标签分别绑定到冻结版本/实现，并由进程级适配器
执行模型/API/内存预算；内置运行器只传递统一预算合同并对已返回结果检查墙钟超限。
当前能力、开发结果和未验收边界见 [能力证据状态](current_capability_evidence.md)。

当前开发夹具实跑为类型化接口探针 0/13、受限自动建模 13/13、简单工具基线 8/13。报告同时保存
源码摘要、统一预算合同、资源计数、配对有效率效应、单机制门控消融和评分器攻击审计。当前攻击审计
要求 13 个合法控制全部通过，并拒绝 47 个缺模型、错参数、非有限值、伪目标、不可行决策、错单位、
错连接/时间语义和歧义强答输出；本次误接受为 0。攻击目录并非穷尽。由于这些任务参与了实现开发，
结果只能作为“入口已接通、机制确实影响这些案例”的开发证据；下一步必须换用冻结的独立确认任务。

## 2026-09-15 冻结内部确认

`scripts/run_modeling_confirmation.py` 先冻结源码、依赖、预算和三臂版本，再用种子 `20260916` 构造
39 个确认案例，最后复核源码未变化。案例来自 13 个既有结构组，但改变数值尺度、行序、附件顺序、
附件名和查询顺序；因此它是参数与表示稳健性确认，不是未知结构留出。

首次冻结运行保留全部结果：旧路径 0/39、当前系统 30/39、简单工具 14/39。当前系统的 9 个失败均为
乱序 ODE 观测触发 `ode_observations_invalid`；未用确认结果回调系统。确认报告保存在
`artifacts/modeling-confirmation/confirmation-20260915.json`，源码冻结摘要和确认包承诺均写入报告。

该确认包已经消费。之后当前分支把乱序 ODE 失败转成开发反例并修复，同时将内置三臂全部迁入受控
worker；所以当前源码不再匹配 v1 冻结摘要，也不得复用上述 39 题给修复版授予确认结论。修复后开发
基准的三臂 39 行全部记录父进程墙钟、一次性进程和 Windows Job 内存限制证据，资源比较资格为
`true`；成绩仍为 0/13、13/13、8/13。
确认运行器还会在生成案例前原子消费协议/种子，失败不会释放；更换输出文件名也不能重跑同一确认种子。

## 2026-09-15 冻结新结构挑战

`scripts/run_modeling_structure_challenge.py` 对当前修复版执行了另一轮先冻结、后生成、只运行一次的
8 结构挑战：立方/三阶交互代数、外部驱动/耦合 ODE、整数/等式优化、三表链/桥表歧义。它们与
13 个开发结构组互斥。旧路径、当前系统和简单工具均为 0/8；当前系统有 5 个结构错误、2 个模型缺失和
1 个歧义处理不完整。所有行均有 OS 资源监督证据，运行前后冻结一致。该结果被保留为核心负证据，
不允许针对同一挑战修复后重跑并改称确认。

## 挑战后扩展确认

上述 8 题没有重跑。项目另建不同参数的开发任务，实现三次/三阶交互、外部驱动/耦合 ODE、整数/等式
优化和三表链/桥表歧义处理；开发三臂为 0/8、8/8、0/8，四个族级消融各自使新版降至 6/8。
优化评分器同时补充整数性独立重算，连续松弛伪解会以 `integrality_violation` 拒绝。

种子 `20260919` 在生成新的 16 个参数与表示变体前被消费并冻结，首次运行为旧路径 0/16、当前系统
16/16、简单工具 0/16，8 个结构组全部成功，48 个三臂行资源监督完整。该结果只确认已实现的八类扩展，
不证明更高阶或任意未知结构泛化。

## 2026-09-15 第二轮库外结构确认

v2 四题因正式冻结前执行过回归，只保留为开发探测。替代的 v3 四题在候选首次执行前冻结，覆盖有理关系、
延迟动力学、基数约束优化和有效期区间联表；当前系统与简单工具均为 0/4。当前系统失败由三个结构错误和
一个模型缺失组成。冻结前后摘要一致，种子 `20260921` 已消费。四个结构组低于统计门槛，报告仅给
描述性差异而不输出置信区间。该结果不用于扩库后重跑，也不能解释为现实任务平均准确率。

## 组合算子语法确认

一元代数入口新增有限算子语法生成，而非为单个公式登记名称。首次 v4 未见拓扑冻结为 3/4；失败来自
数学等价表达式的表面树不同。该结果保留。评分器改用隐藏输入点独立复算等价性后，另建的 v5 四个
未执行嵌套拓扑在首次冻结运行中为当前系统 4/4、简单工具 0/4。种子 `20260923` 已消费，冻结前后
一致。结论仅限预声明的有限算子和深度预算，不是语法外或跨模型族的未知结构保证。

确认后的深度消融为二层 4/4、平坦层 0/4，但二者分别评估 21 与最多 5 个拓扑，不能称同计算量比较。
因此另建 8 个未执行拓扑并在冻结后比较：深度二完整枚举与深度一至三随机抽样均固定每题 21 个拓扑，
结果为 8/8 对 3/8，按结构组配对差 `+0.625`，95% bootstrap 区间 `[0.25,1.0]`。随机策略搜索空间
更宽；结果说明小型已声明空间的完整覆盖有收益，不说明枚举是创新搜索算法。该次为进程内实验，资源
成本不作比较。

## 结构留出

不要按单个参数变体随机切分。使用 `split_structure_holdout` 将同一
`structure_group` 的全部案例放入同一分区：

```python
from core.benchmark_holdout import split_structure_holdout

partitions = split_structure_holdout(cases)
# partitions["development"], ["confirmation"], ["final"]
```

默认最后两个排序后的结构组分别作为确认集和最终集；正式实验应在冻结前显式登记
分组。确认集和最终集不得用于提示词、模型选择或阈值调整。分区工具只保证结构不泄漏，
不保证这些合成任务等价于现实世界未见题。

推荐的冻结分层：

- 代数/符号：隐藏公式，检查留出点预测、结构等价和量纲；
- ODE/动力学：改变初值、参数和噪声，检查独立轨迹、残差、守恒和参数误差；
- 优化：用独立求解器检查可行性和目标差距，只有参考解有保证时才使用“最优”；
- 多表：固定实体关系、粒度、聚合和 point-in-time 口径，检查重复计数与未来泄漏；
- 等价变换/错误注入：变量改名、单位换算、列重排、无关列、缺失条件和单位冲突。

生成器不能调用被测系统，评分器不能修改输入或阈值。开发集、冻结确认集和最终测试集
必须使用不同的结构组或来源组；看到最终测试结果后继续调参，就必须重新冻结并改名为开发集。

外部盲评仍可用于开放式论文质量和表达质量，但没有外部评审时，项目仍可以独立报告：
有效解率、结构恢复、约束满足、留出误差、错误接受、人工步骤、耗时、内存和费用。
这些指标的结论范围限定在登记的任务生成分布内。

同一批冻结任务比较真实可执行旧版与新版时，以任务或 `structure_group` 为配对单位，报告效应大小、
置信区间和完整失败分母；多个随机种子或同一公式的参数变体不能冒充独立任务。统计检验可按
预登记方案补充（多数据集比较可参考 [Demšar 2006](https://www.jmlr.org/papers/v7/demsar06a.html)），
但是否得到 `p<0.05` 不是研发验收目标。
