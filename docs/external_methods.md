# 外部方法来源与接入边界

本项目可以借鉴公开方法，但不会把论文中的示例、排行榜分数或代码直接当成数学真值。所有外部材料只通过 [`examples/benchmark_sources.json`](../examples/benchmark_sources.json) 或 [`data/benchmarks/source_registry.json`](../data/benchmarks/source_registry.json) 登记，运行时不自动下载；执行前应固定版本、随机种子和资源预算，并将结果放入独立的 `method_comparison` 实验臂。

## 已核验的公开来源

| 方法 | 来源 | 可接入位置 | 当前边界 |
| --- | --- | --- | --- |
| SINDy / PDE-FIND / weak-form SINDy | [PySINDy 文档](https://pysindy.readthedocs.io/en/stable/index.html)、[弱形式示例](https://pysindy.readthedocs.io/en/stable/examples/1_feature_overview/example.html) | P8 结构发现、噪声稳健候选 | 文档示例不是题目答案；仍需独立残差、外推和稳定性审计 |
| Universal Differential Equations | [SciML UDE 文档](https://sciml.readthedocs.io/en/latest/UDE.html)、[原始论文](https://arxiv.org/abs/2001.04385) | P9 机理 + 学习修正候选 | 神经修正项不自动具有物理意义；必须与纯机理和统计基线竞争 |
| 联合轨迹 UDE（本地受限后端） | 同上 | `ude_joint` 运行时、P9 | 联合优化线性机理矩阵与神经残差；仅小维度 Euler 轨迹，不是通用 UDE |
| LLM 引导符号回归 | [LLM-SR 官方代码](https://github.com/deep-symbolic-mathematics/LLM-SR)、[论文](https://arxiv.org/abs/2404.18400) | P3/P5 结构提议与反例变异 | 代码需固定 commit 并检查许可证；模型提议必须经过受限编译和执行反馈 |
| 未见方程发现评测 | [LLM-SRBench](https://arxiv.org/abs/2504.10415) | P0/P7 外部方法对照 | 不得与 MCM/ICM 封存题混算；外部基准分数不能证明本项目覆盖率 |
| 真实未见建模题 | [COMAP 2024 题目页](https://www.contest.comap.org/undergraduate/contests/mcm/contests/2024/problems/)、[COMAP 2023 题目页](https://www.contest.comap.org/undergraduate/contests/mcm/contests/2023/problems/) | `external_real` 盲测输入 | 题面和附件需在选择提示/参数后封存；公开题解只能作 `reference_only` |

新增的公开来源目录还登记了 [Mamo](https://github.com/freedomintelligence/mamo)、
[BWOR](https://zenodo.org/records/20120692)、[EngiBench](https://github.com/IDEALLab/EngiBench)
和 [MM-Agent/MM-Bench](https://github.com/usail-hkust/LLM-MM-Agent)。这些来源分别适合
建模 Agent、运筹编译、工程设计优化和 MCM/ICM 题面回归；它们的论文分数或公开参考解
不自动转换为本项目的独立专家评分。

为寻找独立评分合作，又登记了 [IMMC 官方评审流程](https://www.imc-impea.org/immc.php)
和 [公开评审指导材料](https://www.immchallenge.org.au/files/2021_IMMC_Judges_Com.pdf)。
前者说明提交会由三位教授独立评审，后者可用于预注册量规草案；两者都不包含本项目的
封存题目或逐题评分，仍需要向赛事方申请由对方保管题面并在运行后解封评分。

## 推荐接入顺序

1. 先把 PySINDy 的普通、PDE 和 weak-form 运行器包装成同一 `candidate` 契约，统一记录特征库、微分器、稀疏阈值和资源消耗。
2. 再把 UDE 作为可选竞争臂；只有在条件数、末段外推、参数扰动和基线比较均通过时才进入候选集合。
3. 将 LLM-SR 限制为结构生成/变异器；现在只有带参数边界的类型化 JSON 表达式树可以进入本地拟合，源码、仓库和任意回调仍不能绕过类型、单位、来源、资源和安全准入；失败时记录可复现的反例。
4. 使用 COMAP 2023/2024 题面作为真实未见测试输入，用 LLM-SRBench 作为方程发现外部对照，两者分别统计。
5. `data/benchmarks/comap_public_catalog.json` 已登记 2020–2025 官方题目索引，可作为
   公开题源清单；若用于未见题统计，必须在模型冻结后重新抽样并保存题面/附件哈希。

## 不应自动化的事项

- 不下载或复制会员 Judges' Commentary、公开题解全文和未经许可的数据。
- 不把获奖等级、论文报告分数或单次运行成功当成 gold truth。
- 不把“发现了一个拟合良好的方程”表述成“证明了现实机理”；必须保留结构、参数、数值和语义不确定性。

命令行查看这些来源：

```bash
python scripts/list_benchmark_sources.py --role method_comparison
# 新增的外部研究基准目录
python scripts/list_benchmark_sources.py --file data/benchmarks/source_registry.json
```
