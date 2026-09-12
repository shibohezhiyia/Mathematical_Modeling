# Mathematical Modeling

面向数学建模竞赛的通用研究助手。它把题面、多个数据集和可选的外部模型统一编译成可审计的数学结构，支持数据分析、模型比较、验证反证和交互式图表。

## 快速开始

### 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 启动网页应用

```bash
python web/app.py
```

打开 <http://127.0.0.1:5000>，上传数据集或直接粘贴没有附件的题面即可开始。

### 使用外部模型（可选）

复制 `.env.example` 为 `.env`，或在网页请求中临时输入 API Key：

```env
FLASK_SECRET_KEY=请替换为随机长字符串
PUBLIC_MODE=0
```

DeepSeek、OpenAI 兼容服务和 Ollama 的配置见 [使用指南](docs/USAGE.md)。不要把真实密钥提交到 Git。

### 运行测试

```bash
python -m pytest -q
```

## 功能概览

### 题面到数学结构

- 自动拆解题目、小问、目标、约束和变量角色。
- 无数据题支持纯机理建模和通用数学 IR。
- 组合动力学、几何、网络、随机、优化、反演和鲁棒决策等结构。
- 校验单位、量纲、边界条件和可执行证据。

### 多表数据分析

- 支持 CSV、Excel 等常见表格，自动识别字段类型、时间列和候选键。
- 安全处理一对一、一对多、多对多关系，避免笛卡尔积导致卡死。
- 支持筛选、派生、聚合、透视、时间特征、窗口统计和跨表特征。
- 自动分析变量交互、非线性关系、时滞和层级结构。
- 可按固定字节数与 SHA-256 从官方来源获取外部评测数据：见 [`examples/external_dataset_catalog.json`](examples/external_dataset_catalog.json) 和 [`外部题源说明`](docs/benchmark_sources.md)。

### 建模与可信度评估

- 分类、回归、聚类、时序、因果、动力学和综合评价。
- 多模型比较、交叉验证、基线、置乱、稳定性、漂移和敏感性检查。
- 输出预测区间、模型差异、假设账本和可追溯证据，而不是只给单一分数。
- 结构诊断可生成受控搜索提示；参数 bootstrap、域外反证和求解容差检查帮助识别“似对非对”。
- 验证后的数学配方可脱敏、按内容寻址保存，并只能作为新题搜索种子，不能跳过当前题目的验证。
- 大数据自动限流、采样和降低高成本模型，减少内存爆炸与长时间无响应。
- 可视化也有独立的确定性行/列上限；写盘后立即关闭 figure，避免批量图表累积占满内存。
- 研究助手对直接传入的超大 DataFrame 也执行输入级行数/内存上限；保留 `source_rows`、采样策略和“不代表完整扫描”警告。
- 反例协议按规范违反、经验不匹配、数值失败和域外行为分类，支持端点优先测试、局部最小化和修复后重放；有限无反例只标为 `tested_not_falsified`。
- 外部/本地模型提议经过统一 validator 和有界路由；AST 白名单、证明证书和执行清单分别记录安全、数学证明与复现条件，三者不混为一个“已验证”开关。
- 通用原语运行时覆盖受限几何/事件（距离、相交、区域、视线、区间并集）和统计检验（似然、bootstrap、置换检验）；随机过程固定种子并记录预算，有限样本证据不会冒充定理。
- 跨域类比必须显式映射变量、单位、交互、守恒和边界；类比只生成可撤销搜索提议，不能直接授权执行。

### 可视化与结果管理

- 网页端交互式图表，支持筛选、分面、动画维度和多变量联动。
- 研究结果、缓存和临时数据隔离存放，带版本与校验信息。
- 论文写作不参与求解；只有验证通过的结论才可交给末端写作接口。

## 文档

- [使用指南](docs/USAGE.md)：Python API、网页流程、多数据集和多模态示例。
- [研发路线图](ROADMAP.md)：开放世界数学编译、模型搜索、反例修正与可信度评估任务清单。
- [数学图搜索](docs/graph_search.md)：受限原语组合、反例重放、等价去重和候选多样性审计。
- [规模感知执行路线](docs/backend_planner.md)：按 IR 契约和规模选择受控 worker、可信后端或延期执行。
- [模型竞争与判决](docs/model_competition.md)：保留 Pareto 候选、失败分母和反例阻断，不把分数当概率。
- [模型引导的数学图变异](docs/model_guided_graph_mutation.md)：模型只提交受限 IR patch，所有候选仍由类型检查、反例和数值评价器重新判定。
- [开放模型基准](docs/open_model_bench.md)：开发/未见/结构变换/对抗划分与结果协议。
- [真实未见题封存基准](docs/blind_benchmark.md)：题面/答案哈希承诺、泄漏扫描、固定预算运行与解封计分。
- [真实未见题统计评估](docs/blind_statistics.md)：独立解封评分、准确率区间与配对显著性检验。
- [评估来源审计](docs/evaluation_sources.md)：哪些外部来源能、不能支持真实未见题准确率或显著性结论。
- [独立未见题登记](docs/holdout_intake.md)：封存前的分层配额、角色隔离和哈希承诺检查。
- [外部未公开题与评审合作](docs/external_holdout_outreach.md)：获取外部题面和独立评分的合作路径。
- [公开题源目录](docs/public_benchmark_sources.md)：官方题源链接、结构标签与封存案例映射；不复制题面或答案。
- [公开实验配置](docs/public_experiment_config.md)：固定预算、随机种子、方法对照和未见题留出边界。
- [基线优先与分阶段验证](docs/baseline_staged_runner.md)：先展示基线，再区分快速探索和严格确认。
- [未见方程结构基准](docs/symbolic_discovery_benchmark.md)：合成方程支持恢复与留出残差的独立评估。
- [可微数学 IR](docs/differentiable_ir.md)：受限表达式图的 JAX/有限差分梯度执行。
- [有限时域最优控制](docs/optimal_control.md)：线性二次控制候选、边界裁剪和独立验证边界。
- [数学发现工具](docs/dimensional_analysis.md)：Buckingham π 无量纲候选与梯度交叉校验。
- [守恒律候选发现](docs/conservation_discovery.md)：基于导数零空间的线性不变量搜索与留出验证；候选不等于定律证明。
- [结构诊断信号](docs/structure_diagnostics.md)：单调、周期和变点候选，驱动下一轮模型搜索而不替代验证。
- [延迟嵌入潜状态](docs/latent_state_discovery.md)：训练/留出 SVD 候选，明确区分数学潜变量和物理隐变量。
- [未闭合系统竞争解释](docs/unclosed_state_competition.md)：并列筛查潜状态、外部输入、异方差、非平稳与函数库/数值失配，避免残差记忆被直接误判为隐变量。
- [局部可辨识性](docs/identifiability.md)：Jacobian/Fisher 近似、边界单侧差分和参数退化提示。
- [稀疏动力学候选](docs/sindy_discovery.md)：受限多项式库、训练/留出稀疏回归和反证状态。
- [模型集合评估](docs/model_set_assessment.md)：Pareto 候选、预测包络、决策共识与结构分歧。
- [诊断到搜索提示](docs/diagnostic_router.md)：结构/残差信号到可撤销原语方向的桥接。
- [反例引导修复](docs/cegis_repair.md)：将反例转成受控修复优先级并保留重验证门。
- [域外反证](docs/domain_extrapolation.md)：声明域与扩展域的有限压力测试。
- [全局敏感性](docs/sensitivity_analysis.md)：参数置换总效应和搜索优先级筛查。
- [数值稳定性](docs/numerical_stability.md)：多停止容差下的结果一致性检查。
- [通用原语运行时](docs/primitive_runtime.md)：受限几何、事件和统计原语的输入契约与资源边界。
- [跨域类比映射](docs/analogy_mapping.md)：变量/单位/守恒/边界映射及其不可执行策略。
- [四层不确定性传播](docs/uncertainty_propagation.md)：按语义、结构、参数和数值层分解经验方差。
- [预测校准检查](docs/calibration.md)：独立评估预测区间覆盖率、区间得分和概率校准误差。
- [合成校准与方差传播](docs/synthetic_calibration.md)：固定生成器覆盖检查和概率质量方差账本。
- [统一候选指标](docs/candidate_metrics.md)：七目标同构校验与 Pareto 保留。
- [同预算/缓存基准](docs/paired_budget_benchmark.md)：固定预算、冷/热缓存、失败分母和资源指标。
- [worker 权限策略](docs/worker_permissions.md)：网络/子进程默认拒绝的防御层及 OS 隔离边界。
- [参数不确定性](docs/parameter_uncertainty.md)：条件于当前样本的 bootstrap 稳定性。
- [验证后经验库](docs/experience_store.md)：脱敏、内容寻址、结构检索和清理。
- [建模概念库](docs/concept_library.md)：保存适用条件、实现方式和反例的可迁移搜索概念。
- [分级候选评估](docs/staged_evaluation.md)：预留最终确认预算的分阶段候选调度器。
- [增量编译缓存](docs/incremental_compile_cache.md)：只复用可追溯的中间计算产物。
- [模型判决书](docs/model_verdict.md)：把证据、反例、共同结论和未决缺口结构化输出。
- [文档中心](docs/README.md)：文档索引与测试说明。
- [安全策略](SECURITY.md)：API Key、公开仓库和部署安全要求。
- [贡献指南](CONTRIBUTING.md)：提交代码前的检查清单。

## 项目结构

```text
core/       数学编译、数据处理、建模和验证
web/        Flask 应用与交互式前端
extensions/ 可选的外部模型和报告扩展
tests/      自动化测试
scripts/    开发与维护脚本
docs/       面向使用者的文档
```

## 许可证

[MIT License](LICENSE)
