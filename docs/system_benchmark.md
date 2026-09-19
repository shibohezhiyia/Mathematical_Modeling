# 当前系统与旧版基线对照

`core.system_benchmark.compare_frozen_systems` 在同一训练/测试切分上运行：

1. 冻结的 Ridge + 中位数填补 + 标准化基线；
2. 当前 `ModelingEngine`，包含预处理、交叉验证、结构化模型组合和候选选择。

两者只共享训练特征和训练标签，测试标签仅交给独立评分函数。结果保留
`baseline`、`current_system`、最佳模型、运行耗时和预测摘要，失败返回
`not_assessed`，不会把失败当成低分。

## 公开数据实跑

运行：

```powershell
$env:PYTHONPATH='.'
python scripts/run_external_data_benchmark.py --include-current-system `
  --output data/reports/external_data_system_comparison.json
```

2026-09-11 的 6 个 UCI 案例全部完成。当前系统相对 Ridge 的任务级平均
RMSE 变化为 **-0.237916**；逐任务效果为：

```text
[-0.163431, 0.009186, -0.613776, -0.214250, 0.176920, -0.622147]
```

这证明“当前系统确实进入了同一冻结切分的数值对照”，但仍是公开标签的
描述性证据，不是开放式数学建模质量或专家评分。它可以支持该批公开任务范围内的
算法比较；更广的结构发现与建模结论应另用自动真值和结构/来源留出确认，外部盲评仅作可选补充。

本次运行对应的冻结摘要为
`d87704112e75060d05f9b517d494debac8faf99561579c66c3316a9e026c287d`，机器可读
记录见 `examples/system_comparison_20260911.json`。

## 统一动态编译入口

`POST /api/research/dynamic-compile` 接受 `problem_statement`、
`primitive_graph`、`ode_cegis`、`optimization_cegis`、`multitable_cegis`、
`dynamic_competition` 和 `external_method` 八类类型化 JSON 合同。
`dynamic_competition` 会先执行每个声明模型族的 CEGIS，再在显式比较组内做
Pareto 比较；指标缺失时保留为未决，不把不同含义的目标值直接混比。
`problem_statement` 会先将任意题面
编译为问题分析、机理 IR、结构候选和缺失条件；其中已通过确定性解析与单位
核验的关系直接进入现有求解器，未闭合关系保留为可执行前的明确输入门。
携带受支持的原始 records 附件时，它还会自行识别代数观测、时间—状态观测、
资源/容量表或事实/维表模式，完成有界候选选择和求解；不要求调用方先写模型候选。
普通研究页面也会对已上传表格尝试同一路径。缺少查询点、表角色或连接口径时返回
`needs_input`，不把候选提议冒充执行结果。
所有类型均禁止
`source/code/python/command/path/module/import` 字段。它只是受限后端路由，
不允许模型提交 Python 源码，也不把候选执行结果自动升级为现实正确性。

最小调用示例：

```json
{
  "kind": "problem_statement",
  "payload": {
    "problem": "研究一个带约束的资源配置问题，并说明需要哪些证据验证模型。",
    "candidate_budget": 3
  }
}
```

返回的 `status=needs_input` 是严格的输入门，不是失败：它表示题面已经
完成结构化编译，但仍缺少可执行的变量、单位、目标或约束。补充经核验的
`ir_override` 或 `contracts` 后可在同一个入口继续执行；系统不会把语言模型
猜测直接当作数值结果。

## 当前仍需补充证据的项目

- 冻结的自动真值任务上完成新版/旧版同预算配对比较，并报告结构组层面的效应大小与区间；
- 若要额外声称 `conditional_real_unseen` 或开放式论文质量，再补充项目外封存评分或盲评；
- 非规则网格 PDE、任意控制/刚性混合系统、完整动态图因果发现；
- 由部署方提供并保护的 Windows/Linux OS 隔离证明和实机攻击测试；
- 四层不确定性的独立真值校准。当前新增的
  `assess_four_layer_uncertainty` 是条件情景方差分解，不是后验概率。
