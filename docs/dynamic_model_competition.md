# 动态模型竞争

`core.model_competition.compete_dynamic_families` 是跨模型族的执行编排层。它不把模型注册表当作能力边界，也不把不同含义的分数直接混成一个总分。

## 执行顺序

```text
候选族声明
  → 各自 compile/evaluate/diagnose/patch/replay
  → 有界 CEGIS 与反例记录
  → 显式 metric_builder/评估指标
  → comparison_group 内 Pareto 比较
  → 保守判决与未决项
```

每个族需要提供 `ModelFamilyAdapter`、候选列表和案例列表。只有评估结果同时给出 `predictions` 与四个比较轴（`validation_loss`、`complexity`、`constraint_violation`、`instability`）时，候选才进入数值比较；缺少任何轴会保留为 `not_assessed`，不会用零、平均值或模型建议补齐。

`comparison_group` 必须由调用方声明。同组才允许比较，例如同一衰减轨迹的多个 ODE 候选。ODE 轨迹误差可以作为 `validation_loss`，优化模型只有在提供独立 `expected_objective` 时才会生成 `validation_loss`；优化目标值本身不能未经方向和任务语义转换就与 ODE RMSE 比较。

## JSON 入口

`POST /api/research/dynamic-compile`（或 Python 的 `compile_and_execute_model`）支持 `kind=dynamic_competition`。当前可执行族为：

主研究入口 `POST /api/research/run` 也接受同一个 `dynamic_contract`。只要请求体提供
该 typed 合同，主助手会自动启用动态竞争（无需额外开关）；同步、异步线程和跨进程
worker 使用同一编译路径。没有合同时保持普通研究流程，只对可识别的原始记录模式做受限归纳，
不会仅凭自然语言臆造观测或验证案例。
网页首页的“高级：动态模型竞争合同”输入框会做 JSON 对象校验并原样提交该合同；
它是专家模式入口，空白时不改变普通用户流程。

`dynamic_contract` 只用于自定义候选竞争。普通用户不提供合同时，主研究流程会对已上传的
原始记录尝试 `automatic_modeling`：支持的代数、动力学、资源优化和两表关系可直接形成
模型与答案，歧义或缺少预测查询时返回明确缺失项。该受限入口不是任意题面的完整编译器；
支持范围和开发基准见 [当前能力与证据状态](current_capability_evidence.md)。

- `ode`：低维多项式 RHS、受限时滞/事件/单位契约和 SciPy 积分器；
- `linear_program`、`mixed_integer_linear_program`、`quadratic_program`：已验证优化合同；
- `multitable`：键、粒度和聚合检查，尚未提供统一预测指标时会保持未决。
- `pde_find`、`ude`、`ude_neural`、`ude_joint`、`ude_stiff`、`llm_sr`：通过类型化外部方法
  CEGIS；必须在案例中提供独立 `max_metric`，不能将“执行成功”当成模型正确。
- `gnn`：将 JSON 行记录转换为有界表格交互筛查，支持时间窗口/实体隔离参数并输出
  预测 RMSE 与边稳定性；它是预测模型族，不是因果发现。`external_method` 可作为
  `pde_find/ude/ude_neural/ude_joint/ude_stiff/llm_sr` 的显式路由别名。

示意：

```json
{
  "kind": "dynamic_competition",
  "payload": {
    "families": [
      {
        "family": "ode",
        "comparison_group": "decay",
        "candidates": [
          {"id":"a", "state_dim":1, "basis":["linear"], "coefficients":[[-0.5]]}
        ],
        "cases": [
          {"times":[0,0.5,1], "initial":[1], "observations":[[1],[0.7788],[0.6065]]}
        ]
      }
    ],
    "cegis_config": {"max_rounds": 8, "max_candidates": 32}
  }
}
```

## 证据边界

Pareto 非支配不是批准证书；CEGIS 通过只表示在声明案例上未被反驳。资源失败、编译失败、指标缺失和未确认的模型留在分母中。当前实现仍不声称自动发现任意 ODE/PDE、跨领域可比目标、全局稳定性、GNN 因果结构或现实开放式建模质量；这些声明分别需要结构/来源留出、独立参考求解或解析性质、因果识别设计以及平台级资源/隔离证据。专家评分只用于开放式质量的可选补充。

时间序列结构筛查可通过 `/api/research/causal-discovery` 的 `time_lags` 参数调用：
系统比较自回归基线与来源滞后项，以时间尾部误差增益和块 bootstrap 稳定性筛选
有向滞后交互。它可以为动态图/GNN 候选缩小搜索空间，但输出保留
`screened_temporal_predictive` 标记，不能直接当作因果边或数学证明。
