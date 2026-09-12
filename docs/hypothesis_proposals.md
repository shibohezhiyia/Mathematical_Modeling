# 候选机制提议：实验接口

本功能对应路线图 N03–N04 的第一版：把事实抽取与模型假设分开，提供类型化数学图、未知机制和版本化证据接口。它尚不是自动求解或 CEGIS 闭环。

## 网页使用

1. 在研究页面展开“本地小模型 / 外部 API · 事实抽取与候选机制提议”。
2. 配置已有的本地模型或外部 API 服务。
3. 按需要勾选“候选机制提议”。它默认关闭，与“事实关系抽取”独立。
4. 运行研究，在“候选机制提议”中查看假设、原语图、待发现机制和澄清问题。

提议入口只读取题面文字，不读取图片内容或原始数据表；最多增加一次模型请求。两项同时启用时，各自可能产生一次请求。没有配置服务、服务失败或模型输出非法时，不会用猜测替代既有事实和数值结果。外部 API 可能收费，运行前检查所用服务配置。

结果保存在本次运行的 `evidence/model_hypotheses.json`，产物清单中的角色是 `unverified_hypotheses`。这里的目录名称不表示数学结论已有证据；该文件与数值结论证据链分开管理，并随运行清理。

研究链路中的结构候选还会携带一个经过 `primitive_graph` 类型/义务检查的 JSON 草案。它包含节点、引用、量纲占位和未决机制义务，状态固定为 `type_checked_not_executed`；这只是动态结构合成的起点，后续仍需绑定题面来源、单位、数据、资源和求解器，并通过独立验证。候选图不包含 Python、回调或可执行代码。

每个 `unknown_mechanism` 草案现在还携带有限的 `allowed_operators`、搜索预算和待检验 `properties`。这些字段让后续编译器可以在小范围内进行结构变异，而不会把“未知”降级成任意表达式执行；它们仍是搜索空间声明，不能替代单位、边界、来源和数值验证。

## 接口边界

| 对象 | 当前能力 | 不代表什么 |
| --- | --- | --- |
| `ProblemContract` | 精确题面引文、不可变快照、版本和内容哈希 | 引文存在不证明其数学解释正确 |
| `HypothesisIR` | 原语连接、类型、形状、已知量纲和来源引用检查 | 类型合法不等于有解、稳定或符合现实 |
| `UnknownMechanism` | 输入、输出类型、候选算子与待检验性质 | 空节点不是可执行函数；潜变量没有被证实的物理含义 |
| `EvidenceLedger` | 不可变追加记录，绑定契约、候选和检查上下文 | 不接受模型直接提交验证等级；没有形式化证明后端 |
| `SearchController` | 单次请求的模型调用、候选数和节点数预算 | 不是 MCTS、自动修复器或进程沙箱 |

默认最多 3 个候选、每个 64 个节点，允许 0 个候选，不强凑模型数量。接口不接收任意 Python、SymPy 字符串代码或求解器权限。

## 开发接入

入口位于：

- [model_hypotheses.py](../core/model_hypotheses.py)：核心对象、类型检查与预算。
- [hypothesis_generator.py](../core/hypothesis_generator.py)：模型提议适配器与 `proposal_schema()`。
- [semantic_model_compiler.py](../core/semantic_model_compiler.py)：原有事实抽取接口，保持严格来源校验。

`HypothesisGenerator` 复用 `SemanticCompilerConfig` 和 `SemanticCompletionBackend`。可以注入 `CallableSemanticBackend` 测试或接入本地模型，不必调用外部服务。向 `MathModelingAssistant` 显式传入 `hypothesis_generator` 才会启用该阶段。

网页请求 `/api/research/run` 支持独立的布尔字段 `hypothesis_generation`，服务配置沿用 `semantic_provider`、`semantic_base_url`、`semantic_model_name` 和仅请求期使用的 `semantic_api_key`。不开启该字段时，不增加假设提议调用。

模型响应的根字段只有 `hypotheses` 和 `questions`。每个候选必须携带当前 `contract_hash`；模型提交 `verified`、`source`、证据、预算或契约修改等额外控制字段会被拒绝。

```json
{
  "id": "x",
  "op": "variable",
  "inputs": [],
  "type": {"dtype": "real", "shape": [], "dimensions": {"L": 1}},
  "attributes": {"name": "x", "role": "state"},
  "assumption_ids": ["a1"],
  "context_fact_ids": ["statement"]
}
```

上面仅是一个节点，不是完整候选。`dimensions: null` 表示单位未知，`dimensions: {}` 才表示无量纲；`shape: []` 表示标量。每个节点必须引用候选中声明的假设。`context_fact_ids` 仅表示相关题面上下文，不授权把该节点的数值、单位或机制解释当成事实。

`unknown_mechanism` 的 `attributes` 包含 `allowed_operators` 和 `properties`，后者一律是待检验性质。普通算子只能使用空属性对象，变量/参数使用 `name` 与 `role`；变量/参数还可提供经过校验的 `descriptor`，绑定值域、时间语义、可观测性和来源。常量仅使用有限数值 `value`。描述符不会绕过原有形状、量纲和来源校验。

题意存在多种解释时，可用 `SemanticHypothesisSet` 保存并列解释。每个解释绑定契约事实、列出未验证假设和待澄清问题，并只能处于 `candidate`、`user_confirmed` 或 `rejected` 状态；它不修改 `ProblemContract`，也不会给数学图或数值结果授予可信等级。

用户回答澄清问题时，调用 `ProblemContract.record_confirmation(question, answer)` 创建新的不可变契约版本。回答会以新的精确引文事实追加到题面，保留 `parent_hash`；旧版本的候选、缓存和证据不会被静默复用。设置 `hard_constraint=True` 才会把该回答列入硬约束事实，否则它只是可追溯的用户确认，仍需重新编译和验证。

网页研究结果会把每个后端返回的澄清问题渲染为回答框。问题也可以附带 `question_options`：每个问题最多显示 3 个带影响说明的建议选项；当模型只给出两个选项时，服务端会追加“暂时无法确定”，避免把未经支持的分支伪装成穷尽选择。点击只会把选项文字填入回答框，用户仍可修改或完全自由填写；选项不会绕过服务端契约确认。点击“写入题意并重新分析”时，`/api/research/clarify` 只接受当前结果中的问题索引，在服务端复核原契约、追加用户回答并返回新 revision；浏览器随后用新契约哈希和完整题面重新调用研究流程。哈希必须与当前会话保存的契约一致，题目文本也必须逐字匹配，否则拒绝重跑。API Key 不写入契约或会话。当前实现是安全的全流程重算，还没有做到只重算受该事实影响的候选子图。

`rank_questions()` 提供一个明确标注的初期启发式：按“涉及的假设数 × 未涉及的假设数 ÷ 声明获取成本”排序，最多返回三个问题。它不是概率信息增益；用户不回答时系统必须保留全部解释。

交互层可以传入 `options_by_question`，为问题提供 2–3 个选项。每个选项包含 `id`、`label`、受影响的 `hypothesis_ids`，以及可选的 `impact` 说明。选项只生成用户确认用的契约修订入口，不会自动修改事实、候选图或数值证据；选项引用未知假设或超过 3 个时会在排序阶段拒绝。

表达式依赖必须无环；动力学反馈通过状态符号和方程表达，不把真实系统强制建成因果 DAG。微分、积分、除法和对数仍保留边界或定义域验证义务。

## 证据与失败处理

通过检查的候选始终保持 `authority: hypothesis_only` 和 `execution_status: not_executed`。接口只写入检查范围明确的 `type_check` 记录，不把模型自报的成功写入数值证据。

应用代码今后记录 `numerical_test` 或 `counterexample` 时，必须给出输入哈希、评价器版本、定义域和随机种子。读取这类记录须显式传入相同检查上下文；契约、候选、数据或检查范围变化时不能复用旧记录。

模型响应限制为 256 KB，并检查 JSON 重复键、深度、节点数、形状、算子参数和非有限值。失败返回结构化错误码，不保存原始响应、密钥或完整供应商异常。失败的请求仍扣减模型调用预算。

提议器会从当前 `ProblemContract` 的 `[用户确认] ...\n回答：...` 事实中提取已回答问题，并用规范化文本抑制完全重复的问题；被抑制的问题保存在 `suppressed_questions` 元数据中，便于审计。这不是语义相似度模型，换一种说法仍可能需要用户判断，不能把未重复提问误解为问题已解决。

对于文本相似但不完全相同的问题，提议器只记录 `possible_repeated_questions` 提示，不自动删除；网页和报告会明确显示“请人工判断”。相似度阈值是防重复的工程筛查，不是语义等价证明。

这些边界是非执行提议接口的防护，不是操作系统级安全保证。自动数值编译、真实数据角色绑定、反例驱动修复、跨运行搜索预算和最终测试集冻结仍在后续阶段实现。

已有、经过核验的 ODE/NLP 契约现已接入 [受控数值执行](solver_runtime.md)。这不改变本提议入口的 `not_executed` 状态，也不表示未知机制已被自动补齐。

## 测试

```bash
python -m pytest tests/test_model_hypotheses.py tests/test_semantic_model_compiler.py tests/test_research_web_api.py -q
```

测试使用可控替身，不会验证真实小模型或外部 API 在新题上的生成质量。该质量仍需固定预算的未见题基准。
