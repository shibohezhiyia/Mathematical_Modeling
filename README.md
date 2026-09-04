# 建模比赛智能分析引擎

面向数学建模比赛的题目驱动数学论证助手。可同时读取多个附件数据集，自动发现表间关系、分析跨表变量交互、拆解题目任务，并执行预测、聚类、因果、动力学或综合评价。核心产物是机器可检查的数学模型规范、假设账本、反证记录和论证图；论文写作不参与求解，只能在全部验证结束后作为可选 API 末端。底层同时提供完整的 AutoML 能力。

## 项目结构

```
Mathematical Modeling/
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── data_module.py             # 数据模块（加载/识别/清洗）
│   ├── missing_engine.py          # 缺失值智能分析引擎
│   ├── auto_pipeline.py           # 自动缺失处理流程
│   ├── performance_scheduler.py   # 自动性能调度器
│   ├── accelerators.py            # GPU/多进程加速层
│   ├── parallel_modeling.py       # 并行建模引擎（旧）
│   ├── modeling_engine.py         # 建模引擎 V2（新核心）⭐
│   ├── modeling_assistant.py      # 题目驱动、多数据集研究编排器 ⭐
│   ├── mathematical_reasoning.py  # 数学规范、量纲、证据包与论证图 ⭐
│   ├── mechanistic_modeling.py    # 零数据数学 IR、通用算子与安全求解 ⭐
│   ├── four_layer_modeling.py     # 题意契约→统一 IR→结构选解→独立审计 ⭐
│   ├── universal_math_solvers.py  # 通用关系验证器与结构化求解后端 ⭐
│   ├── semantic_model_compiler.py # 受约束的小模型/API语义编译层 ⭐
│   ├── table_transformer.py       # 声明式表变换注册表、组合执行与审计 ⭐
│   ├── integrated_pipeline.py     # 集成流水线 V2
│   └── workspace_manager.py       # 工作空间管理器（磁盘隔离）
├── utils/                         # 工具函数
│   ├── __init__.py
│   └── helpers.py
├── tests/                         # 完整自动化回归、边界与压力测试
├── workspace/                     # 工作目录（临时/缓存/报告/数据）
├── data/                          # 数据目录
├── demo.py
├── demo_missing.py
├── requirements.txt
├── .env.example                   # 本地/发布环境变量模板
├── SECURITY.md                    # 公开部署安全基线
├── LICENSE                        # MIT 许可证
└── README.md
```

---

## 模块零：题目驱动的多数据集研究助手 ⭐

在“数学建模研究助手”中粘贴完整题目，点击“一键完成研究”。附件数据是可选输入：有附件时进入多数据集分析，无附件时进入纯题面数学 IR 与机理建模。系统会自动完成：

- 构建带类型的数学模型规范：记录观测、目标、处理、结果、时间、控制候选等变量角色，以及单位、量纲、可用时点、目标、约束和缺失条件；
- 将零数据题面编译为 `mathmodel.mechanistic-ir/v2`：抽取带原文来源的实体、显式量、关系、目标、约束、初边值条件，并组合动力学、几何、事件、网络、随机、优化、反演和鲁棒决策算子；核心层不包含赛题名称、对象编号或故事背景分支；
- 自然语言只生成候选 IR，绝不直接执行。数学结构目录当前识别 27 类，其中 20 类具有经过验证的通用后端；只有结构化关系通过符号白名单、有限数值、完整条件、单位覆盖、静态量纲和资源上限检查后才进入求解器；
- 对显式公式执行不求值的静态量纲检查；未知单位标记为 `not_assessed`，等式两侧或加减项量纲矛盾直接使规范变为 `invalid`；
- 建立关键假设账本，区分可检验、部分可检验、领域限定和原则上不可由当前数据检验的假设；审计结果会反向更新假设状态；
- 为每类任务建立零模型、竞争模型、求解器要求与反证路线，记录模型竞赛而不是只展示胜出算法；
- 对显式声明的连续线性规划安全编译代数表达式并调用 HiGHS；同时复算原始可行性、KKT 与互补松弛残差，搜索近优解变量范围，执行 30 次目标系数扰动，并从场景方案中返回最小最大遗憾稳健候选。非线性、整数、参数不明确或缺约束时拒绝自动求解；
- 把数据指纹、假设、数值证书、审计项和结论连接成有向论证图；任何没有证据入边的结论都会自动降级为 `undetermined`；
- 将结论分为“数学上已验证”“在明确假设下成立”“经验支持”“当前反证”和“不可判定”，不再用一个模糊总分掩盖局部失败；
- 拆解题型、小问、目标、约束和候选数学模型，并生成带输入、输出和上游依赖的多子问题执行图；
- 识别事实表、维度表、关联键及一对一/一对多/多对多关系；
- 支持字段异名和“地区+年份”等复合键，重复列、空表及嵌套对象会自动规范化或隔离；
- 对明细表先按实体键聚合，再计算跨表交互，避免笛卡尔积卡死；
- 关系候选列只构建一次值域/频数草图并跨数据集对复用，避免多附件下重复规范化；
- 字段角色使用语义硬门：编码、ID、编号等字段即使存成整数且大量重复，也只作为关联、分组或切分元数据，不进入相关、PCA、趋势和预测目标候选；
- 按变量类型选择 Spearman、非线性互信息、相关比 η² 或校正 Cramér's V，并进行 FDR 多重检验校正；
- FDR 在全部已检验假设上统一执行，并补充控制其他数值变量后的条件相关与分块稳定性；
- 对具有父子层级的可加总数据，按题面提及字段和已验证函数依赖自动绑定“时间 × 上层维度 × 下层维度”（如区域—站点、产品族—产品），输出分位数、变异系数、HHI、头部份额和周期效应；层级联动先移除趋势与周期效应，再统一执行 BH-FDR，未通过校正的效应只能列为探索结果；
- 从题目识别一个或多个目标列，分别建模，并把其他附件中的聚合特征加入预测模型；
- 当题目要求“日 × 品类/地区/实体”总量时，先在完整的有界分析帧上按目标粒度聚合，再验证季节朴素基线与趋势模型；绝不先随机抽取交易明细再冒充总量预测，并为每个组输出末段留出误差和 90% 区间；
- 超大时序明细上传后会生成 `mathmodel.research-frame/v2` 研究缓存：可加总事实表对全部原始行执行精确“日×实体”预聚合并保留原始行数、全时段和总量；不能保证总量的覆盖样本会标记 `aggregation_complete=false`，预测、层级统计和决策阶段会拒绝拿它计算总量；
- 同一道题包含多个输出粒度时按子问题分别编译，例如同时生成品类七日预测与单品单日预测，不再让整道题共享一个错误粒度；显式日期区间会成为预测索引，早于数据末日的历史区间不会被冒充未来结果；
- 时间任务使用 point-in-time 跨表联接，禁止未来明细进入历史样本；
- 基于 OOF 残差和跨折波动选择反馈调参方向；参数只在开发集搜索，只有独立确认集收益为正且成对重采样改善概率不低于 80% 才采用；
- 自动执行结果可信度审计：检查验证隔离、目标泄漏、简单基线、验证结果置乱、bootstrap 指标区间、跨折稳定性、分群误差、分布漂移、输入扰动敏感性和单特征依赖；
- 构造由不同算法族组成的“近优模型集合”，比较其 OOF 预测是否一致；分数相近但结论相反时直接给出失败证据；
- 检测到客户/设备/企业等重复实体时自动使用 GroupKFold/实体隔离留出；实体键只作为切分元数据，不进入模型；
- 自动编码使用不读取目标值的频数编码，高基数目标编码会被降级；默认不在交叉验证前执行监督式特征筛选，防止验证折提前看到答案；
- 自动执行时间顺序验证、无监督聚类或熵权 TOPSIS 综合评价；聚类使用多随机种子 ARI 稳定性复算，排名使用 100 次权重扰动和逐一删指标复算；
- 对最多 5 个高维数值数据集自动执行 PCA 潜在结构、载荷解释和稳健重构异常检测，并通过分半子空间与输入扰动验证结果是否稳定；
- 对动力学题使用积分弱形式稀疏辨识候选方程，以时间末段外推、方程项稳定性和残差记忆进行反证；
- 当题目显式声明处理变量与结果变量时，执行交叉拟合正交化处理效应估计，并审计重叠性、跨折稳定性、安慰剂置乱和不可检验的混杂假设；
- 回归模型生成有限样本校正的保序预测区间；时间任务使用固定近期权重并明确提示交换性保证的边界；
- 综合评价除 TOPSIS 外同时输出不依赖权重的 Pareto 非支配集，避免把偏好依赖的唯一名次伪装成客观事实；
- 将分组需求预测、损耗、成本和历史价格边界编译为通用多选 MILP，输出带预测区间传播的补货候选；只有价格效应在全样本及前后分段均为负且变化范围充分时才允许调价，否则锁定参考价格，并明确不把观察性关联写成因果最优；
- 当售价、时间、成本和分组能够验证对齐时，单独检验成本加成率—目标总量关系：先按“日期×组”对齐，移除趋势与周期效应，再同时要求 BH-FDR 与分半方向稳定；检验未显著也是可报告的负结果，不会被包装成存在关系；
- 层级有限动作编译器不包含领域词汇：任意候选动作、决策单元、激活数量上下界、上层覆盖需求和效用会被编译为两阶段 MILP，先最小化加权未满足量，再在最小缺口内最大化效用。单品选择、站点启用、设施配置等只是同一数学原语的语义绑定实例；
- 同一通用有限动作还可绑定任意有限情景收益与概率，第二阶段会精确编译“期望效用 + 下尾 CVaR”，并独立复算每个情景结果、最坏结果、CVaR 和求解器目标残差；预测适配器把点预测与区间端点转换为数量动作和压力情景，但未校准的权重只生成备选方案，题目没有风险偏好时不会擅自覆盖名义最优；
- 把“还需采集哪些数据及其作用”识别为独立的数据需求审计任务，根据题目契约与现有字段角色输出带优先级、采集设计和可支持任务的数据缺口，不再误分类为优化求解；
- 对网络题、不确定性题和时序题执行有边界的专项计算，也可注册领域分析器扩展优化、微分方程等题型；
- 输出数据关系图、交互强度图、潜在结构/异常图、模型验证图，以及严格分区、带版本和 SHA-256 校验的运行产物清单。

### 通用表数据编译层

“多表数据管理 → 通用数据变换流水线”不是若干固定按钮，而是一份可组合的声明式协议。当前注册了 19 类题目无关操作：字段选择/删除/重命名、条件筛选、稳定排序、去重、类型转换、分组缺失填补、安全公式派生、多键多指标汇总、加权均值/分位数/占比、透视、宽长表转换、时间特征、周期重采样、面板窗口、数值变换、分箱、类别编码，以及坐标到距离边表。异名复合键关联独立支持键类型对齐、一对一/一对多关系验证和连接膨胀预估。

界面可以根据题目中的处理目标和当前字段画像生成候选流水线；候选只会载入编辑器，不会直接修改数据。点击“预览”后，每一步都会报告输入/输出规模、增删字段、耗时和风险提示；只有整条流水线成功时“应用”才会事务式替换当前建模表。中小表保留最多 3 个、合计不超过 128MB 的撤销快照；超过 64MB 的单表只保留审计，避免为了撤销把大表内存翻倍。透视、宽转长、独热编码、距离边和多对多关联均有单元格、列数或膨胀预算，超限时要求先筛选或聚合。

Python 中使用同一契约：

```python
from core import TableTransformationEngine

result = TableTransformationEngine().execute(df, [
    {"operation": "derive_columns", "params": {
        "expressions": {"利润": "收入 - 成本", "利润率": "利润 / 收入"}
    }},
    {"operation": "aggregate", "params": {
        "group_by": ["地区", "月份"],
        "aggregations": [
            {"column": "利润", "function": "sum", "output": "利润合计"}
        ],
    }},
])
print(result.data)
print(result.audit)
```

派生公式由受限 AST 解释器逐节点计算，不调用 `eval`；含空格或符号的字段可写成 `col("单位成本(元)")`。已提交的处理结果会以“当前处理结果（优先）”加入自动研究数据集，同时保留原始附件供跨表核验。

### 数学数据编译与结论反证

通用变换层之上新增“数学数据编译器”。它不会把每种清洗方案打成一个含混总分，而是先推断并显式记录一行数据的实体×时间粒度、目标估计对象、字段单位、可加/不可加语义、候选主键和经验函数依赖，再为不同数学问题编译相互独立的视图：原始观测、缺失敏感性、严格历史时序、实体层估计对象和时间粒度估计对象。改变粒度的视图必须通过输出键唯一性、可加总量守恒、有限值和目标产生时点审计，失败的流水线会被阻断，不能进入模型。

对每个数值解释变量，系统会在完整样本、中位数填补、缩尾、组内残差、组间聚合和时间聚合视图中重新估计秩相关；目标列永不插补，可加量在时间聚合时求和，率/价格类取均值。总体预测变量检验和同一关系的替代视图分别执行 BH-FDR，并要求效应量阈值与 Spearman 方向的 95% Fisher-z 近似区间同时通过。只有总体方向和反向视图都通过这些门，才将“存在稳定总体规律”标成“当前反证/拒绝采用”；纯噪声、弱效应或区间跨零只会报告为证据不足。通过该审计只表示在已检验视图下具有经验稳定性，不代表因果识别成立。

多表自动研究使用同一编译器生成跨表数学契约，不实际物化连接：先按技术键、时间键和维度键的优先级渐进搜索，发现高覆盖单键后停止组合爆炸，否则继续比较二/三字段复合键；数值字符串、整数/浮点键和字符串日期/时间戳会成对规范化。高基数键在两表独立采样时使用有门槛的捕获—再捕获重叠校正，避免把真实复合键误判为低覆盖；样本基数结论一律标记为需要全表复审。系统估计一对一、一对多、多对一、多对多、连接膨胀和安全的特征补充方向。唯一侧属性可以向多侧补充特征，但唯一侧可加量会被复制，不能在连接后直接求总量；原始多对多连接会进入证据包的“当前反证”并被禁止，要求先按估计对象粒度聚合。双方都有时间字段但连接键不含有效时间时，契约会强制要求 point-in-time 对齐。

字段语义采用“双通道”而不是依赖固定中文/英文列名：默认启发式负责零配置启动；专业字段、缩写和异名跨表键可通过受校验的 `semantic_hints` 显式绑定。不同表中的字段只要共享 `semantic_id` 即可成为候选连接键；没有显式绑定时，系统会在角色相容的字段之间按值域重叠提出异名候选，但该连接只能标为 `restricted`，必须经过全表基数和业务含义复审。错误字段、非数值度量、非法角色或非法可加性会直接拒绝，外部小模型/API 因此只能提出语义契约，不能绕过确定性审计。

```python
result = compiler.compile(
    frame,
    problem="评估处理量对结果的影响",
    semantic_hints={
        "target": "outcome_v2",
        "grain": ["sample_no", "event_at"],
        "columns": {
            "sample_no": {"role": "technical_id", "semantic_id": "sample"},
            "event_at": {"role": "time", "semantic_id": "event_time"},
            "dose_x": {"role": "measure", "unit": "mg", "additivity": "additive"},
            "outcome_v2": {"role": "measure", "unit": "%", "additivity": "non_additive"},
        },
    },
)
```

多表调用使用 `{"datasets": {"表名": <单表 semantic_hints>}}`；网页接口 `/api/data/math-compile` 接收同名 `semantic_hints` 字段。协议版本为 `mathmodel.data-compilation/v2`，每个字段同时保留 `semantic_source=heuristic|explicit_hint`，便于追溯哪些语义来自自动推断、哪些来自用户或模型确认。若只有一张表声明 `target`，它会自动成为主估计对象；多张表同时声明目标时必须用 `primary_dataset` 或“表名.字段名”消除歧义。

网页的“数学多视图审计”会展示数据契约、候选视图、守恒/泄漏检查和结论翻转；可采用视图只能载入通用流水线编辑器，仍需预览后人工提交。超过分析预算的大表只在最多 50,000 行的均匀覆盖样本上编译候选并标为“限定采用”，不会为审计复制整张大表；真正应用时再对完整数据事务式执行并复查不变量。自动研究还会生成 `evidence/mathematical_data_compilation.json` 和多视图效应图，并把反转关系写入论证图的禁止结论集合。

### 多维交互图形工作台

上传或选中数据后，进入“数据概览”即可使用交互图形工作台，不需要先训练模型。图形通道可分别绑定 X、Y、颜色、点大小、分面、播放维度和最多 8 个悬浮详情字段；支持散点、折线、面积、柱状和平行坐标。连续值上下界、连续轴分箱、最大图元、透明度、点大小、分面窗口和播放帧均可用滑块调整，图内还保留 X/Y 缩放滑块、框选缩放、恢复和下载图片功能。

浏览器不会直接接收整张大表。`InteractiveVisualizationCompiler` 先校验字段与图形语义，向量化执行连续/分类筛选，再在最多 200,000 行扫描预算内聚合，最终最多输出 15,000 个图元；超出预算时使用确定性覆盖样本，并在图下明确展示源行数、筛选行数、扫描范围、聚合方式和“样本聚合不是精确总量”警告。接口协议为 `mathmodel.interactive-visualization/v2`：`GET /api/visualization/explore/schema` 会把字段区分为度量、时间、低基数维度、文本标签和编码/标识符，避免把商品编码、分类编码之类的整数伪装成连续变量。工作台可直接切换已上传文件或 Sheet，并按“类别构成、分组比较、数值关系、时间趋势”选择分析目标；缺少数值度量时会停用散点、趋势和平行坐标。X/Y 重复、颜色/分面重复、计数散点以及对近唯一名称计数等无信息配置会在前后端同时拒绝。高基数颜色系列只保留高频组，其余在聚合前完整合并为“其他”，不会截断记录或改变聚合结果。`POST /api/visualization/explore/data` 返回可审计图元。静态 ECharts 5.5.0 已固定在 `web/static/vendor/`，离线环境不再依赖公共 CDN。

```json
{
  "chart_type": "scatter",
  "encodings": {
    "x": "投入", "y": "产出", "color": "地区",
    "size": "规模", "facet": "批次", "animation": "月份",
    "tooltip": ["方案", "成本"]
  },
  "filters": [
    {"field": "投入", "kind": "range", "min": 10, "max": 100},
    {"field": "地区", "kind": "in", "values": ["东部", "西部"]}
  ],
  "aggregation": {"function": "none", "time_unit": "none", "bins": 20},
  "max_points": 5000
}
```

### 核心产物与论文边界

`evidence/evidence_bundle.json` 是主产物。每条结论包含证据等级、处置状态、数据来源、支持证据、反证证据、依赖假设、数值自洽证书、适用范围、失效条件和下一步补强方法。任务未求解、关键角色不明确或硬性检查失败时，系统必须输出不可判定或当前反证，不能用“推荐某方法”冒充已经得到结果。

每次运行使用一个独立目录；未指定 `output_dir` 时会自动创建带 UTC 时间和随机后缀的运行目录，避免覆盖上一次结果。目录协议固定为：

```text
<run_root>/
├── artifact_manifest.json              # 唯一权威索引、格式版本、状态、SHA-256
├── evidence/                           # 可复算/可审计的机器产物，必须保留
│   ├── mathematical_model_spec.json
│   ├── evidence_bundle.json
│   ├── mathematical_data_compilation.json # 估计对象、多视图不变量与方向翻转审计（有数据时）
│   ├── mechanistic_model.json           # 零数据题面的通用数学 IR 与算子图（如适用）
│   ├── 01_semantic_contract.json        # 第一层：题意、角色、假设与来源
│   ├── 02_unified_mathematical_ir.json  # 第二层：归一化数学结构
│   ├── 03_solver_plan.json              # 第三层：选解、预算与降级计划
│   ├── 04_independent_audit.json        # 第四层：独立复算与风险标记
│   └── research_result.json
├── reports/
│   └── mathematical_argument.md        # 论证摘要，不是自动论文
├── charts/                              # 按阶段编号的 PNG
├── cache/
│   └── cache_manifest.json             # 可删除；缓存也有独立版本与清单
├── temp/                                # 原子写入暂存；可删除
└── logs/                                # 本次运行日志的预留目录
```

`artifact_manifest.json` 使用 `mathmodel.run-artifacts/v1` 协议，记录每个正式产物的相对路径、媒体类型、格式版本、字节数、SHA-256、是否必需和是否可删除。只有 `cache/` 与 `temp/` 被标记为安全清理类别；清理 API 会逐文件校验目录边界，不能删除 `evidence/`、`reports/` 或 `charts/`。如需删除全部结果，直接删除对应的单个 `<run_root>`，不要从共享报告目录中按扩展名筛选。

Python 中可预览或清理某次运行的缓存：

```python
from core import RunArtifactManager

artifacts = RunArtifactManager.open_existing("data/reports/my_study")
print(artifacts.clear_cache(dry_run=True))  # 只列出将删除的缓存
print(artifacts.clear_cache())              # 证据、报告和图表不受影响
```

通用计算缓存统一采用 `cache_manifest.json + entries/ + metadata/ + temp/` 布局；逻辑键只写入元数据，物理文件名固定为 SHA-256，既避免一个目录堆积大量文件，也不会因外部键包含路径字符而越出缓存目录。

`writing_contract` 默认 `enabled=false`。最终确需生成论文时，写作 API 只能改写 `allowed_claim_ids`，并必须保留假设和适用边界；`rejected` 与 `unresolved` 结论进入 `prohibited_claim_ids`，禁止被写成肯定事实。写作 API 不允许计算新数字、补造公式或改变数学结论。

显式线性优化示例：

```text
建立优化模型；决策变量=x,y；最小化 3*x + 2*y；
约束 x + y >= 10；x >= 0；y >= 0
```

这类完整表达会进入安全符号编译器；`3*x` 等表达式只解析为系数，不通过 `eval` 执行。系统返回名义最优解、最大约束违反、KKT 数值证书、近优解范围、参数扰动稳定性和最小最大遗憾候选。默认 5% 系数扰动只用于压力测试；在题目确认不确定集合前，稳健候选不会自动替换名义解。只写“优化成本”则保持 `needs_input`。

### 无附件题目与通用数学 IR

只提供题面即可运行，`datasets` 可以省略：

```python
from core import run_modeling_study

result = run_modeling_study(
    problem="建立状态变量的动力学方程，在给定约束下优化控制策略",
    output_dir="data/reports/no_dataset_study",
)
mechanistic = result["specialized_results"]["mechanistic_model"]
print(mechanistic["operator_graph"])
print(mechanistic["compiler_plan"]["blocked_by"])
```

系统的普适性来自统一 IR 和可组合算子，不来自题目模板。主执行链已经拆成四个可分别审计的层次：

1. `semantic_contract`：只保存实体、变量角色、目标、约束、假设与原文来源；
2. `mathematical_ir`：把领域适配结果归一为初值问题、有界非线性规划、连续事件测度或仿真驱动规划等数学形式；
3. `solver_plan`：只依据数学形式选算法，给每个节点设置变量数、评估次数和软墙钟预算，并隔离单节点失败；矩阵/图规模和迭代数是硬上限，墙钟值只有在后端提供原生时限时才是硬限制；
4. `independent_audit`：不信任求解器自报标签，重新检查有限性、约束违反、容差收敛、区间并集、敏感性与全局最优证书。

当前 20 类可执行结构包括线性方程、括区间多项式根、线性最小二乘、ODE 初值问题、连续事件测度、LP、MILP、凸 QP、一般有界 NLP、仿真驱动优化、多目标线性 Pareto、显式情景鲁棒/随机规划、有限时域动态规划、最短路、最大流、最小费用流、二部匹配、马尔可夫链和加权样本期望。PDE、DAE、边值问题、非线性方程组、非线性标定、最优控制和离散事件仿真目前只识别、不伪装成已执行。

```python
mechanistic = result["specialized_results"]["mechanistic_model"]
pipeline = mechanistic["four_layer_pipeline"]
print(pipeline["semantic_contract"]["unresolved_bindings"])
print(pipeline["mathematical_ir"]["nodes"])
print(pipeline["solver_plan"]["nodes"])
print(pipeline["independent_audit"]["result_audits"])
```

### 可选语义模型编译

网页中的“本地小模型 / 外部 API 辅助题面编译”可接 Ollama、DeepSeek 官方 API、本机 OpenAI 兼容服务或其他 HTTPS OpenAI 兼容 API。该模型位于题面抽取和统一 IR 之间，只提出候选契约：每个必需字段必须引用题面中的精确原句，数值必须能回溯到对应引文，之后还要通过确定性模式、单位、维度、规模和求解前校验。模型写入的 `parse_status`、来源或“已验证”声明都会被删除，API 失败则退回确定性编译。

```python
from core import MathModelingAssistant, SemanticCompilerConfig, SemanticModelCompiler

semantic_compiler = SemanticModelCompiler(SemanticCompilerConfig(
    provider="ollama",
    base_url="http://localhost:11434",
    model_name="qwen2.5:3b",
))
result = MathModelingAssistant(
    output_dir="data/reports/semantic-study",
    semantic_compiler=semantic_compiler,
).run(problem="完整题面……", datasets={})
```

DeepSeek 可在网页中直接选择并点击“测试 API”，也可以用代码接入：

```python
semantic_compiler = SemanticModelCompiler(SemanticCompilerConfig(
    provider="deepseek",
    base_url="https://api.deepseek.com",
    model_name="deepseek-v4-pro",  # 速度优先可改为 deepseek-v4-flash
    api_key="运行时传入的密钥",
))
```

网页“AI 智能分析”支持多模态输入：可附加最多 5 张 PNG/JPEG/WEBP/GIF 图片（单张不超过 6 MB、总计不超过 20 MB）。图片仅以内存 data URL 随本次请求发送，不写入会话、缓存或报告；要解读题目截图、图表或几何示意图，请选择 `deepseek-v4-flash-vision-exp`（或其他支持 OpenAI `image_url` 消息格式的模型）。

其他外部 API 使用 `provider="openai_compatible"`、HTTPS `base_url` 和运行时 `api_key`；密钥不会写入会话、报告、证据文件或模型配置摘要。DeepSeek 提供商只允许官方 `api.deepseek.com` 地址。本地服务只能使用回环地址，外部服务会拒绝 HTTP、重定向以及解析到私有/保留地址的主机。若应用已经内置了一个小模型，可用 `CallableSemanticBackend` 注入返回 JSON 的本地生成函数，不需要经过网络。

### 公开部署安全基线

默认启动仍是本机开发模式。准备公开仓库或公网部署时，先复制 `.env.example` 并设置随机的 `FLASK_SECRET_KEY`；公网服务还要设置独立的 `ADMIN_TOKEN` 和 `PUBLIC_MODE=1`。公网模式会关闭 debug、限制请求体大小、对训练/上传/研究接口限流，并要求管理接口携带 `X-Admin-Token`。请将 Flask 放在 HTTPS 反向代理后，不要直接把开发服务器暴露到互联网；完整安全说明见 `SECURITY.md`。

一个新题通常只需要被映射为已有的状态演化、几何事件、区间度量、网络流或优化等算子；只有真正出现新的数学原语时才注册新算子，而不是为每道题写求解分支：

```python
from core import MechanisticOperatorRegistry, OperatorDefinition

registry = MechanisticOperatorRegistry()
registry.register(OperatorDefinition(
    key="variational_energy",
    category="optimization",
    description="minimize an energy functional",
    required_bindings=("decision_variables", "objective", "constraints"),
    produces=("stationary_solution",),
    solver_route="variational_solver",
    triggers=(r"变分|能量泛函|variational",),
))
```

对于需要自动数值执行的结构，调用端（可由最后的解析 API 产生）提交结构化 `mechanistic_ir`；本地编译器会重新验证，不能靠传入 `parse_status=machine_verified` 绕过检查：

```python
result = run_modeling_study(
    problem="状态 x 满足一阶微分方程，计算 0 到 10 秒的轨迹",
    mechanistic_ir={"relations": [{
        "kind": "ode_system",
        "state_variables": ["x"],
        "rhs": {"x": "-k*x"},
        "initial_values": {"x": 100.0},
        "parameters": {"k": 0.2},
        "time_variable": "t",
        "time_span": [0.0, 10.0],
        "output_points": 301,
        "units": {"x": "人", "k": "1/s", "t": "s"},
    }]},
)
print(result["specialized_results"]["mechanistic_model"]["numerical_results"])
```

关系可通过安全路径组成有向无环计算图。下游占位字段只会在上游结果生成后绑定，并在绑定后重新执行完整验证：

```python
mechanistic_ir = {"relations": [
    {
        "id": "root", "kind": "polynomial_root", "variable": "z",
        "coefficients": [1, 0, -4], "bracket": [0, 3], "units": {"z": "1"},
    },
    {
        "id": "system", "kind": "linear_system", "variables": ["x"],
        "coefficient_matrix": [[1]], "right_hand_side": [0], "units": {"x": "1"},
        "input_bindings": [{
            "source_relation_id": "root", "source_path": "root",
            "target_path": "right_hand_side.0",
        }],
    },
]}
```

非线性优化关系使用 `kind="optimization_problem"`，声明 `decision_variables`、安全代数 `objective`、`direction`、有限 `bounds`、`initial_values`、`parameters`、`units`，以及由 `lhs/sense/rhs` 组成的约束。系统通过 SLSQP 多起点返回约束可行候选；除非另有凸性或全局上下界证书，证据会明确标为“局部最优候选；未证明全局最优”。

若题面缺少事件语义、单位、初边值、目标或约束，状态保持 `needs_model_completion`，报告会列出缺口和合理解释分支。这里的“普适”指同一建模语言可覆盖并组合不同数学结构、遇到未知情况能安全降级；不表示系统能在信息不足时猜出唯一正确模型。

`core` 公共入口采用延迟加载；普通数据分析不会预先导入 PyTorch、图像或文本模型。20 个数据集、合计 100 万行的多表关系、交互与潜在结构压力场景（每表分析采样上限 5000 行）在当前开发机上约 4.2 秒完成。实际耗时仍取决于列数、键基数、磁盘和硬件。

也可以直接在 Python 中使用：

```python
from core import run_modeling_study

result = run_modeling_study(
    problem="分析订单、客户与地区数据，预测客户满意度并解释影响因素",
    datasets={
        "customers": customers_df,
        "orders": orders_df,
        "regions": regions_df,
    },
    target="customers.satisfaction",  # 可省略，由题目自动识别
    output_dir="data/reports/my_study",
)

print(result["relationships"])
print(result["interactions"])
print(result["conclusions"])
print(result["model_result"]["credibility_audit"])
print(result["mathematical_model_spec"]["readiness"])
print(result["evidence_bundle"]["claims"])
print(result["evidence_bundle"]["writing_contract"])
```

题目同时包含多个输出指标时可直接省略 `target` 让系统识别，也可显式传入列表：

```python
result = run_modeling_study(
    problem="分别预测销量和利润，并分析共同影响因素",
    datasets={"business": business_df},
    target=["business.sales", "business.profit"],
)
print(result["model_results"])
```

需要补充尚未内置的数学计算时，可注册按任务类型触发、故障隔离的分析器；扩展应实现通用数学方法，而不是识别某道赛题名称：

```python
from core import MathModelingAssistant

assistant = MathModelingAssistant(output_dir="data/reports/optimization")
assistant.register_analyzer(
    "optimization",
    lambda **ctx: my_optimizer(ctx["datasets"], ctx["problem"]),
)
result = assistant.run("在约束下优化资源配置", {"inputs": inputs_df})
```

关联置信度来自字段语义、值域覆盖和键唯一性证据。系统不会仅凭列位置强行拼表；多对多关系默认先聚合，并在报告中保留风险提示。

可信度审计不输出容易掩盖风险的综合分数，而是逐项返回 `pass`、`warning`、`fail` 或 `not_assessed`。即使预测分数很高，只要发现目标复制、未来信息、实体重叠、结果未显著超过置乱，或近优模型给出互相冲突的结论，最终仍会判为“不可信”。排名、聚类、潜在结构和异常名单也有各自的扰动/稳定性审计。审计只能主动寻找反证，不能把统计相关证明为因果关系，也不能替代真正的外部数据验证。

论证摘要包含“多子问题执行图”和“题型能力边界”：已经实际计算的任务标记为 `executed`，具备数据但尚未运行的任务标记为 `ready`，缺少目标列、约束、初边值条件等关键数学信息时标记为 `needs_input`；依赖的上游问题尚未完成时标记为 `blocked`。系统不会把方法建议伪装成已经得到的数值结论。

### 数学方法研究依据

新增能力不是按算法名称堆叠，而是把论文思想转化为“适用条件—计算结果—反证检查—解释边界”四部分：

- 动力方程发现参考 SIAM 的 [Weak SINDy](https://doi.org/10.1137/20M1343166)：使用窗口积分降低噪声和数值求导偏差；当前实现面向 ODE 候选发现，不冒充通用 PDE 求解器。
- 因果效应参考 *The Econometrics Journal* 的 [Double/debiased machine learning](https://doi.org/10.1111/ectj.12097)：使用正交残差与交叉拟合降低干扰函数过拟合偏差；无未观测混杂和 SUTVA 仍被标记为不可由观察数据验证。
- 预测区间参考 *Biometrika* 的 [Localized conformal prediction](https://doi.org/10.1093/biomet/asac040) 与 *The Annals of Statistics* 的 [Conformal prediction beyond exchangeability](https://doi.org/10.1214/23-AOS2276)：普通回归使用有限样本分位数校正，时间数据使用固定近期权重并单独报告覆盖边界。
- 多目标评价参考 *Mathematical Methods of Operations Research* 的 [scalarization approximation theory](https://doi.org/10.1007/s00186-023-00823-2)：先保留非支配方案，再讨论依赖偏好的标量排序。
- 分布鲁棒优化参考 *Mathematical Programming* 的 [Wasserstein DRO](https://doi.org/10.1007/s10107-017-1172-1) 与带侧信息的稳健随机规划；该能力只会在决策变量、目标函数、约束和不确定参数均明确时启用，不从自然语言臆造优化模型。

因果任务需明确写出角色，例如：`估计政策效应，处理变量=policy，结果变量=income，使用处理前协变量`。只写“分析 policy 与 income 的因果关系”时，系统会拒绝自动指定方向并返回 `needs_input`。

---

## 模块一：数据模块

### 数据加载 `DataLoader`
- **自动格式识别**：CSV / Excel / JSON / Parquet / TSV
- **智能编码**：自动 fallback UTF-8 → GBK

### 类型识别 `TypeDetector`
智能识别 **8 种数据类型**：数值型、类别型、日期时间型、文本型、布尔型、ID 型、常量型、空列。

### 基础清洗 `DataCleaner`
删除无用列、缺失值填充、异常值截断、类型优化降精度。

---

## 模块二：缺失值智能分析引擎

自动区分 **3 种缺失模式**：真缺失（随机）、结构性缺失（业务条件）、目标缺失（待预测）。

```python
from core.auto_pipeline import AutoMissingPipeline
pipeline = AutoMissingPipeline()
train_df, test_df, report = pipeline.run(df)
```

---

## 模块三：建模引擎 V2 ⭐（核心新增）

### 1. 自动任务类型判断

系统能自动判断 **3 种任务类型**：

| 任务类型 | 判断逻辑 | 示例 |
|---------|---------|------|
| **分类** | 唯一值≤10 或 唯一值比例<5% 或 非数值型 | 二分类、多分类 |
| **回归** | 数值型且唯一值多、比例高 | 房价预测、销量预测 |
| **聚类** | 无目标列（y=None） | 客户分群 |

```python
from core.modeling_engine import TaskTypeDetector
TaskTypeDetector.detect(y, X)  # → TaskType.CLASSIFICATION/REGRESSION/CLUSTERING
```

### 2. 丰富的模型库

| 类别 | 分类模型 | 回归模型 | 聚类模型 |
|-----|---------|---------|---------|
| **线性/统计** | LogisticRegression, LDA, QDA, NaiveBayes | LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Huber, ARD, PLS, TheilSen | — |
| **SVM** | SVC | SVR | — |
| **KNN** | KNeighborsClassifier | KNeighborsRegressor | — |
| **树模型** | DecisionTree | DecisionTree | — |
| **集成** | RandomForest, ExtraTrees, GradientBoosting | RandomForest, ExtraTrees, GradientBoosting | — |
| **梯度提升** | XGBoost, LightGBM, CatBoost | XGBoost, LightGBM, CatBoost | — |
| **神经网络** | MLPClassifier | MLPRegressor | — |
| **聚类** | — | — | KMeans, DBSCAN, Agglomerative, GaussianMixture, Spectral |

### 3. 统一模型接口

```python
from core.modeling_engine import ModelLibrary

# 查看可用模型
print(ModelLibrary.list_models(TaskType.CLASSIFICATION))
#    key      name       category   gpu  params
# 0   lr  LogisticRegression  linear  False     3
# 1   rf      RandomForest  ensemble  False     4
# ...

# 创建模型实例
model = ModelLibrary.create_model('xgb', TaskType.CLASSIFICATION, max_depth=5)
```

### 4. 智能编码（自动选择策略）

```python
from core.modeling_engine import AutoEncoder

enc = AutoEncoder()
X_encoded = enc.fit_transform(X, y)
print(enc.get_encoding_report())
#      column  strategy    encoder_type
# 0   gender   onehot  OneHotEncoder
# 1     city   ordinal  OrdinalEncoder
# 2  user_id  frequency  mapping
```

| 基数 | 策略 |
|-----|------|
| ≤2 | Label Encoding |
| ≤10 | One-Hot Encoding |
| 11~50 | Ordinal Encoding |
| >50（有标签） | Target Encoding |
| >50（无标签） | Frequency Encoding |

### 5. 自动特征选择

```python
from core.modeling_engine import AutoFeatureSelector, FeatureSelectionStrategy

selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MI, n_features=20)
X_selected = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
print(selector.get_feature_importance())
```

| 策略 | 说明 |
|-----|------|
| `variance` | 方差阈值（删除低方差特征） |
| `mi` | 互信息选择 |
| `rfe` | 递归特征消除 |
| `model_based` | 基于随机森林重要性 |
| `correlation` | 高共线性过滤 |
| `pca` | PCA降维 |
| `none` | 不选择 |

### 6. K折交叉验证

```python
from core.modeling_engine import CrossValidator

cv = CrossValidator(n_splits=5)
result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
# result.oof_pred: Out-of-Fold预测
# result.mean_scores: 平均指标
# result.fitted_models: 每折训练好的模型
# result.feature_importance: 特征重要性
```

### 7. 多模型融合

```python
from core.modeling_engine import EnsembleBuilder, EnsembleMethod

builder = EnsembleBuilder(method=EnsembleMethod.WEIGHTED)
blend = builder.blend(cv_results, X_test, task_type)
# blend['oof']: OOF融合结果
# blend['test']: 测试集融合结果
# blend['weights']: 各模型权重
```

| 融合方法 | 说明 |
|---------|------|
| `weighted` | 按CV分数加权平均（默认） |
| `voting_hard` | 硬投票（分类） |
| `voting_soft` | 软投票（分类，需概率） |
| `stacking` | 堆叠（元学习器） |
| `best_single` | 只用最优单模型 |

### 8. 完整建模流程

```python
from core.modeling_engine import ModelingEngine

engine = ModelingEngine(
    task_type='classification',           # None=自动判断
    model_keys=['lr', 'rf', 'xgb'],       # None=全部模型
    n_splits=5,
    encoding='onehot',                    # auto/onehot/label/target/none
    feature_selection='mi',               # mi/variance/rfe/model_based/correlation/pca/none
    ensemble='weighted'                   # weighted/voting_hard/voting_soft/stacking/best_single
)

result = engine.fit(X_train, y_train, X_test)

result.leaderboard          # 模型排行榜
result.feature_importance   # 集成特征重要性
result.ensemble_result      # 融合结果
result.cv_results[0].oof_pred  # OOF预测

predictions = engine.predict(X_test)
engine.print_report()
```

---

## 模块四：集成流水线（端到端）

```python
from core.integrated_pipeline import IntegratedPipeline, quick_run

# 方式1：完整控制
pipeline = IntegratedPipeline(
    target_col='target',
    task_type='classification',        # None=自动判断
    model_keys=['lr', 'rf', 'xgb', 'lgb'],
    encoding='onehot',
    feature_selection='mi',
    ensemble='weighted',
    n_splits=5,
    allow_disk_write=True              # False=禁止一切磁盘写入
)
result = pipeline.run(df)
pipeline.print_summary()

# 方式2：一行快速运行
result = quick_run(df, target_col='target')
```

### 输出结果

```python
result.task_type            # 'classification'
result.leaderboard          # DataFrame: 模型排行榜
result.feature_importance   # DataFrame: Top20重要特征
result.predictions          # 测试集预测
result.oof_predictions      # OOF预测
result.ensemble_weights     # {'XGBoost': 0.4, 'LightGBM': 0.35, ...}
result.encoding_report      # DataFrame: 编码策略
result.preprocessing_info   # {'original_features': 50, 'encoded_features': 72, 'selected_features': 30}
```

---

## 模块五：自动性能调度

系统自动根据数据规模选择策略：

| 策略 | 数据规模 | 特点 |
|-----|---------|------|
| **STANDARD** | < 1万行 | 完整分析，5折CV |
| **FAST** | 1万~100万行 | 采样加速，3折CV |
| **ULTRA** | > 100万行 | 极简分析，GPU优先 |

---

## 模块六：工作空间管理器（磁盘隔离）

```python
from core.workspace_manager import get_workspace_manager, set_workspace_config

# 关闭磁盘写入（纯内存模式）
set_workspace_config(allow_disk_write=False)

# C盘路径自动重定向到项目目录
wm = get_workspace_manager()
wm.safe_path('C:/Users/xxx/result.csv')
# → 项目/workspace/reports/result.csv
```

---

## 安装与测试

```bash
# 安装基础依赖
pip install -r requirements.txt

# 可选：安装建模库获得最佳体验
pip install xgboost lightgbm catboost

# 运行全部测试（850+ 项，覆盖率 83%+）
python -m pytest tests/ -v

# 查看覆盖率报告
python -m pytest tests/ --cov=core --cov-report=term
```

## 核心类速查

| 类 | 文件 | 用途 |
|---|------|------|
| `DataModule` | `data_module.py` | 数据加载→分析→清洗 |
| `AutoMissingPipeline` | `auto_pipeline.py` | 缺失分析自动流程 |
| `ModelingEngine` | `modeling_engine.py` | **建模核心（编码+特征选择+CV+融合）** |
| `AutoEncoder` | `modeling_engine.py` | 智能编码（OneHot/Label/Target） |
| `AutoFeatureSelector` | `modeling_engine.py` | 自动特征选择 |
| `ModelLibrary` | `modeling_engine.py` | 统一模型库 |
| `CrossValidator` | `modeling_engine.py` | K折交叉验证 |
| `EnsembleBuilder` | `modeling_engine.py` | 模型融合 |
| `IntegratedPipeline` | `integrated_pipeline.py` | 端到端完整流水线 |
| `WorkspaceManager` | `workspace_manager.py` | 磁盘IO隔离与开关控制 |
| `ResultCache` | `result_cache.py` | 双层缓存（内存+磁盘），避免重复计算 |
| `RLOptimizer` | `reinforcement_learning.py` | DQN 强化学习超参优化 |
| `TorchNAS` | `nas.py` | 神经架构搜索 |
| `ImageResNet` | `multimodal.py` | 图像分类（ResNet18） |
| `TextBERT` | `multimodal.py` | 文本分类（DistilBERT） |
| `FairnessEngine` | `fairness.py` | 公平性分析 |
| `ExplainabilityEngine` | `explainability.py` | SHAP/LIME 模型解释 |
