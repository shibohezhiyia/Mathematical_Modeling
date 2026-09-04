"""
Universal Math Modeling Problem Solver

Analyzes any text-based modeling problem and recommends
a general modeling framework, not domain-specific templates.
"""
import re
from typing import Any, Dict, List

# 预编译正则表达式，避免每次调用时重新编译
_PROBLEM_TYPE_PATTERNS = {
    'data_requirements': re.compile(
        r'还需[^。；\n]{0,30}数据|需要[^。；\n]{0,30}(?:采集|收集|补充)[^。；\n]{0,20}数据|'
        r'(?:采集|收集|补充)[^。；\n]{0,20}(?:哪些|什么)?数据|哪些数据|数据需求|'
        r'数据[^。；\n]{0,20}(?:帮助|作用|用途)|意见和理由'
    ),
    'optimization': re.compile(
        r'最大化|最小化|最大|最小|最优|优化|分配|调度|规划|配置|'
        r'路径(?:选择|规划|优化)|成本|利润|效费|节约|资源配置|约束|方案|策略|'
        r'尽可能|尽量|决策|选择[^。；\n]{0,30}(?:位置|时间|方向|速度)'
    ),
    'differential_equations': re.compile(r'变化|增长|传播|扩散|动态|演化|速率|随时间|导数|微分|方程|运动|轨迹|弹道'),
    'prediction_forecast': re.compile(
        r'预测|预报|估计|趋势|未来|'
        r'下(?:一|个)(?:时刻|周期|季度|年度|月份?|周|天)|会达到|销量|人口|疫情'
    ),
    'classification': re.compile(r'分类|识别|判别|诊断|判断|是否|好坏|等级|类型'),
    'clustering': re.compile(r'聚类|分组|划分|聚成|相似|类别|群落'),
    'simulation': re.compile(
        r'模拟|仿真|蒙特卡洛|随机|概率|风险|不确定|抽样|'
        r'有效(?:持续|作用|覆盖|遮蔽)?(?:时间|时长)|事件(?:持续)?时间|作用时长'
    ),
    'graph_network': re.compile(r'网络|图|节点|边|路径|流量|连接|路线|拓扑|最短|联通'),
    'statistical_inference': re.compile(
        r'显著|相关|回归|检验|置信(?:区间|水平|度)|假设|分布|频率|统计|影响因素|作用因素'
    ),
    'causal_inference': re.compile(r'因果|处理效应|干预效应|政策效果|政策效应|反事实|因果效应|treatment effect|causal'),
    'evaluation_ranking': re.compile(r'评价|评估|排名|排序|指标|得分|综合|优劣'),
    'anomaly_detection': re.compile(r'异常|离群|异常点|异常值|反常|突变|预警|故障检测|outlier|anomaly'),
    'dimension_reduction': re.compile(r'主成分|因子分析|降维|维度约简|指标压缩|特征提取|pca|PCA'),
}
_ENTITY_PATTERN = re.compile(r'([A-Z]{1,3}\d{1,3})[（\(]([\d\-,.\s]+)[）\)]')
# 预编译 _extract_variables / _extract_constraints 正则
_VAR_PATTERNS = [
    re.compile(r'(\d+(?:\.\d+)?)\s*(m/s|km/h|kg|t|s|min|h|m|km|℃|°|度|个|架|枚)'),
    re.compile(r'([一-龥]+?)(?:分别为|为|是|等于|约|大概)\s*(\d+(?:\.\d+)?)'),
    re.compile(r'坐标[为是]\s*\(?\s*([\d\-,\.\s]+)\s*\)?'),
]
_RANGE_PATTERN = re.compile(r'([\d\.]+)\s*[~到-]\s*([\d\.]+)\s*(m/s|km/h|kg|t|s|min|h|m|km)')
_MAX_PATTERN = re.compile(r'(?:至多|最多|不超过|不大于|≤|<=)\s*(\d+)')
_MIN_PATTERN = re.compile(r'(?:至少|最少|不低于|不小于|≥|>=)\s*(\d+)')
_INTERVAL_PATTERN = re.compile(r'间隔\s*(\d+(?:\.\d+)?)\s*s?')
_MAXIMIZE_PATTERN = re.compile(r'(?:使|让|求|要)(.*?)(?:尽可能大|最大|最长|最高|最优|最好)')
_MINIMIZE_PATTERN = re.compile(r'(?:使|让|求|要)(.*?)(?:尽可能小|最小|最短|最低|最少)')
_EXPLICIT_SUBPROBLEM_PATTERN = re.compile(
    r'(?<![0-9A-Za-z_\u4e00-\u9fff])(?:问题|任务|小问|Problem|Task)\s*'
    r'(?:[一二三四五六七八九十]+|\d+(?!\.\d))\s*[、.．)）:：]?\s*',
    re.IGNORECASE,
)
_LINE_SUBPROBLEM_PATTERN = re.compile(
    r'^\s*(?:[一二三四五六七八九十]+|\d+)\s*[、.．)）:：]\s*',
    re.MULTILINE,
)

_TASK_FORMULAS: Dict[str, List[str]] = {
    'data_requirements': [
        '信息缺口 = 任务所需角色 − 已观测角色',
        '采集优先级 = 决策价值 × 可识别性提升 ÷ 采集成本',
        '验证设计 = 时间/实体/干预粒度 + 单位 + 缺失机制',
    ],
    'optimization': ['min/max f(x)', 'g_i(x) ≤ 0, h_j(x) = 0', '敏感性: ∂f*/∂θ'],
    'differential_equations': ['dx/dt = f(t, x, θ)', 'x(t₀) = x₀', '参数拟合: min Σ(yᵢ-x(tᵢ;θ))²'],
    'prediction_forecast': ['ŷₜ₊ₕ = f(yₜ, yₜ₋₁, Xₜ)', 'RMSE = √mean((y-ŷ)²)', '预测区间: ŷ ± z·SE'],
    'classification': ['P(y=k|x) = softmax(f_k(x))', 'F1 = 2PR/(P+R)', '交叉熵: -Σ y log(p)'],
    'clustering': ['min Σᵢ ||xᵢ-μcᵢ||²', 'silhouette = (b-a)/max(a,b)'],
    'simulation': ['E[g(X)] ≈ (1/N)Σg(Xᵢ)', 'CI = x̄ ± z·s/√N'],
    'graph_network': ['G=(V,E,W)', '最短路: d(v)=min[d(v),d(u)+w(u,v)]'],
    'statistical_inference': ['H₀ vs H₁', 'y = Xβ + ε', 'CI(β)=β̂ ± t·SE(β̂)'],
    'causal_inference': ['Y=θD+g(X)+ε', 'D=m(X)+ν', 'θ̂=Σν̂(Y-ĝ(X))/Σν̂²'],
    'evaluation_ranking': ['wⱼ=(1-eⱼ)/Σ(1-eⱼ)', 'Cᵢ=Dᵢ⁻/(Dᵢ⁺+Dᵢ⁻)'],
    'anomaly_detection': ['eᵢ=||xᵢ-x̂ᵢ||²', 'robust z=(eᵢ-median(e))/(1.4826·MAD)', '异常阈值: z>3.5'],
    'dimension_reduction': ['Z=(X-μ)/σ', 'Σvⱼ=λⱼvⱼ', '累计解释率=Σ₁ᵏλⱼ/Σλⱼ'],
}

# === 模块级常量：原本散落在各函数体中的大型字典 ===
# 提升到模块顶部后只在 import 时构建一次，避免每次 analyze_problem 调用重建。
# （虽然数据量不大，但是 9 个 task_type × 数百行字符串模板会重复构建数百次）

# 各任务类型对应的推荐模型类别（name/description/confidence）
_MODEL_CLASSES: Dict[str, Dict[str, Any]] = {
    'data_requirements': {
        'name': '数据需求与可识别性审计',
        'description': '从待解任务、现有字段和证据缺口反推应补充的数据及采集设计',
        'confidence': 85,
    },
    'optimization': {
        'name': '数学优化模型',
        'description': '线性/非线性规划、整数规划、动态规划、启发式算法',
        'confidence': 90,
    },
    'differential_equations': {
        'name': '微分方程与动力学模型',
        'description': 'ODE/PDE、 compartment模型、传播动力学、运动学方程',
        'confidence': 85,
    },
    'prediction_forecast': {
        'name': '预测与预报模型',
        'description': '时间序列(ARIMA/Prophet)、回归、机器学习、深度学习',
        'confidence': 85,
    },
    'classification': {
        'name': '分类与识别模型',
        'description': '逻辑回归、决策树、SVM、集成学习、神经网络',
        'confidence': 80,
    },
    'clustering': {
        'name': '聚类分析模型',
        'description': 'K-Means、层次聚类、DBSCAN、谱聚类',
        'confidence': 80,
    },
    'simulation': {
        'name': '仿真与模拟模型',
        'description': '蒙特卡洛模拟、Agent-based、离散事件仿真',
        'confidence': 80,
    },
    'graph_network': {
        'name': '图论与网络模型',
        'description': '最短路径、最大流、最小生成树、网络优化、PageRank',
        'confidence': 85,
    },
    'statistical_inference': {
        'name': '统计推断模型',
        'description': '假设检验、回归分析、方差分析、贝叶斯推断',
        'confidence': 80,
    },
    'causal_inference': {
        'name': '正交化因果效应模型',
        'description': '交叉拟合、双重机器学习、处理效应与识别假设审计',
        'confidence': 85,
    },
    'evaluation_ranking': {
        'name': '评价与排名模型',
        'description': '层次分析法(AHP)、TOPSIS、熵权法、模糊综合评价',
        'confidence': 75,
    },
    'anomaly_detection': {
        'name': '稳健异常检测模型',
        'description': '稳健统计、PCA重构误差、孤立森林、变化点检测',
        'confidence': 80,
    },
    'dimension_reduction': {
        'name': '降维与潜在结构模型',
        'description': 'PCA、因子分析、流形学习、载荷与子空间稳定性',
        'confidence': 80,
    },
}
# 兜底引用：未匹配到 task_type 时使用 evaluation_ranking 对应的配置
_DEFAULT_MODEL_CLASS: Dict[str, Any] = _MODEL_CLASSES['evaluation_ranking']

_TASK_IO: Dict[str, Dict[str, List[str]]] = {
    'data_requirements': {
        'requires': ['待解决任务', '现有字段角色与证据缺口'],
        'produces': ['分级数据清单', '采集粒度与时点', '数据用途与可验证收益'],
    },
    'prediction_forecast': {
        'requires': ['目标变量', '可用于预测时获得的特征', '时间任务需时间列'],
        'produces': ['验证预测', '误差与区间', '影响因素'],
    },
    'classification': {
        'requires': ['类别目标', '预测时可获得的特征'],
        'produces': ['类别预测', '混淆矩阵', '分类置信证据'],
    },
    'clustering': {
        'requires': ['至少两个可比较指标', '实体粒度一致'],
        'produces': ['群体标签', '簇画像', '聚类稳定性'],
    },
    'evaluation_ranking': {
        'requires': ['评价对象', '至少两个指标', '正负向属性'],
        'produces': ['综合得分', '排名', '权重敏感性'],
    },
    'optimization': {
        'requires': ['决策变量', '可计算目标函数', '约束与变量边界'],
        'produces': ['可行解', '目标值', '约束松弛与敏感性'],
    },
    'differential_equations': {
        'requires': ['状态变量', '时间', '初边值条件', '机理或变化率关系'],
        'produces': ['状态轨迹', '参数估计', '稳定性与相图'],
    },
    'simulation': {
        'requires': ['随机输入分布', '状态转移或结果函数'],
        'produces': ['输出分布', '置信区间', '极端情景风险'],
    },
    'graph_network': {
        'requires': ['节点', '边', '路径任务需权重与起终点'],
        'produces': ['连通性', '中心性', '路径或流量结果'],
    },
    'statistical_inference': {
        'requires': ['可检验假设', '变量与样本单位'],
        'produces': ['效应量', '区间', '多重校正显著性'],
    },
    'causal_inference': {
        'requires': ['显式处理变量', '结果变量', '处理前混杂变量', '无未观测混杂等识别假设'],
        'produces': ['正交化处理效应', '置信区间', '重叠性与安慰剂检验'],
    },
    'anomaly_detection': {
        'requires': ['正常参照样本或可比较的多变量观测'],
        'produces': ['异常名单', '异常强度', '主要偏离变量'],
    },
    'dimension_reduction': {
        'requires': ['至少两个数值指标', '足够样本'],
        'produces': ['潜在维度', '主成分载荷', '重构误差'],
    },
}

# 通用建模步骤（1-4 步），与具体任务类型无关
_COMMON_STEPS: List[str] = [
    '1. 问题理解：明确已知条件、决策变量、约束条件和目标函数',
    '2. 符号定义：为所有变量、参数、集合建立规范的数学符号',
    '3. 基本假设：列出简化假设（如忽略次要因素、理想化条件）',
    '4. 数据整理：提取题目中所有数值参数，建立参数表',
]

# 各任务类型专属步骤（5-10 步）
_TYPE_STEPS: Dict[str, List[str]] = {
    'data_requirements': [
        '5. 任务反推：把每个待解结论映射为必须观测的变量角色',
        '6. 缺口审计：区分已有、可派生、需外采和不可观测变量',
        '7. 可识别性检查：判断新增数据能否区分竞争假设或减少决策不确定性',
        '8. 采集设计：明确实体、时间、空间、单位、频率和干预记录',
        '9. 优先级排序：按决策价值、不可替代性、成本和时效分级',
        '10. 验收规则：为覆盖率、缺失机制、延迟和漂移设定检查标准',
    ],
    'optimization': [
        '5. 建立优化模型：定义决策变量 x、目标函数 f(x)、约束条件 g(x)≤0, h(x)=0',
        '6. 判断问题规模：小规模用精确算法（单纯形、内点法），大规模用启发式（遗传算法、模拟退火）',
        '7. 选择求解工具：scipy.optimize（连续）、PuLP/CVXPY（线性）、DEAP/自编码GA（离散）',
        '8. 求解并验证：检查解的可行性，分析约束松紧度',
        '9. 敏感性分析：参数扰动对最优解的影响',
        '10. 输出结果表格：按题目要求整理为 Excel/CSV',
    ],
    'differential_equations': [
        '5. 建立动力学方程：根据物理/生物/化学规律列写微分方程',
        '6. 确定初边值条件：初始状态、边界约束',
        '7. 解析求解（若可能）：分离变量、特征线法、格林函数',
        '8. 数值求解：scipy.integrate.odeint/solve_ivp（ODE）、有限差分/有限元（PDE）',
        '9. 数值实验：参数扫描、相图分析、稳定性分析',
        '10. 结果可视化：轨迹图、相平面图、时空演化图',
    ],
    'prediction_forecast': [
        '5. 数据探索：趋势、季节、周期、异常值检测',
        '6. 特征工程：滞后特征、滑动窗口、差分、对数变换',
        '7. 基线模型：移动平均、指数平滑、线性趋势',
        '8. 进阶模型：ARIMA/SARIMA、Prophet、LSTM/Transformer',
        '9. 模型评估：训练集/验证集划分，交叉验证，指标（RMSE/MAPE/R²）',
        '10. 预测与置信区间：点预测 + 区间估计',
    ],
    'classification': [
        '5. 数据预处理：缺失值处理、编码、标准化、特征选择',
        '6. 基线模型：Logistic Regression、KNN、决策树',
        '7. 集成模型：Random Forest、XGBoost、LightGBM',
        '8. 深度学习：MLP、CNN（图像）、BERT（文本）',
        '9. 评估：Accuracy、Precision、Recall、F1、AUC-ROC、混淆矩阵',
        '10. 可解释性：特征重要性、SHAP、LIME',
    ],
    'clustering': [
        '5. 特征标准化：消除量纲影响',
        '6. 降维可视化：PCA、t-SNE、UMAP',
        '7. 确定簇数：肘部法则、轮廓系数、Gap Statistic',
        '8. 聚类算法：K-Means（球形）、DBSCAN（任意形状）、层次聚类（树状图）',
        '9. 结果评估：轮廓系数、Davies-Bouldin指数、可视化检验',
        '10. 簇特征分析：每个簇的中心、范围、主导特征',
    ],
    'simulation': [
        '5. 建立仿真模型：定义实体、状态、事件、转移规则',
        '6. 确定随机分布：根据数据或假设选择分布（正态、指数、泊松等）',
        '7. 蒙特卡洛模拟：大量重复实验，统计输出分布',
        '8. Agent-based：定义智能体行为规则和交互规则',
        '9. 结果分析：均值、方差、置信区间、极端情景',
        '10. 参数敏感性：哪些参数对结果影响最大',
    ],
    'graph_network': [
        '5. 建立图模型：节点集合 V、边集合 E、权重 w(e)',
        '6. 图属性分析：度分布、连通性、聚类系数',
        '7. 选择算法：最短路径(Dijkstra/Floyd)、最大流(Edmonds-Karp)、最小生成树(Kruskal/Prim)',
        '8. 网络优化：线性规划建模、整数规划建模',
        '9. 复杂网络分析：中心性、社区发现、鲁棒性',
        '10. 可视化：networkx + matplotlib/pyvis',
    ],
    'statistical_inference': [
        '5. 描述统计：均值、方差、分位数、分布形态',
        '6. 假设检验：t检验、卡方检验、方差分析(ANOVA)、KS检验',
        '7. 回归分析：线性/多项式/非线性回归、正则化',
        '8. 置信区间与显著性：p值、置信水平、效应量',
        '9. 模型诊断：残差分析、QQ图、异方差检验',
        '10. 结论表述：统计显著性与实际意义',
    ],
    'causal_inference': [
        '5. 定义因果角色：明确处理、结果、处理前混杂变量和估计目标',
        '6. 审计识别假设：一致性、SUTVA、可忽略性、正值性与时间先后',
        '7. 交叉拟合：分别估计结果模型和处理模型，避免同样本过拟合偏差',
        '8. 正交化估计：使用残差对残差回归估计平均处理效应',
        '9. 反证检查：重叠性、安慰剂置乱、折间稳定性与敏感性分析',
        '10. 边界表述：观察数据的因果结论必须连同不可检验假设报告',
    ],
    'evaluation_ranking': [
        '5. 建立指标体系：目标层、准则层、指标层',
        '6. 数据标准化：正向化、归一化、Z-score',
        '7. 确定权重：AHP成对比较、熵权法（客观）、组合赋权',
        '8. 综合评价：TOPSIS（距理想解距离）、灰色关联、模糊综合',
        '9. 敏感性分析：权重变化对排名的影响',
        '10. 结果输出：排名表、雷达图、权重分布图',
    ],
    'anomaly_detection': [
        '5. 建立正常参照：使用中位数、MAD和稳健尺度避免异常点污染基线',
        '6. 多变量检测：比较PCA重构误差、孤立森林或局部密度方法',
        '7. 阈值校准：结合经验分位数、业务损失或已知异常样本确定阈值',
        '8. 稳定性验证：改变抽样、特征和阈值，检查异常名单重合度',
        '9. 归因分析：报告导致异常的主要变量及其偏离方向',
        '10. 输出预警：区分统计异常、数据错误和业务重要事件',
    ],
    'dimension_reduction': [
        '5. 数据适用性：检查尺度、共线性、KMO和Bartlett球形检验',
        '6. 标准化：消除量纲影响并处理缺失和近零方差变量',
        '7. 选择维数：综合累计解释率、平行分析和重构误差',
        '8. 稳定性验证：在重抽样数据上比较主子空间和载荷方向',
        '9. 解释结构：报告主成分载荷、共同度和潜在指标含义',
        '10. 下游验证：比较降维前后的预测、聚类或评价结论',
    ],
}
# 兜底引用：未匹配到 task_type 时使用 evaluation_ranking 对应的步骤
_DEFAULT_TYPE_STEPS: List[str] = _TYPE_STEPS['evaluation_ranking']

# 各任务类型对应的 Python 代码框架模板（~250 行）
_CODE_FRAMEWORKS: Dict[str, str] = {
    'data_requirements': '''
# 数据需求不是凭经验罗列，而是由“任务契约 - 现有角色”得到。
required_roles = set(task_contract["required_roles"])
observed_roles = set(schema_roles["observed_roles"])
derivable_roles = set(schema_roles["derivable_roles"])
gaps = required_roles - observed_roles - derivable_roles

recommendations = rank_by_value_of_information(
    gaps,
    decision_impact=task_contract["decision_impact"],
    identifiability_gain=task_contract["identifiability_gain"],
    collection_cost=collection_cost,
)
# 每项必须同时输出：字段角色、粒度、时点、单位、用途和验收规则。
''',
    'optimization': '''
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog

# ===== 参数定义 =====
# 从题目中提取所有数值参数
params = {...}

# ===== 决策变量 =====
# x = [x1, x2, ...]

# ===== 目标函数 =====
def objective(x):
    return ...  # 根据题目建立

# ===== 约束条件 =====
constraints = [
    {'type': 'ineq', 'fun': lambda x: ...},  # g(x) >= 0
    {'type': 'eq',   'fun': lambda x: ...},  # h(x) = 0
]
bounds = [(lb, ub), ...]  # 变量上下界

# ===== 求解 =====
# 小规模连续: SLSQP
result = minimize(objective, x0=np.zeros(n), method='SLSQP', bounds=bounds, constraints=constraints)

# 大规模/非凸: 遗传算法
# result = differential_evolution(objective, bounds, maxiter=200, seed=42, workers=-1)

print("最优解:", result.x)
print("最优值:", result.fun)

# ===== 输出到Excel =====
import pandas as pd
pd.DataFrame({...}).to_excel('result.xlsx', index=False)
''',
    'differential_equations': '''
import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

# ===== 参数 =====
params = {...}

# ===== ODE定义 =====
def dy_dt(t, y):
    # y = [y1, y2, ...]
    dydt = [...]
    return dydt

# ===== 初值条件 =====
y0 = [...]
t_span = (0, t_max)
t_eval = np.linspace(0, t_max, 1000)

# ===== 求解 =====
sol = solve_ivp(dy_dt, t_span, y0, t_eval=t_eval, method='RK45')

# ===== 结果分析 =====
plt.plot(sol.t, sol.y[0])
plt.xlabel('t'); plt.ylabel('y')
plt.show()
''',
    'prediction_forecast': '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ===== 数据准备 =====
# df = pd.read_csv('data.csv')
# X, y = df[features], df[target]

# ===== 基线: ARIMA =====
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)

# ===== 进阶: XGBoost / LightGBM =====
# from xgboost import XGBRegressor
# model = XGBRegressor(n_estimators=200, max_depth=6)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# ===== 评估 =====
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"RMSE={rmse:.4f}, MAPE={mape:.4f}")
''',
    'classification': '''
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== 数据准备 =====
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ===== 模型训练 =====
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===== 预测与评估 =====
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ===== 特征重要性 =====
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
''',
    'clustering': '''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# ===== 数据标准化 =====
X_scaled = StandardScaler().fit_transform(X)

# ===== 确定K值 =====
scores = []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))
best_k = np.argmax(scores) + 2

# ===== 聚类 =====
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# ===== 结果分析 =====
df['cluster'] = labels
print(df.groupby('cluster').mean())
''',
    'simulation': '''
import numpy as np
import pandas as pd

# ===== 参数 =====
n_simulations = 10000

# ===== 蒙特卡洛模拟 =====
results = []
for i in range(n_simulations):
    # 随机抽样
    sample = np.random.normal(mu, sigma)
    # 模拟逻辑
    outcome = simulate_one(sample)
    results.append(outcome)

# ===== 结果统计 =====
results = np.array(results)
print(f"Mean={results.mean():.4f}, Std={results.std():.4f}")
print(f"95% CI: [{np.percentile(results,2.5):.4f}, {np.percentile(results,97.5):.4f}]")

# ===== 可视化 =====
import matplotlib.pyplot as plt
plt.hist(results, bins=50, edgecolor='black')
plt.show()
''',
    'graph_network': '''
import networkx as nx
import pandas as pd

# ===== 建图 =====
G = nx.DiGraph()  # or nx.Graph()
G.add_nodes_from([...])
G.add_weighted_edges_from([(u, v, w), ...])

# ===== 最短路径 =====
path = nx.shortest_path(G, source='A', target='B', weight='weight')
length = nx.shortest_path_length(G, source='A', target='B', weight='weight')

# ===== 最大流 =====
flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink')

# ===== 中心性分析 =====
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# ===== 可视化 =====
nx.draw(G, with_labels=True, node_color='lightblue')
''',
    'statistical_inference': '''
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# ===== 描述统计 =====
print(df.describe())

# ===== 假设检验 =====
# t检验
stat, pvalue = stats.ttest_ind(group_a, group_b)
print(f"t={stat:.4f}, p={pvalue:.4f}")

# ===== 回归分析 =====
X = sm.add_constant(df[['x1', 'x2']])
model = sm.OLS(df['y'], X).fit()
print(model.summary())

# ===== 置信区间 =====
conf = model.conf_int(alpha=0.05)
''',
    'causal_inference': '''
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor

# 必须由题目明确指定，不能靠相关性自动命名因果角色
Y = df[outcome].to_numpy(dtype=float)
D = df[treatment].to_numpy(dtype=float)
X = pd.get_dummies(df[pre_treatment_controls], drop_first=False)
folds = KFold(5, shuffle=True, random_state=42)

g_hat = cross_val_predict(HistGradientBoostingRegressor(), X, Y, cv=folds)
m_hat = cross_val_predict(HistGradientBoostingRegressor(), X, D, cv=folds)
y_residual = Y - g_hat
d_residual = D - m_hat
theta = np.sum(d_residual * y_residual) / np.sum(d_residual ** 2)
print('正交化处理效应:', theta)
# 仍需核查无未观测混杂、SUTVA、正值性和处理先于结果。
''',
    'evaluation_ranking': '''
import numpy as np
import pandas as pd

# ===== 数据标准化 =====
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

# ===== 熵权法求权重 =====
def entropy_weight(df):
    p = df / df.sum()
    e = -np.nansum(p * np.log(p)) / np.log(len(df))
    w = (1 - e) / (1 - e).sum()
    return w

# ===== TOPSIS =====
def topsis(df, weights):
    norm = df / np.sqrt((df**2).sum())
    weighted = norm * weights
    ideal_best = weighted.max()
    ideal_worst = weighted.min()
    d_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    score = d_worst / (d_best + d_worst)
    return score

# ===== 计算并排名 =====
weights = entropy_weight(df)
scores = topsis(df, weights)
rank = scores.rank(ascending=False)
''',
    'anomaly_detection': '''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

X_scaled = RobustScaler().fit_transform(X)
pca = PCA(n_components=0.9, svd_solver='full', random_state=42)
scores = pca.fit_transform(X_scaled)
reconstructed = pca.inverse_transform(scores)
error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
median = np.median(error)
mad = np.median(np.abs(error - median))
robust_z = (error - median) / max(1.4826 * mad, 1e-12)
anomaly_mask = robust_z > 3.5
''',
    'dimension_reduction': '''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.9, svd_solver='full', random_state=42)
scores = pca.fit_transform(X_scaled)
loadings = pca.components_.T
print('保留维数:', pca.n_components_)
print('累计解释率:', pca.explained_variance_ratio_.sum())
''',
}


def analyze_problem(description: str) -> Dict[str, Any]:
    """Analyze a modeling problem and return a general framework."""
    desc = description.lower()
    
    # Step 1: Identify the core task type
    task_scores = {key: len(pattern.findall(desc)) for key, pattern in _PROBLEM_TYPE_PATTERNS.items()}
    task_type = min(task_scores, key=lambda key: (-task_scores[key], key))
    if task_scores[task_type] == 0:
        task_type = 'evaluation_ranking'
    
    # Step 2: Extract variables, constraints, and objectives
    variables = _extract_variables(desc)
    constraints = _extract_constraints(desc)
    objectives = _extract_objectives(desc)
    
    # Step 3: Recommend model class and methods
    model_class = _recommend_model_class(task_type, desc)
    
    # Step 4: Generate general modeling steps
    steps = _generate_steps(task_type, model_class)
    
    # Step 5: Generate code framework
    code = _generate_code_framework(task_type, model_class)
    
    ranked_tasks = [
        {'task_type': key, 'score': score}
        for key, score in sorted(task_scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    ]
    subproblems = _extract_subproblems(description)
    if not subproblems:
        meaningful_objectives = [
            item for item in objectives if item != '分析题目并建立数学模型'
        ]
        subproblems = meaningful_objectives or [description.strip()]
    task_graph = _build_task_graph(
        subproblems or [description], fallback_task=task_type
    )

    result = {
        'task_type': task_type,
        'task_candidates': ranked_tasks,
        'model_class': model_class['name'],
        'model_description': model_class['description'],
        'variables': variables,
        'constraints': constraints,
        'objectives': objectives,
        'subproblems': subproblems[:10],
        'task_graph': task_graph,
        'formulas': _TASK_FORMULAS.get(task_type, _TASK_FORMULAS['statistical_inference']),
        'steps': steps,
        'code_framework': code,
        'confidence': model_class['confidence'],
    }
    # Backward-compatible presentation keys used by the original web client.
    result.update({
        'model': result['model_class'],
        'approach': result['steps'],
        'key_features': result['variables'],
        'code_template': result['code_framework'],
    })
    return result


def _extract_subproblems(description: str) -> List[str]:
    """Extract real question sections without treating coordinates as list markers.

    Explicit markers such as ``问题2`` may occur inline. Bare numbered markers are
    accepted only at the beginning of a line, so ``(0, 200, 0)`` can never split
    the statement. The marker is retained for downstream provenance alignment.
    """
    text = str(description)
    matches = list(_EXPLICIT_SUBPROBLEM_PATTERN.finditer(text))
    if not matches:
        matches = list(_LINE_SUBPROBLEM_PATTERN.finditer(text))
    if not matches:
        stripped = text.strip()
        return [stripped] if len(stripped) >= 4 else []
    sections: List[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start():end].strip(' ：:；;。\n')
        if len(section) >= 4:
            sections.append(section)
    return sections[:100]


def _build_task_graph(subproblems: List[str], fallback_task: str) -> List[Dict[str, Any]]:
    """Turn multi-part prose into composable task nodes with explicit data flow."""
    graph: List[Dict[str, Any]] = []
    flow_markers = ('上述', '前述', '前面', '基于此', '基于以上', '根据以上', '利用预测', '由问题')
    composition_markers = ('并', '同时', '然后', '随后', '再进行', '以及', '并据此')
    dependency_sources = {
        'optimization': {
            'prediction_forecast', 'classification', 'evaluation_ranking',
            'simulation', 'graph_network', 'statistical_inference', 'causal_inference',
        },
        'evaluation_ranking': {
            'prediction_forecast', 'dimension_reduction', 'statistical_inference',
            'causal_inference',
        },
        'simulation': {'prediction_forecast', 'differential_equations', 'optimization'},
        'anomaly_detection': {'dimension_reduction', 'prediction_forecast'},
    }
    for subproblem_index, text in enumerate(subproblems[:10], 1):
        normalized = str(text).lower()
        scores = {
            key: len(pattern.findall(normalized))
            for key, pattern in _PROBLEM_TYPE_PATTERNS.items()
        }
        ranked = [
            {'task_type': key, 'score': score}
            for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if score > 0
        ]
        if not ranked:
            ranked = [{'task_type': fallback_task, 'score': 0}]
        selected_tasks = [ranked[0]['task_type']]
        if any(marker in normalized for marker in composition_markers):
            # A single numbered question may itself be a pipeline (for example
            # "降维并检测异常" or "预测并据此优化"). Preserve at most three
            # explicit task signals instead of mislabelling them as alternatives.
            selected_tasks = [item['task_type'] for item in ranked[:3]]

        def first_position(task_type: str) -> int:
            match = _PROBLEM_TYPE_PATTERNS[task_type].search(normalized)
            return match.start() if match else len(normalized)

        selected_tasks.sort(key=lambda task_type: (first_position(task_type), -scores[task_type]))
        for selected in selected_tasks:
            if len(graph) >= 20:
                break
            dependencies: List[str] = []
            if (
                selected != 'data_requirements'
                and graph
                and any(marker in normalized for marker in flow_markers)
            ):
                dependencies.append(graph[-1]['id'])
            accepted_sources = dependency_sources.get(selected, set())
            if accepted_sources:
                for prior in reversed(graph):
                    if prior['task_type'] in accepted_sources:
                        dependencies.append(prior['id'])
                        break
            io = _TASK_IO.get(selected, _TASK_IO['statistical_inference'])
            evidence = []
            for match in _PROBLEM_TYPE_PATTERNS[selected].finditer(normalized):
                token = match.group(0)
                if token not in evidence:
                    evidence.append(token)
            node = {
                'id': f'task_{len(graph) + 1}',
                'subproblem_index': subproblem_index,
                'text': str(text).strip(),
                'task_type': selected,
                'alternatives': ranked[:3],
                'evidence_tokens': evidence[:8],
                'requires': list(io['requires']),
                'produces': list(io['produces']),
                'depends_on': list(dict.fromkeys(dependencies)),
                'status': 'planned',
            }
            graph.append(node)
    return graph


def _identify_task_type(desc: str) -> str:
    """Identify the core mathematical task.

    优化：将 9 个独立 findall+len 调用压缩为单个 dict 推导，
    减少样板代码并利用预编译的 _PROBLEM_TYPE_PATTERNS。
    """
    scores = {key: len(p.findall(desc)) for key, p in _PROBLEM_TYPE_PATTERNS.items()}

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'evaluation_ranking'


def _extract_variables(desc: str) -> List[str]:
    """Extract potential variables from the description."""
    # Look for patterns like "x m/s", "y kg", numbers with units, or named parameters
    variables = []
    
    for pat in _VAR_PATTERNS:
        for m in pat.finditer(desc):
            variables.append(m.group(0))
    
    # Look for entities
    entities = _ENTITY_PATTERN.findall(desc)
    for e in entities:
        variables.append(f"{e[0]}: ({e[1]})")
    
    return list(dict.fromkeys(variables))[:15]  # dedup, max 15


def _extract_constraints(desc: str) -> List[str]:
    """Extract constraints from the description."""
    constraints = []
    
    # Range constraints
    for m in _RANGE_PATTERN.finditer(desc):
        constraints.append(f"范围约束: {m.group(1)} ~ {m.group(2)} {m.group(3)}")
    
    # At most / at least
    for m in _MAX_PATTERN.finditer(desc):
        constraints.append(f"上限约束: ≤ {m.group(1)}")
    for m in _MIN_PATTERN.finditer(desc):
        constraints.append(f"下限约束: ≥ {m.group(1)}")
    
    # Time interval
    for m in _INTERVAL_PATTERN.finditer(desc):
        constraints.append(f"时间间隔约束: ≥ {m.group(1)} s")
    
    # Spatial
    if '范围内' in desc or '距离' in desc:
        constraints.append("空间距离约束")
    
    return constraints if constraints else ['需从题目中进一步提取']


def _extract_objectives(desc: str) -> List[str]:
    """Extract optimization or analysis objectives."""
    objectives = []
    
    # Maximization
    for m in _MAXIMIZE_PATTERN.finditer(desc):
        objectives.append(f"最大化: {m.group(1).strip()}")
    
    # Minimization
    for m in _MINIMIZE_PATTERN.finditer(desc):
        objectives.append(f"最小化: {m.group(1).strip()}")
    
    # Generic goals
    if '给出' in desc or '求出' in desc or '计算' in desc:
        objectives.append('计算/求解具体数值结果')
    if '策略' in desc or '方案' in desc:
        objectives.append('设计最优策略/方案')
    if '保存到文件' in desc or 'excel' in desc or 'xlsx' in desc:
        objectives.append('输出结果到文件')
    
    return objectives if objectives else ['分析题目并建立数学模型']


def _recommend_model_class(task_type: str, desc: str) -> Dict[str, Any]:
    """Recommend appropriate mathematical model class.

    模型类别字典已提升为模块级常量 _MODEL_CLASSES。
    未匹配到 task_type 时回退到 _DEFAULT_MODEL_CLASS。
    """
    return _MODEL_CLASSES.get(task_type, _DEFAULT_MODEL_CLASS)


def _generate_steps(task_type: str, model_class: Dict) -> List[str]:
    """Generate general modeling steps.

    通用步骤 _COMMON_STEPS 与各类型专属步骤 _TYPE_STEPS 均为模块级常量。
    未匹配到 task_type 时回退到 _DEFAULT_TYPE_STEPS。
    """
    return _COMMON_STEPS + _TYPE_STEPS.get(task_type, _DEFAULT_TYPE_STEPS)


def _generate_code_framework(task_type: str, model_class: Dict) -> str:
    """Generate a general Python code framework.

    框架模板已提升为模块级常量 _CODE_FRAMEWORKS。
    未匹配到 task_type 时回退到 evaluation_ranking 模板。
    """
    return _CODE_FRAMEWORKS.get(task_type, _CODE_FRAMEWORKS['evaluation_ranking']).strip()


def generate_modeling_report(description: str) -> str:
    """Generate a human-readable modeling report."""
    result = analyze_problem(description)
    var_lines = [f"- {v}" for v in result['variables']] if result['variables'] else ['- 未自动提取到，请手动整理']
    lines = [
        f"## 任务类型: {result['task_type']}",
        f"## 推荐模型: {result['model_class']}",
        f"{result['model_description']}",
        "",
        "## 提取的变量与参数",
    ]
    lines.extend(var_lines)
    lines.extend([
        "",
        "## 约束条件",
        *[f"- {c}" for c in result['constraints']],
        "",
        "## 目标",
        *[f"- {o}" for o in result['objectives']],
        "",
        "## 建模步骤",
        *[f"{step}" for step in result['steps']],
        "",
        "## Python 代码框架",
        "```python",
        result['code_framework'],
        "```",
    ])
    return '\n'.join(lines)
