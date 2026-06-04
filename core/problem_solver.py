"""
Universal Math Modeling Problem Solver

Analyzes any text-based modeling problem and recommends
a general modeling framework, not domain-specific templates.
"""
import re
# 预编译正则表达式，避免每次调用时重新编译
_PROBLEM_TYPE_PATTERNS = {
    'optimization': re.compile(r'最大|最小|最优|优化|分配|调度|规划|配置|路径|成本|利润|效费|节约|资源|约束|方案|策略|尽可能|尽量|投放点|起爆点|航向|飞行方向|速度'),
    'differential_equations': re.compile(r'变化|增长|传播|扩散|动态|演化|速率|随时间|导数|微分|方程|运动|轨迹|弹道'),
    'prediction_forecast': re.compile(r'预测|预报|估计|趋势|未来|下一|将|会达到|销量|人口|疫情'),
    'classification': re.compile(r'分类|识别|判别|诊断|判断|是否|好坏|等级|类型'),
    'clustering': re.compile(r'聚类|分组|划分|聚成|相似|类别|群落'),
    'simulation': re.compile(r'模拟|仿真|蒙特卡洛|随机|概率|风险|不确定|抽样'),
    'graph_network': re.compile(r'网络|图|节点|边|路径|流量|连接|路线|拓扑|最短|联通'),
    'statistical_inference': re.compile(r'显著|相关|回归|检验|置信|假设|分布|频率|统计'),
    'evaluation_ranking': re.compile(r'评价|评估|排名|排序|指标|得分|综合|优劣'),
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
from typing import Any, Dict, List


def analyze_problem(description: str) -> Dict[str, Any]:
    """Analyze a modeling problem and return a general framework."""
    desc = description.lower()
    
    # Step 1: Identify the core task type
    task_type = _identify_task_type(desc)
    
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
    
    return {
        'task_type': task_type,
        'model_class': model_class['name'],
        'model_description': model_class['description'],
        'variables': variables,
        'constraints': constraints,
        'objectives': objectives,
        'steps': steps,
        'code_framework': code,
        'confidence': model_class['confidence'],
    }


def _identify_task_type(desc: str) -> str:
    """Identify the core mathematical task."""
    scores = {
        'optimization': 0,
        'differential_equations': 0,
        'prediction_forecast': 0,
        'classification': 0,
        'clustering': 0,
        'simulation': 0,
        'graph_network': 0,
        'statistical_inference': 0,
        'evaluation_ranking': 0,
    }
    
    # Optimization
    scores['optimization'] += len(_PROBLEM_TYPE_PATTERNS['optimization'].findall(desc))
    
    # Differential equations / dynamics
    scores['differential_equations'] += len(_PROBLEM_TYPE_PATTERNS['differential_equations'].findall(desc))
    
    # Prediction / forecasting
    scores['prediction_forecast'] += len(_PROBLEM_TYPE_PATTERNS['prediction_forecast'].findall(desc))
    
    # Classification / recognition
    scores['classification'] += len(_PROBLEM_TYPE_PATTERNS['classification'].findall(desc))
    
    # Clustering / grouping
    scores['clustering'] += len(_PROBLEM_TYPE_PATTERNS['clustering'].findall(desc))
    
    # Simulation
    scores['simulation'] += len(_PROBLEM_TYPE_PATTERNS['simulation'].findall(desc))
    
    # Graph / network
    scores['graph_network'] += len(_PROBLEM_TYPE_PATTERNS['graph_network'].findall(desc))
    
    # Statistical inference
    scores['statistical_inference'] += len(_PROBLEM_TYPE_PATTERNS['statistical_inference'].findall(desc))
    
    # Evaluation / ranking
    scores['evaluation_ranking'] += len(_PROBLEM_TYPE_PATTERNS['evaluation_ranking'].findall(desc))
    
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'evaluation_ranking'


def _extract_variables(desc: str) -> List[str]:
    """Extract potential variables from the description."""
    # Look for patterns like "x m/s", "y kg", numbers with units, or named parameters
    variables = []
    
    # Number + unit patterns
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(m/s|km/h|kg|t|s|min|h|m|km|℃|°|度|个|架|枚)',
        r'([\u4e00-\u9fa5]+?)(?:分别为|为|是|等于|约|大概)\s*(\d+(?:\.\d+)?)',
        r'坐标[为是]\s*\(?\s*([\d\-,\.\s]+)\s*\)?',
    ]
    
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
    """Recommend appropriate mathematical model class."""
    classes = {
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
        'evaluation_ranking': {
            'name': '评价与排名模型',
            'description': '层次分析法(AHP)、TOPSIS、熵权法、模糊综合评价',
            'confidence': 75,
        },
    }
    return classes.get(task_type, classes['evaluation_ranking'])


def _generate_steps(task_type: str, model_class: Dict) -> List[str]:
    """Generate general modeling steps."""
    common_steps = [
        '1. 问题理解：明确已知条件、决策变量、约束条件和目标函数',
        '2. 符号定义：为所有变量、参数、集合建立规范的数学符号',
        '3. 基本假设：列出简化假设（如忽略次要因素、理想化条件）',
        '4. 数据整理：提取题目中所有数值参数，建立参数表',
    ]
    
    type_steps = {
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
        'evaluation_ranking': [
            '5. 建立指标体系：目标层、准则层、指标层',
            '6. 数据标准化：正向化、归一化、Z-score',
            '7. 确定权重：AHP成对比较、熵权法（客观）、组合赋权',
            '8. 综合评价：TOPSIS（距理想解距离）、灰色关联、模糊综合',
            '9. 敏感性分析：权重变化对排名的影响',
            '10. 结果输出：排名表、雷达图、权重分布图',
        ],
    }
    
    return common_steps + type_steps.get(task_type, type_steps['evaluation_ranking'])


def _generate_code_framework(task_type: str, model_class: Dict) -> str:
    """Generate a general Python code framework."""
    frameworks = {
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
    }
    
    return frameworks.get(task_type, frameworks['evaluation_ranking']).strip()


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
