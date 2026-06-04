# 建模比赛智能分析引擎

面向数据建模比赛的端到端智能分析引擎，涵盖数据加载、类型识别、缺失值智能分析、自动性能调度、自动编码、特征选择、**K折交叉验证**、**多模型并行训练**、**模型融合**等全链路能力。

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
│   ├── integrated_pipeline.py     # 集成流水线 V2
│   └── workspace_manager.py       # 工作空间管理器（磁盘隔离）
├── utils/                         # 工具函数
│   ├── __init__.py
│   └── helpers.py
├── tests/                         # 单元测试（82项全部通过）
│   ├── test_data_module.py        # 16项
│   ├── test_missing_engine.py     # 22项
│   ├── test_workspace.py          # 13项
│   └── test_modeling_engine.py    # 31项 ⭐新增
├── workspace/                     # 工作目录（临时/缓存/报告/数据）
├── data/                          # 数据目录
├── demo.py
├── demo_missing.py
├── requirements.txt
└── README.md
```

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
