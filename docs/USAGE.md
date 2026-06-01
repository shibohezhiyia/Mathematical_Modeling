# 智能图表 - 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行 Web 界面

```bash
python web/app.py
```

访问 `http://localhost:5000` 即可使用交互式界面。

---

## Python API 使用示例

### 示例 1：端到端分类任务

```python
import pandas as pd
from core.data_module import DataModule
from core.integrated_pipeline import IntegratedPipeline

# 加载数据
df = pd.read_csv('your_data.csv')

# 运行完整流水线
pipeline = IntegratedPipeline(
    target_col='target',
    strategy_preference='accuracy_first',  # 或 'speed_first', 'balanced'
    optimizer='bayesian',
    optimize_hyperparams=True,
    hyperparam_trials=20,
    deep_learning={'enabled': True, 'models': ['torch_mlp']}
)

result = pipeline.run(df)

# 查看结果
print(result.leaderboard)
print(result.decision.recommended_model)
print(result.cv_scores)

# 导出预测
pipeline.export_predictions('predictions.csv', id_col='id')
```

### 示例 2：自动数据清洗

```python
from core.data_module import DataModule
from core.auto_pipeline import AutoMissingPipeline, PipelineConfig

# 配置清洗流程
config = PipelineConfig(
    target_col='target',
    fast_mode=False,
    structural_threshold=0.90,
    drop_col_threshold=0.95
)

pipeline = AutoMissingPipeline(config)
train_df, test_df, report = pipeline.run(df)

print(f"训练集: {train_df.shape}")
print(f"测试集: {test_df.shape if test_df is not None else '无'}")
print(report)
```

### 示例 3：超参数优化

```python
from core.optimizer_factory import OptimizerFactory
from core.modeling_engine import ModelLibrary, TaskType

# 创建优化器
optimizer = OptimizerFactory.create('rl', n_trials=30, cv_folds=3)

# 优化单个模型
result = optimizer.optimize(
    model_key='xgboost',
    X=X_train,
    y=y_train,
    task_type='classification'
)

print(f"最优参数: {result.best_params}")
print(f"最优分数: {result.best_score:.4f}")

# 查看优化历史
for trial in result.optimization_history[:5]:
    print(f"Trial {trial['trial']}: {trial['score']:.4f}")
```

### 示例 4：RL 优化器（DQN 智能搜索）

```python
from core.reinforcement_learning import RLOptimizer

optimizer = RLOptimizer(
    n_trials=50,
    n_parallel=4,
    hidden_dim=128,
    subset_schedule=[(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
)

result = optimizer.optimize('lightgbm', X, y, 'classification')
print(f"RL 最优分数: {result.best_score:.4f}")
print(f"采样器类型: {result.sampler_type}")
```

### 示例 5：自定义搜索空间

```python
from core.search_space import SearchSpace

space = SearchSpace({
    'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart']},
    'use_gpu': {'type': 'bool'},
})

# 随机采样
params = space.sample(random_state=42)
print(params)

# 构建离散候选（用于 RL/GA）
candidates = space.build_candidates(n=8)
print(candidates)
```

### 示例 6：性能调度（自动策略选择）

```python
from core.performance_scheduler import PerformanceScheduler

scheduler = PerformanceScheduler()
plan = scheduler.schedule(df)

print(f"推荐策略: {plan.strategy.value}")
print(f"并行数: {plan.n_jobs}")
print(f"使用 GPU: {plan.use_gpu}")
print(f"交叉验证折数: {plan.cv_folds}")
print(f"最大模型数: {plan.max_models}")

# 获取完整报告
print(scheduler.get_recommendation_text())
```

### 示例 7：模型解释（SHAP + LIME）

```python
from core.explainability import ExplainabilityEngine

engine = ExplainabilityEngine()

# 全局特征重要性
importance = engine.explain_model(model, X_test, method='shap')
print(importance)

# 单样本局部解释
local = engine.explain_instance(
    model, X_test, instance_index=0, method='lime'
)
print(local)

# 生成报告
engine.generate_report('explain_report.txt')
```

### 示例 8：公平性分析

```python
from core.fairness import FairnessEngine

engine = FairnessEngine()
report = engine.analyze(
    model, X_test, y_test,
    sensitive_attr='gender',
    task_type='classification'
)

print(f"人口统计平等差: {report.demographic_parity_diff:.4f}")
print(f"均等赔率差: {report.equalized_odds_diff:.4f}")

# 各群体指标
for group, metrics in report.group_metrics.items():
    print(f"群体 {group}: 准确率={metrics['accuracy']:.3f}")
```

### 示例 9：深度学习模型

```python
from core.deep_learning import TorchMLP, TorchGRU
from core.nas import TorchNAS

# PyTorch MLP
mlp = TorchMLP(
    hidden_dims=[128, 64],
    dropout=0.3,
    epochs=50,
    task_type='classification'
)
mlp.fit(X_train, y_train)
preds = mlp.predict(X_test)

# GRU 时序模型
gru = TorchGRU(hidden_dim=64, epochs=30, task_type='regression')
gru.fit(X_train, y_train)

# 神经架构搜索（NAS）
nas = TorchNAS(task_type='classification', n_candidates=4, epochs=10)
nas.fit(X_train, y_train)
print(f"最优架构: {nas.best_params_}")
```

### 示例 10：多模态数据

```python
from core.multimodal import ImageResNet, TextBERT

# 图像分类
img_df = pd.DataFrame({
    'image_path': ['path/to/img1.jpg', 'path/to/img2.jpg'],
    'label': [0, 1]
})
img_model = ImageResNet(task_type='classification', epochs=5)
img_model.fit(img_df)

# 文本分类
text_df = pd.DataFrame({
    'text': ['这是一个好评', '这是一个差评'],
    'label': [1, 0]
})
text_model = TextBERT(task_type='classification', epochs=2)
text_model.fit(text_df)
```

### 示例 11：大文件分块读取

```python
from core.data_module import DataLoader

loader = DataLoader()

# 自动检测大文件并分块读取
df = loader.load('large_file.csv', auto_chunk=True)

# 手动分块读取
df = loader.load_chunked('very_large.csv', chunk_size=100000)
```

### 示例 12：结果缓存

```python
from core.result_cache import get_result_cache, cached

cache = get_result_cache()

# 直接使用缓存
cache.set('my_key', {'score': 0.95})
result = cache.get('my_key')

# 装饰器自动缓存
@cached(ttl_seconds=3600)
def expensive_evaluation(model, X, y):
    from sklearn.model_selection import cross_val_score
    return cross_val_score(model, X, y, cv=5).mean()

score = expensive_evaluation(model, X, y)  # 首次计算
score = expensive_evaluation(model, X, y)  # 直接命中缓存
```

### 示例 13：并行计算

```python
from core.accelerators import ParallelEngine, parallelize

# 并行引擎
engine = ParallelEngine(n_jobs=-1, backend='auto')
results = engine.map(my_function, data_list)

# 装饰器方式
@parallelize(n_jobs=4, backend='thread')
def process_batch(items):
    return [item * 2 for item in items]

results = process_batch([1, 2, 3, 4, 5])
```

### 示例 14：GPU 加速

```python
from core.accelerators import get_gpu_manager, auto_gpu_model
from xgboost import XGBClassifier

# 检查 GPU 状态
gpu = get_gpu_manager()
print(f"GPU 可用: {gpu.available}")
print(f"显存信息: {gpu.get_memory_info()}")

# 自动启用 GPU 加速
model = auto_gpu_model(XGBClassifier, use_gpu=True, n_estimators=100)

# 显存检查
if gpu.check_memory(min_free_mb=1000):
    model.fit(X_train, y_train)
```

---

## Web API 端点

### 训练模型

```bash
curl -X POST http://localhost:5000/api/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "target_col": "target",
    "optimizer": "bayesian",
    "deep_learning": {"enabled": true},
    "auto_decision_mode": "balanced"
  }'
```

### 模型解释

```bash
curl -X POST http://localhost:5000/api/model/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_index": 0,
    "instance_index": 5,
    "method": "shap"
  }'
```

### 获取结果

```bash
curl http://localhost:5000/api/model/result
```

---

## 配置参考

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy_preference` | str | `'balanced'` | 策略偏好：accuracy_first/speed_first/stability_first/balanced |
| `optimizer` | str | `'bayesian'` | 超参优化器：bayesian/rl/random/hyperband/genetic/auto |
| `n_splits` | int | 5 | K折交叉验证折数 |
| `optimize_hyperparams` | bool | False | 是否启用超参优化 |
| `hyperparam_trials` | int | 20 | 超参搜索次数 |
| `auto_sample` | bool | True | 大数据自动降采样 |
| `max_samples` | int | 50000 | 最大样本数 |
| `deep_learning` | dict | None | 深度学习配置 |
| `ensemble` | str | `'weighted'` | 融合策略 |
| `feature_selection` | str | `'mi'` | 特征选择策略 |
| `explainability` | bool | False | 是否生成模型解释 |
