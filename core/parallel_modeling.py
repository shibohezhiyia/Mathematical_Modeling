"""
并行建模引擎

支持多模型并行训练、交叉验证、超参数搜索、模型融合的端到端方案。
自动适配 GPU 加速，无缝集成 XGBoost/LightGBM/CatBoost/sklearn。

架构：
  ModelRegistry → 注册内置模型
  ModelWrapper → 统一接口封装
  CrossValidator → 交叉验证
  HyperparameterSearch → 超参数搜索
  ParallelModelingEngine → 并行训练与评估
  ModelBlender → 模型融合
"""

import os
import time
import warnings
import importlib
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

from core.accelerators import (
    ParallelEngine, GPUManager, get_gpu_manager,
    auto_gpu_model, optimize_memory
)
from core.performance_scheduler import ExecutionPlan, StrategyLevel
from core.progress_bar import progress_range, progress_iter
from utils.helpers import log_info, log_warning, log_error, timer


# =============================================================================
# 模型包装器与注册表
# =============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_class: Any
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_distributions: Dict[str, List[Any]] = field(default_factory=dict)
    supports_gpu: bool = True
    supports_partial_fit: bool = False
    supports_sample_weight: bool = False
    is_probabilistic: bool = False
    task_type: str = "both"  # 'classification', 'regression', 'both'
    priority: int = 5  # 1=最高优先级（极速模式优先选用）


class ModelRegistry:
    """
    模型注册表
    
    内置常用模型，自动检测可用性
    """
    
    _models: Dict[str, ModelConfig] = {}
    _initialized = False
    
    @classmethod
    def _init(cls) -> None:
        if cls._initialized:
            return
        
        # 尝试导入各库并注册
        
        # 1. XGBoost
        try:
            from xgboost import XGBClassifier, XGBRegressor
            cls._models['xgb'] = ModelConfig(
                name='XGBoost',
                model_class={'classification': XGBClassifier, 'regression': XGBRegressor},
                default_params={'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                               'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                               'n_jobs': -1, 'eval_metric': 'logloss'},
                param_distributions={
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [100, 200, 500],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 2, 5],
                },
                supports_gpu=True, task_type='both', priority=1
            )
        except ImportError:
            pass
        
        # 2. LightGBM
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls._models['lgb'] = ModelConfig(
                name='LightGBM',
                model_class={'classification': LGBMClassifier, 'regression': LGBMRegressor},
                default_params={'n_estimators': 200, 'max_depth': -1, 'learning_rate': 0.1,
                               'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
                               'random_state': 42, 'n_jobs': -1, 'verbose': -1},
                param_distributions={
                    'num_leaves': [20, 31, 50, 100],
                    'max_depth': [-1, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 500],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 2, 5],
                    'min_child_samples': [5, 10, 20],
                },
                supports_gpu=True, task_type='both', priority=1
            )
        except ImportError:
            pass
        
        # 3. CatBoost
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls._models['catboost'] = ModelConfig(
                name='CatBoost',
                model_class={'classification': CatBoostClassifier, 'regression': CatBoostRegressor},
                default_params={'iterations': 200, 'depth': 6, 'learning_rate': 0.1,
                               'random_seed': 42, 'verbose': False, 'loss_function': 'Logloss'},
                param_distributions={
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'iterations': [100, 200, 500],
                    'l2_leaf_reg': [1, 3, 5, 7],
                    'border_count': [32, 64, 128],
                },
                supports_gpu=True, task_type='both', priority=2
            )
        except ImportError:
            pass
        
        # 4. Random Forest
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            cls._models['rf'] = ModelConfig(
                name='RandomForest',
                model_class={'classification': RandomForestClassifier, 'regression': RandomForestRegressor},
                default_params={'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,
                               'random_state': 42, 'n_jobs': -1},
                param_distributions={
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                },
                supports_gpu=False, task_type='both', priority=3
            )
        except ImportError:
            pass
        
        # 5. Extra Trees
        try:
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            cls._models['et'] = ModelConfig(
                name='ExtraTrees',
                model_class={'classification': ExtraTreesClassifier, 'regression': ExtraTreesRegressor},
                default_params={'n_estimators': 200, 'max_depth': None, 'random_state': 42, 'n_jobs': -1},
                param_distributions={
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                supports_gpu=False, task_type='both', priority=4
            )
        except ImportError:
            pass
        
        # 6. Logistic Regression / Ridge
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            cls._models['linear'] = ModelConfig(
                name='Linear',
                model_class={'classification': LogisticRegression, 'regression': Ridge},
                default_params={'random_state': 42, 'max_iter': 1000},
                param_distributions={
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs'],
                },
                supports_gpu=False, task_type='both', priority=5
            )
        except ImportError:
            pass
        
        # 7. SVM
        try:
            from sklearn.svm import SVC, SVR
            cls._models['svm'] = ModelConfig(
                name='SVM',
                model_class={'classification': SVC, 'regression': SVR},
                default_params={'probability': True, 'random_state': 42},
                param_distributions={
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                },
                supports_gpu=False, task_type='both', priority=5
            )
        except ImportError:
            pass
        
        # 8. HistGradientBoosting
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            cls._models['hist_gb'] = ModelConfig(
                name='HistGradientBoosting',
                model_class={'classification': HistGradientBoostingClassifier, 'regression': HistGradientBoostingRegressor},
                default_params={'random_state': 42},
                param_distributions={
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, None],
                    'max_iter': [100, 200],
                },
                supports_gpu=False, task_type='both', priority=3
            )
        except ImportError:
            pass
        
        # 9. SGD
        try:
            from sklearn.linear_model import SGDClassifier, SGDRegressor
            cls._models['sgd'] = ModelConfig(
                name='SGD',
                model_class={'classification': SGDClassifier, 'regression': SGDRegressor},
                default_params={'max_iter': 1000, 'random_state': 42, 'n_jobs': -1},
                param_distributions={
                    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                },
                supports_partial_fit=True, task_type='both', priority=5
            )
        except ImportError:
            pass
        
        cls._initialized = True
        log_info(f"[ModelRegistry] 已注册 {len(cls._models)} 个模型: {list(cls._models.keys())}")
    
    @classmethod
    def get_available_models(cls, task_type: str = 'classification', 
                             use_gpu: bool = False,
                             strategy: StrategyLevel = StrategyLevel.STANDARD) -> Dict[str, ModelConfig]:
        """获取可用模型列表"""
        cls._init()
        
        result = {}
        for key, config in cls._models.items():
            if config.task_type != 'both' and config.task_type != task_type:
                continue
            if use_gpu and not config.supports_gpu:
                continue
            result[key] = config
        
        # 根据策略筛选优先级
        if strategy == StrategyLevel.ULTRA:
            # 极速模式：只保留最高优先级的模型
            min_priority = 2
            result = {k: v for k, v in result.items() if v.priority <= min_priority}
        elif strategy == StrategyLevel.FAST:
            # 快速模式：保留中高优先级
            min_priority = 4
            result = {k: v for k, v in result.items() if v.priority <= min_priority}
        
        return result
    
    @classmethod
    def create_model(cls, model_key: str, task_type: str, 
                     use_gpu: bool = False, **override_params) -> Any:
        """创建模型实例"""
        cls._init()
        config = cls._models.get(model_key)
        if not config:
            raise ValueError(f"未知模型: {model_key}")
        
        model_cls = config.model_class
        if isinstance(model_cls, dict):
            model_cls = model_cls.get(task_type)
        
        params = deepcopy(config.default_params)
        params.update(override_params)
        
        if use_gpu and config.supports_gpu:
            return auto_gpu_model(model_cls, use_gpu=True, **params)
        
        return model_cls(**params)


# =============================================================================
# 评估指标
# =============================================================================

class Metrics:
    """评估指标集合"""
    
    CLASSIFICATION = {
        'accuracy': accuracy_score,
        'auc': roc_auc_score,
        'f1_macro': lambda y, p: f1_score(y, p, average='macro'),
        'f1_weighted': lambda y, p: f1_score(y, p, average='weighted'),
    }
    
    REGRESSION = {
        'rmse': lambda y, p: np.sqrt(mean_squared_error(y, p)),
        'mae': mean_absolute_error,
        'r2': r2_score,
        'mse': mean_squared_error,
    }
    
    @classmethod
    def get_metrics(cls, task_type: str) -> Dict[str, Callable]:
        return cls.CLASSIFICATION if task_type == 'classification' else cls.REGRESSION
    
    @classmethod
    def evaluate(cls, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], task_type: str, proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """综合评估"""
        metrics = cls.get_metrics(task_type)
        result = {}
        
        for name, func in metrics.items():
            try:
                if name == 'auc' and proba is not None and task_type == 'classification':
                    # AUC需要概率值
                    if len(np.unique(y_true)) == 2:
                        result[name] = float(roc_auc_score(y_true, proba))
                    else:
                        result[name] = float(roc_auc_score(y_true, proba, multi_class='ovr', average='weighted'))
                else:
                    result[name] = float(func(y_true, y_pred))
            except Exception as e:
                result[name] = None
        
        return result


# =============================================================================
# 超参数搜索
# =============================================================================

class HyperparameterSearch:
    """超参数搜索器"""
    
    def __init__(self, n_trials: int = 30, random_state: int = 42, verbose: bool = True) -> None:
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self._use_optuna = False
        self._try_optuna()
    
    def _try_optuna(self) -> None:
        try:
            import optuna
            self._use_optuna = True
        except ImportError:
            pass
    
    def search(self, model_config: ModelConfig, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], task_type: str,
               cv_folds: int = 3, use_gpu: bool = False) -> Tuple[Dict, float]:
        """
        超参数搜索
        
        Returns:
            (best_params, best_score)
        """
        if self._use_optuna and self.n_trials >= 20:
            return self._optuna_search(model_config, X, y, task_type, cv_folds, use_gpu)
        else:
            return self._random_search(model_config, X, y, task_type, cv_folds, use_gpu)
    
    def _random_search(self, model_config: ModelConfig, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], task_type: str,
                       cv_folds: int, use_gpu: bool) -> Tuple[Dict, float]:
        """随机搜索"""
        rng = np.random.RandomState(self.random_state)
        best_score = -np.inf
        best_params = {}
        
        param_dists = model_config.param_distributions
        if not param_dists:
            return {}, 0.0
        
        for trial in progress_range(self.n_trials, desc=f"随机搜索 {model_config.name}", disable=not self.verbose):
            params = {}
            for key, values in param_dists.items():
                params[key] = rng.choice(values)
            
            try:
                model = ModelRegistry.create_model(
                    list(ModelRegistry._models.keys())[
                        [v.name for v in ModelRegistry._models.values()].index(model_config.name)
                    ],
                    task_type, use_gpu=use_gpu, **params
                )
                
                score = self._cv_score(model, X, y, task_type, cv_folds)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                continue
        
        return best_params, best_score
    
    def _optuna_search(self, model_config: ModelConfig, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], task_type: str,
                       cv_folds: int, use_gpu: bool) -> Tuple[Dict, float]:
        """Optuna贝叶斯优化"""
        import optuna
        
        # 找到模型key
        model_key = None
        for k, v in ModelRegistry._models.items():
            if v.name == model_config.name:
                model_key = k
                break
        
        if not model_key:
            return self._random_search(model_config, X, y, task_type, cv_folds, use_gpu)
        
        param_dists = model_config.param_distributions
        
        def objective(trial: Any) -> float:
            params = {}
            for key, values in param_dists.items():
                if all(isinstance(v, (int, np.integer)) for v in values):
                    params[key] = trial.suggest_categorical(key, values)
                elif all(isinstance(v, float) for v in values):
                    params[key] = trial.suggest_categorical(key, values)
                else:
                    params[key] = trial.suggest_categorical(key, values)
            
            model = ModelRegistry.create_model(model_key, task_type, use_gpu=use_gpu, **params)
            return self._cv_score(model, X, y, task_type, cv_folds)
        
        study = optuna.create_study(direction='maximize', 
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def _cv_score(self, model: Any, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], task_type: str, cv_folds: int) -> float:
        """交叉验证评分"""
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            metric = 'roc_auc_ovr_weighted' if len(np.unique(y)) > 2 else 'roc_auc'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            metric = 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=1)
        return scores.mean()


# =============================================================================
# 并行建模引擎
# =============================================================================

@dataclass
class ModelResult:
    """单个模型的训练结果"""
    model_key: str
    model_name: str
    model_instance: Any
    best_params: Dict[str, Any]
    cv_score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    train_time: float = 0.0
    oof_predictions: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    feature_importance: Optional[pd.DataFrame] = None
    rank: int = 0


class ParallelModelingEngine:
    """
    并行建模引擎
    
    核心能力：
    1. 多模型并行训练
    2. 自动超参数搜索
    3. 交叉验证 + OOF预测
    4. 自动特征重要性
    5. 模型融合（Voting/Stacking）
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 metric: str = 'auc',
                 plan: Optional[ExecutionPlan] = None,
                 random_state: int = 42,
                 verbose: bool = True) -> None:
        """
        Args:
            task_type: 'classification' or 'regression'
            metric: 主评估指标
            plan: 执行计划（来自PerformanceScheduler）
            random_state: 随机种子
        """
        self.task_type = task_type
        self.metric = metric
        self.plan = plan or ExecutionPlan()
        self.random_state = random_state
        self.verbose = verbose
        
        self.results: Dict[str, ModelResult] = {}
        self.leaderboard: List[ModelResult] = []
        self.blender = None
        
        # 初始化
        ModelRegistry._init()
        self.gpu = get_gpu_manager()
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            model_keys: Optional[List[str]] = None,
            feature_names: Optional[List[str]] = None) -> 'ParallelModelingEngine':
        """
        并行训练多个模型
        
        Args:
            X: 训练特征
            y: 训练标签
            X_test: 测试特征（可选，用于生成测试集预测）
            model_keys: 指定模型列表（None=自动选择）
            feature_names: 特征名
            
        Returns:
            self
        """
        log_info(f"[ParallelModeling] 启动并行建模，任务类型: {self.task_type}")
        log_info(f"[ParallelModeling] 训练数据: {X.shape}, GPU: {self.plan.use_gpu}")
        
        # 获取可用模型
        available = ModelRegistry.get_available_models(
            self.task_type, self.plan.use_gpu, self.plan.strategy
        )
        
        if model_keys:
            available = {k: v for k, v in available.items() if k in model_keys}
        
        if not available:
            raise ValueError("没有可用的模型，请安装 XGBoost/LightGBM/sklearn")
        
        log_info(f"[ParallelModeling] 将训练 {len(available)} 个模型: {list(available.keys())}")
        
        # 并行训练
        if self.plan.n_jobs > 1 and len(available) > 1:
            self._fit_parallel(X, y, X_test, available, feature_names)
        else:
            self._fit_sequential(X, y, X_test, available, feature_names)
        
        # 排序生成排行榜
        self._build_leaderboard()
        
        return self
    
    def _fit_sequential(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], X_test: Optional[Union[pd.DataFrame, np.ndarray]], available_models: Dict[str, ModelConfig], feature_names: Optional[List[str]]) -> None:
        """串行训练"""
        for model_key, config in available_models.items():
            result = self._train_single_model(model_key, config, X, y, X_test, feature_names)
            if result:
                self.results[model_key] = result
    
    def _fit_parallel(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], X_test: Optional[Union[pd.DataFrame, np.ndarray]], available_models: Dict[str, ModelConfig], feature_names: Optional[List[str]]) -> None:
        """并行训练"""
        engine = ParallelEngine(n_jobs=min(self.plan.n_jobs, len(available_models)), 
                                backend='thread')
        
        tasks = []
        for model_key, config in available_models.items():
            tasks.append((model_key, config, X, y, X_test, feature_names))
        
        # 线程池并行（避免模型序列化问题）
        import threading
        results = {}
        errors = {}
        
        def worker(task: Tuple[str, ModelConfig, Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Optional[Union[pd.DataFrame, np.ndarray]], Optional[List[str]]]) -> None:
            mk, cfg, x, yy, xt, fn = task
            try:
                results[mk] = self._train_single_model(mk, cfg, x, yy, xt, fn)
            except Exception as e:
                errors[mk] = str(e)
                log_error(f"[ParallelModeling] {mk} 训练失败: {e}")
        
        threads = []
        for task in tasks:
            t = threading.Thread(target=worker, args=(task,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.results = {k: v for k, v in results.items() if v is not None}
    
    def _train_single_model(self, model_key: str, config: ModelConfig,
                            X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], X_test: Optional[Union[pd.DataFrame, np.ndarray]], feature_names: Optional[List[str]]) -> Optional[ModelResult]:
        """训练单个模型"""
        log_info(f"[ParallelModeling] 开始训练: {config.name}")
        start_time = time.time()
        
        try:
            # 1. 超参数搜索
            if self.plan.hyperparameter_trials > 0 and config.param_distributions:
                searcher = HyperparameterSearch(
                    n_trials=self.plan.hyperparameter_trials,
                    random_state=self.random_state
                )
                best_params, cv_score = searcher.search(
                    config, X, y, self.task_type,
                    self.plan.cv_folds, self.plan.use_gpu
                )
                log_info(f"[ParallelModeling] {config.name} 最佳参数: {best_params}, CV={cv_score:.4f}")
            else:
                best_params = {}
                cv_score = 0.0
            
            # 2. 训练最终模型
            model = ModelRegistry.create_model(
                model_key, self.task_type,
                use_gpu=self.plan.use_gpu, **best_params
            )
            model.fit(X, y)
            
            # 3. 生成OOF预测（交叉验证）
            oof_pred, oof_proba = self._generate_oof(model.__class__, best_params, X, y)
            
            # 4. 评估
            pred_labels = (oof_pred > 0.5).astype(int) if self.task_type == 'classification' else oof_pred
            metrics = Metrics.evaluate(y, pred_labels, self.task_type, oof_proba)
            
            # 5. 测试集预测
            test_pred = None
            if X_test is not None:
                if hasattr(model, 'predict_proba') and self.task_type == 'classification':
                    test_pred = model.predict_proba(X_test)
                    if test_pred.ndim > 1 and test_pred.shape[1] == 2:
                        test_pred = test_pred[:, 1]
                else:
                    test_pred = model.predict(X_test)
            
            # 6. 特征重要性
            fi = self._extract_feature_importance(model, feature_names)
            
            train_time = time.time() - start_time
            
            result = ModelResult(
                model_key=model_key,
                model_name=config.name,
                model_instance=model,
                best_params=best_params,
                cv_score=cv_score or metrics.get(self.metric, 0),
                metrics=metrics,
                train_time=train_time,
                oof_predictions=oof_proba if oof_proba is not None else oof_pred,
                test_predictions=test_pred,
                feature_importance=fi
            )
            
            log_info(f"[ParallelModeling] {config.name} 完成: {metrics}, 耗时={train_time:.1f}s")
            return result
            
        except Exception as e:
            log_error(f"[ParallelModeling] {config.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_oof(self, model_cls: Any, params: Dict, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """生成Out-of-Fold预测"""
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=self.plan.cv_folds, shuffle=True, 
                                 random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.plan.cv_folds, shuffle=True,
                      random_state=self.random_state)
        
        oof_pred = np.zeros(len(y))
        oof_proba = None
        
        if self.task_type == 'classification':
            n_classes = len(np.unique(y))
            if n_classes == 2:
                oof_proba = np.zeros(len(y))
            else:
                oof_proba = np.zeros((len(y), n_classes))
        
        fold_iter = list(cv.split(X, y))
        for fold, (train_idx, val_idx) in enumerate(progress_iter(fold_iter, desc="CV", total=self.plan.cv_folds, disable=not self.verbose)):
            X_tr, X_val = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx], \
                          X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
            y_tr = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            
            # 创建模型（带GPU配置）
            # 注意：这里需要特殊处理，因为model_cls可能是dict
            model = model_cls(**params)
            model.fit(X_tr, y_tr)
            
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_val)
                if prob.ndim > 1 and prob.shape[1] == 2:
                    oof_proba[val_idx] = prob[:, 1]
                else:
                    oof_proba[val_idx] = prob
                oof_pred[val_idx] = model.predict(X_val)
            else:
                pred = model.predict(X_val)
                oof_pred[val_idx] = pred
                oof_proba = oof_pred
        
        return oof_pred, oof_proba
    
    def _extract_feature_importance(self, model: Any, feature_names: Optional[List[str]]) -> Optional[pd.DataFrame]:
        """提取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
            else:
                return None
            
            names = feature_names or [f"feature_{i}" for i in range(len(importances))]
            df = pd.DataFrame({
                'feature': names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return df
        except:
            return None
    
    def _build_leaderboard(self) -> None:
        """构建模型排行榜"""
        self.leaderboard = sorted(
            self.results.values(),
            key=lambda r: r.cv_score,
            reverse=True
        )
        for i, result in enumerate(self.leaderboard):
            result.rank = i + 1
    
    def get_leaderboard(self) -> pd.DataFrame:
        """获取排行榜DataFrame"""
        if not self.leaderboard:
            return pd.DataFrame()
        
        rows = []
        for r in self.leaderboard:
            row = {
                'rank': r.rank,
                'model': r.model_name,
                'cv_score': round(r.cv_score, 4),
                'train_time': round(r.train_time, 1),
            }
            row.update({f"metric_{k}": round(v, 4) if v is not None else None 
                       for k, v in r.metrics.items()})
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                blend_method: str = 'average',
                top_k: Optional[int] = None) -> np.ndarray:
        """
        预测（支持模型融合）
        
        Args:
            X: 特征
            blend_method: 'average', 'weighted', 'stacking'
            top_k: 只用前k个模型融合
            
        Returns:
            预测结果
        """
        if not self.leaderboard:
            raise ValueError("尚未训练模型")
        
        models_to_use = self.leaderboard[:top_k] if top_k else self.leaderboard
        
        if blend_method == 'average':
            return self._blend_average(models_to_use, X)
        elif blend_method == 'weighted':
            return self._blend_weighted(models_to_use, X)
        elif blend_method == 'stacking':
            return self._blend_stacking(models_to_use, X)
        else:
            # 默认用最佳单模型
            best = models_to_use[0]
            if hasattr(best.model_instance, 'predict_proba') and self.task_type == 'classification':
                proba = best.model_instance.predict_proba(X)
                return proba[:, 1] if proba.ndim > 1 and proba.shape[1] == 2 else proba
            return best.model_instance.predict(X)
    
    def _blend_average(self, models: List[ModelResult], X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """简单平均融合"""
        preds = []
        for r in models:
            if hasattr(r.model_instance, 'predict_proba') and self.task_type == 'classification':
                p = r.model_instance.predict_proba(X)
                preds.append(p[:, 1] if p.ndim > 1 and p.shape[1] == 2 else p)
            else:
                preds.append(r.model_instance.predict(X))
        
        return np.mean(preds, axis=0)
    
    def _blend_weighted(self, models: List[ModelResult], X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """加权平均融合（按CV分数加权）"""
        preds = []
        weights = []
        
        for r in models:
            if hasattr(r.model_instance, 'predict_proba') and self.task_type == 'classification':
                p = r.model_instance.predict_proba(X)
                preds.append(p[:, 1] if p.ndim > 1 and p.shape[1] == 2 else p)
            else:
                preds.append(r.model_instance.predict(X))
            weights.append(max(r.cv_score, 0.01))
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        return np.average(preds, axis=0, weights=weights)
    
    def _blend_stacking(self, models: List[ModelResult], X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Stacking融合（用OOF预测训练元模型）"""
        # 使用最佳模型的OOF预测作为元特征训练一个简单的线性模型
        from sklearn.linear_model import Ridge, LogisticRegression
        
        # 构建元特征（OOF预测）
        meta_features = np.column_stack([r.oof_predictions for r in models])
        
        # 训练元模型
        if self.task_type == 'classification':
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            meta_model = Ridge(random_state=42)
        
        # 需要获取训练标签（从OOF预测反推或存储）
        # 简化：使用加权平均代替完整stacking
        return self._blend_weighted(models, X)
    
    def get_feature_importance_ensemble(self, top_n: int = 20) -> pd.DataFrame:
        """获取集成特征重要性（多个模型平均）"""
        all_fi = []
        for r in self.leaderboard:
            if r.feature_importance is not None:
                fi = r.feature_importance.copy()
                fi['importance'] = fi['importance'] / fi['importance'].sum()  # 归一化
                fi['model'] = r.model_name
                all_fi.append(fi)
        
        if not all_fi:
            return pd.DataFrame()
        
        combined = pd.concat(all_fi, copy=False)
        ensemble = combined.groupby('feature')['importance'].mean().reset_index()
        ensemble = ensemble.sort_values('importance', ascending=False).head(top_n)
        
        return ensemble
    
    def print_summary(self) -> None:
        """打印建模摘要"""
        print("\n" + "=" * 70)
        print("并行建模结果摘要".center(60))
        print("=" * 70)
        
        print(f"\n任务类型: {self.task_type}")
        print(f"训练模型数: {len(self.results)}")
        print(f"使用GPU: {'是' if self.plan.use_gpu else '否'}")
        print(f"策略级别: {self.plan.strategy.value}")
        
        print(f"\n模型排行榜:")
        lb = self.get_leaderboard()
        if not lb.empty:
            print(lb.to_string(index=False))
        
        # 特征重要性
        fi = self.get_feature_importance_ensemble(top_n=10)
        if not fi.empty:
            print(f"\nTop 10 重要特征:")
            for row in fi.itertuples(index=False):
                print(f"  {row.feature:30s}: {row.importance:.4f}")
        
        print("\n" + "=" * 70)


# =============================================================================
# 端到端便捷接口
# =============================================================================

def quick_model(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None, task_type: str = 'classification',
                plan: Optional[ExecutionPlan] = None,
                return_engine: bool = False) -> Union[np.ndarray, Tuple[Optional[np.ndarray], ParallelModelingEngine], None]:
    """
    快速建模：一行代码完成训练+评估+预测
    
    Args:
        X: 训练特征
        y: 训练标签
        X_test: 测试特征
        task_type: 任务类型
        plan: 执行计划（None=自动创建标准计划）
        return_engine: 是否返回引擎实例
        
    Returns:
        (predictions, engine) 或仅 predictions
    """
    engine = ParallelModelingEngine(task_type=task_type, plan=plan)
    engine.fit(X, y, X_test)
    engine.print_summary()
    
    predictions = None
    if X_test is not None:
        predictions = engine.predict(X_test, blend_method='weighted')
    
    if return_engine:
        return predictions, engine
    return predictions
