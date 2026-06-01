"""
超参数优化器抽象基类与通用工具

所有优化策略（贝叶斯、RL、随机、Hyperband、遗传算法）统一继承 BaseOptimizer。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from core.modeling_engine import ModelLibrary, TaskType, TaskTypeDetector
from core.search_space import SearchSpace
from core.smart_early_stopper import SmartEarlyStopper, SmartEarlyStopConfig
from core.adaptive_search_space import AdaptiveSearchSpace, AdaptationConfig
from utils.helpers import log_info, log_warning


@dataclass
class OptimizationResult:
    """优化结果（所有优化器统一返回此类型）"""
    model_key: str
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0
    optimize_time: float = 0.0
    sampler_type: str = ""


class BaseOptimizer(ABC):
    """
    超参数优化器抽象基类
    
    子类必须实现 optimize() 方法。
    optimize_all() 提供默认的串行多模型优化。
    """
    
    def __init__(self,
                 n_trials: int = 30,
                 cv_folds: int = 3,
                 random_state: int = 42,
                 verbose: bool = True,
                 use_adaptive_space: bool = False,
                 adaptive_config: Optional[AdaptationConfig] = None,
                 use_early_stop: bool = False,
                 early_stop_config: Optional[SmartEarlyStopConfig] = None,
                 trial_timeout: Optional[int] = 120) -> None:
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self.use_adaptive_space = use_adaptive_space
        self.adaptive_config = adaptive_config
        self.use_early_stop = use_early_stop
        self.early_stop_config = early_stop_config or SmartEarlyStopConfig(direction='maximize')
        self.trial_timeout = trial_timeout
        self._early_stopper: Optional[SmartEarlyStopper] = None
        if self.use_early_stop:
            self._early_stopper = SmartEarlyStopper(self.early_stop_config)
    
    @abstractmethod
    def optimize(self,
                 model_key: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: Union[str, TaskType],
                 metric: Optional[str] = None,
                 custom_search_space: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        为单个模型执行超参数优化
        
        Args:
            model_key: 模型标识
            X: 特征
            y: 标签
            task_type: 任务类型
            metric: 评估指标
            custom_search_space: 自定义搜索空间
            
        Returns:
            OptimizationResult
        """
        ...
    
    def optimize_all(self,
                     model_keys: List[str],
                     X: pd.DataFrame,
                     y: pd.Series,
                     task_type: Union[str, TaskType],
                     metric: Optional[str] = None) -> Dict[str, OptimizationResult]:
        """
        为多个模型串行执行超参数优化
        
        Returns:
            Dict[model_key, OptimizationResult]
        """
        results = {}
        for key in model_keys:
            try:
                result = self.optimize(key, X, y, task_type, metric)
                results[key] = result
            except Exception as e:
                log_warning(f"[{self.__class__.__name__}] {key} 优化失败: {e}")
        return results
    
    def _is_deep_learning_model(self, model: Any) -> bool:
        """检测是否为深度学习模型（需要特殊处理）"""
        name = model.__class__.__name__
        return name in ('TorchMLP', 'TorchCNN1D', 'TorchLSTM', 'TorchGRU', 'TorchNAS', 'TorchResMLP', 'TabNetWrapper')
    
    def _limit_dl_epochs(self, model: Any, max_epochs: int = 15) -> Any:
        """限制深度学习模型的 epochs（超参搜索专用）"""
        if self._is_deep_learning_model(model) and hasattr(model, 'epochs'):
            original_epochs = model.epochs
            if original_epochs > max_epochs:
                model.epochs = max_epochs
                log_info(f"[BaseOptimizer] DL model epoch reduced: {original_epochs} -> {max_epochs} for hyperopt")
        return model
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                        task_type: TaskType, metric: Optional[str] = None,
                        step_callback: Optional[callable] = None,
                        n_jobs: Optional[int] = None) -> float:
        """
        公共模型评估方法：K折交叉验证（带结果缓存 + 单 trial 超时）
        
        使用模型参数+数据哈希作为缓存键，避免重复评估相同配置。
        通过 ThreadPoolExecutor 给 cross_val_score 加超时保护，防止 SVM 等模型在
        某些参数组合下无限卡住。
        """
        # 自定义 metric 名称映射为 sklearn scorer
        _METRIC_MAP = {
            'rmse': 'neg_root_mean_squared_error',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'mape': 'neg_mean_absolute_percentage_error',
        }
        if task_type == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = _METRIC_MAP.get(metric, metric)
            if scoring is None:
                scoring = 'roc_auc_ovr_weighted' if len(np.unique(y)) > 2 else 'roc_auc'
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = _METRIC_MAP.get(metric, metric) or 'neg_mean_squared_error'
        
        # 深度学习模型：超参搜索时自动限制 epochs
        model = self._limit_dl_epochs(model, max_epochs=20)
        
        # 并行 CV（默认使用优化器配置的 n_jobs，传统ML模型受益明显）
        cv_n_jobs = n_jobs if n_jobs is not None else getattr(self, 'n_jobs', 1)
        # 深度学习模型避免并行CV（内存爆炸）
        if self._is_deep_learning_model(model):
            cv_n_jobs = 1
        
        def _run_cv():
            if step_callback:
                # 逐 fold 评估，支持中间步骤报告（供 Pruner 使用）
                return self._evaluate_model_fold_by_fold(model, X, y, cv, scoring, step_callback)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cv_n_jobs)
            return float(scores.mean())
        
        def _run_cv_with_timeout():
            if not self.trial_timeout or self.trial_timeout <= 0:
                return _run_cv()
            # max_workers=2 避免单个慢 trial 阻塞整个超参搜索流水线
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(_run_cv)
                try:
                    return future.result(timeout=self.trial_timeout)
                except FutureTimeoutError:
                    model_name = model.__class__.__name__
                    log_warning(f"[BaseOptimizer] Model evaluation timeout ({self.trial_timeout}s): {model_name}")
                    raise TimeoutError(f"Model evaluation timeout ({self.trial_timeout}s): {model_name}")
        
        # 尝试使用结果缓存
        try:
            from core.result_cache import get_result_cache
            cache = get_result_cache()
            
            # 构建缓存键：模型类名 + 参数 + 数据形状 + 任务类型 + metric
            model_params = getattr(model, 'get_params', lambda: {})()
            cache_key = cache._make_key(
                model.__class__.__name__,
                sorted(model_params.items()),
                X.shape,
                list(y.values[:100]),  # 取前100个标签值作为数据指纹
                task_type.value,
                scoring,
                self.cv_folds,
                self.random_state
            )
            
            cached = cache.get(cache_key)
            if cached is not None:
                return float(cached)
            
            result = _run_cv_with_timeout()
            cache.set(cache_key, result)
            return result
        except TimeoutError:
            raise
        except Exception:
            # 缓存失败时不影响正常评估
            return _run_cv_with_timeout()
    
    def _evaluate_model_fold_by_fold(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                       cv, scoring: str, step_callback: callable) -> float:
        """逐 fold 评估模型，每完成一折调用 step_callback 报告分数（供 Pruner 使用）
        
        对 LGB/XGB/CatBoost 启用 early stopping，大幅缩短超参搜索时间。
        """
        from sklearn.base import clone
        from sklearn.metrics import get_scorer
        
        scorer = get_scorer(scoring)
        scores = []
        model_name = model.__class__.__name__
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            model_fold = clone(model)
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 对支持 early stopping 的树模型注入验证集
            if model_name in ('LGBMClassifier', 'LGBMRegressor'):
                try:
                    import lightgbm as lgb
                    model_fold.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                except Exception:
                    model_fold.fit(X_tr, y_tr)
            elif model_name in ('XGBClassifier', 'XGBRegressor'):
                try:
                    model_fold.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except Exception:
                    model_fold.fit(X_tr, y_tr)
            elif model_name in ('CatBoostClassifier', 'CatBoostRegressor'):
                try:
                    model_fold.fit(
                        X_tr, y_tr,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except Exception:
                    model_fold.fit(X_tr, y_tr)
            else:
                model_fold.fit(X_tr, y_tr)
            
            score = scorer(model_fold, X_val, y_val)
            scores.append(float(score))
            
            # 报告中间结果
            if step_callback:
                step_callback(float(np.mean(scores)), fold_idx + 1)
        
        return float(np.mean(scores))
    
    def _get_search_space(self, model_key: str, task_type: TaskType,
                          custom_search_space: Optional[Dict] = None) -> SearchSpace:
        """获取模型的搜索空间（返回 SearchSpace 对象，支持自适应）"""
        models = ModelLibrary.get_models(task_type, [model_key])
        if not models:
            raise ValueError(f"未知模型: {model_key}")
        spec = models[model_key]
        config = custom_search_space or spec.hyperparam_space
        if self.use_adaptive_space:
            return AdaptiveSearchSpace(config, self.adaptive_config)
        return SearchSpace(config)
    
    def _check_trial_early_stop(self, score: float) -> bool:
        """检查 trial 级早停"""
        if not self.use_early_stop or self._early_stopper is None:
            return False
        should_stop, reason = self._early_stopper.trial_check(score)
        if should_stop and self.verbose:
            log_info(f"[EarlyStop] Trial 早停触发: {reason.value}")
        return should_stop
    
    def _report_trial_score(self, score: float) -> None:
        """报告 trial 分数（用于早停统计）"""
        if self.use_early_stop and self._early_stopper is not None:
            self._early_stopper.trial_report(score)
    
    def _adapt_space(self, search_space: SearchSpace, score: float) -> None:
        """自适应调整搜索空间"""
        if not self.use_adaptive_space or not isinstance(search_space, AdaptiveSearchSpace):
            return
        search_space.update_history(search_space._last_params or {}, score)
        if search_space.should_adapt():
            report = search_space.adapt(direction=self.early_stop_config.direction)
            if self.verbose and report['adapted_params']:
                log_info(f"[AdaptiveSpace] 调整参数: {report['adapted_params']}")
    
    def get_early_stop_stats(self) -> Optional[Dict[str, Any]]:
        """获取早停统计"""
        if self._early_stopper is not None:
            return self._early_stopper.get_stats()
        return None
