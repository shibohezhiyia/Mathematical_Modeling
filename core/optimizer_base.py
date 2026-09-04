"""
超参数优化器抽象基类与通用工具

所有优化策略（贝叶斯、RL、随机、Hyperband、遗传算法）统一继承 BaseOptimizer。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import get_scorer

from core.modeling_engine import ModelLibrary, TaskType
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
                 trial_timeout: Optional[int] = 120,
                 fold_type: str = 'default') -> None:
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        self.use_adaptive_space = use_adaptive_space
        self.adaptive_config = adaptive_config
        self.use_early_stop = use_early_stop
        self.early_stop_config = early_stop_config or SmartEarlyStopConfig(direction='maximize')
        self.trial_timeout = trial_timeout
        self.fold_type = str(fold_type or 'default')
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

    @staticmethod
    def _data_fingerprint(X: pd.DataFrame, y: pd.Series, max_rows: int = 128) -> str:
        """Build a bounded content fingerprint for safe CV-result caching.

        Shape and the first target values are not enough: unrelated datasets can
        otherwise reuse a stale score.  Evenly spaced rows cover the full data
        range while keeping hashing cost independent of dataset size.
        """
        n_rows = len(X)
        if n_rows:
            positions = np.unique(np.linspace(0, n_rows - 1, min(max_rows, n_rows), dtype=np.int64))
            X_sample = X.iloc[positions]
            y_sample = pd.Series(y).iloc[positions]
        else:
            X_sample = X.iloc[:0]
            y_sample = pd.Series(y).iloc[:0]
        digest = hashlib.blake2b(digest_size=16)
        digest.update(repr(tuple(map(str, X.columns))).encode("utf-8", errors="replace"))
        digest.update(repr(tuple(map(str, X.dtypes))).encode("utf-8", errors="replace"))
        try:
            X_hash = pd.util.hash_pandas_object(X_sample, index=True).to_numpy(dtype=np.uint64)
        except (TypeError, ValueError):
            X_hash = pd.util.hash_pandas_object(X_sample.astype(str), index=True).to_numpy(dtype=np.uint64)
        try:
            y_hash = pd.util.hash_pandas_object(y_sample, index=True).to_numpy(dtype=np.uint64)
        except (TypeError, ValueError):
            y_hash = pd.util.hash_pandas_object(y_sample.astype(str), index=True).to_numpy(dtype=np.uint64)
        digest.update(X_hash.tobytes())
        digest.update(y_hash.tobytes())
        return digest.hexdigest()
    
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
        if self.fold_type == 'time':
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            scoring = _METRIC_MAP.get(metric, metric)
            if scoring is None:
                scoring = (
                    'roc_auc_ovr_weighted'
                    if task_type == TaskType.CLASSIFICATION and len(np.unique(y)) > 2
                    else 'roc_auc' if task_type == TaskType.CLASSIFICATION
                    else 'neg_mean_squared_error'
                )
        elif task_type == TaskType.CLASSIFICATION:
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
            # A ``with ThreadPoolExecutor`` block calls shutdown(wait=True), which
            # used to make the timeout illusory: after raising, it still waited for
            # the slow CV job.  On timeout, detach the worker and let the optimizer
            # abort further trials for this model instead of stacking more jobs.
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run_cv)
            try:
                result = future.result(timeout=self.trial_timeout)
            except FutureTimeoutError:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                model_name = model.__class__.__name__
                log_warning(f"[BaseOptimizer] Model evaluation timeout ({self.trial_timeout}s): {model_name}")
                raise TimeoutError(f"Model evaluation timeout ({self.trial_timeout}s): {model_name}")
            except BaseException:
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            executor.shutdown(wait=True)
            return result
        
        # 尝试使用结果缓存。缓存故障可以降级，但模型评估/剪枝异常必须
        # 原样传播；过去两者位于同一个 try 中，评估失败会把整次 CV 重跑。
        cache = None
        cache_key = None
        try:
            from core.result_cache import get_result_cache
            cache = get_result_cache()
            
            # 构建缓存键：模型、参数、数据内容指纹、任务和验证配置。
            model_params = getattr(model, 'get_params', lambda: {})()
            data_fingerprint = self._data_fingerprint(X, y)
            cache_key = cache._make_key(
                'optimizer_cv_v2',
                model.__class__.__name__,
                tuple(sorted(model_params.items())),
                X.shape,
                data_fingerprint,
                task_type.value,
                scoring,
                self.cv_folds,
                self.random_state,
                self.fold_type,
            )
            
            cached = cache.get(cache_key)
            if cached is not None:
                return float(cached)
        except Exception:
            cache = None
            cache_key = None

        result = _run_cv_with_timeout()
        if cache is not None and cache_key is not None:
            try:
                cache.set(cache_key, result)
            except Exception:
                pass
        return result
    
    def _evaluate_model_fold_by_fold(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                       cv, scoring: str, step_callback: callable) -> float:
        """逐 fold 评估模型，每完成一折调用 step_callback 报告分数（供 Pruner 使用）
        
        对 LGB/XGB/CatBoost 启用 early stopping，大幅缩短超参搜索时间。
        """
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
