"""
贝叶斯超参数优化器

基于 Optuna，支持：
- TPE (Tree-structured Parzen Estimator) - 默认
- CMA-ES (Covariance Matrix Adaptation)
- 随机搜索
- 早停剪枝 (Pruning)

继承 BaseOptimizer 统一接口。
无 Optuna 时自动回退到随机搜索。
"""

import time
from typing import Dict, Optional, Any, Union
from copy import deepcopy

import numpy as np
import pandas as pd

from core.modeling_engine import ModelLibrary, TaskType
from core.optimizer_base import BaseOptimizer, OptimizationResult
from core.workspace_manager import get_workspace_manager
from core.progress_bar import progress_range
from utils.helpers import log_info, log_warning

# 尝试导入 Optuna
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    log_warning("[BayesianOptimizer] Optuna 未安装，将使用随机搜索回退")


class SamplerType:
    """采样器类型"""
    TPE = "tpe"
    CMAES = "cmaes"
    RANDOM = "random"


class BayesianOptimizer(BaseOptimizer):
    """
    贝叶斯优化器（TPE / CMA-ES / Random）
    
    继承 BaseOptimizer，统一接口。
    """
    
    def __init__(self,
                 n_trials: int = 30,
                 sampler: str = SamplerType.TPE,
                 pruner: str = 'median',
                 cv_folds: int = 3,
                 timeout: Optional[int] = 300,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 direction: str = 'maximize',
                 verbose: bool = True,
                 trial_timeout: Optional[int] = 120) -> None:
        super().__init__(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state, verbose=verbose, trial_timeout=trial_timeout)
        self.sampler_type = sampler
        self.pruner_type = pruner
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.direction = direction
        
        self._optuna_available = OPTUNA_AVAILABLE
        if not self._optuna_available and sampler != SamplerType.RANDOM:
            log_warning(f"[BayesianOptimizer] Optuna未安装，采样器强制设为random")
            self.sampler_type = SamplerType.RANDOM
    
    def optimize(self,
                 model_key: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: Union[str, TaskType],
                 metric: Optional[str] = None,
                 custom_search_space: Optional[Dict] = None) -> OptimizationResult:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        search_space = self._get_search_space(model_key, task_type, custom_search_space)
        models = ModelLibrary.get_models(task_type, [model_key])
        spec = models[model_key]
        
        if not search_space:
            log_info(f"[BayesianOptimizer] {spec.name} 无搜索空间，跳过优化")
            return OptimizationResult(
                model_key=model_key,
                best_params=deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                sampler_type='none'
            )
        
        start_time = time.time()
        
        # 大数据保护：sklearn 原生 GBDT 在 >30k 样本下超参搜索极慢且无法被超时中断，直接跳过
        n_samples = len(X)
        if model_key == 'gbdt' and n_samples > 30000:
            log_info(f"[BayesianOptimizer] {model_key} 大数据({n_samples}样本)跳过超参优化，使用默认参数")
            from core.optimizer_base import OptimizationResult
            return OptimizationResult(
                model_key=model_key,
                best_params=deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                optimize_time=0.0,
                sampler_type='none'
            )
        # HistGradientBoosting 大数据下也限制 trials 数（虽然比原生 GBDT 快得多）
        if model_key == 'hist_gb' and n_samples > 50000:
            self.n_trials = min(self.n_trials, 8)
            log_info(f"[BayesianOptimizer] {model_key} 大数据({n_samples}样本)限制 trials 到 {self.n_trials}")
        
        # Auto-tune n_trials based on data size (lightweight heuristic)
        data_size = n_samples * len(X.columns)
        original_n_trials = self.n_trials
        if data_size < 1000 and self.n_trials > 20:
            self.n_trials = 20
            log_info(f"[BayesianOptimizer] Small dataset ({data_size} cells), auto-limit trials to {self.n_trials}")
        elif data_size > 100000 and self.n_trials > 15:
            self.n_trials = max(15, self.n_trials // 2)
            log_info(f"[BayesianOptimizer] Large dataset ({data_size} cells), auto-limit trials to {self.n_trials} for speed")
        
        if self._optuna_available and self.sampler_type != SamplerType.RANDOM:
            result = self._optuna_optimize(
                model_key, spec, search_space, X, y, task_type, metric
            )
        else:
            result = self._random_search_optimize(
                model_key, spec, search_space, X, y, task_type, metric
            )
        
        self.n_trials = original_n_trials
        
        result.optimize_time = time.time() - start_time
        log_info(f"[BayesianOptimizer] {spec.name} 优化完成: "
                 f"best_score={result.best_score:.4f}, trials={result.n_trials}, "
                 f"time={result.optimize_time:.1f}s")
        
        return result
    
    def _optuna_optimize(self, model_key: str, spec, search_space,
                         X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                         metric: Optional[str]) -> OptimizationResult:
        """使用 Optuna 进行贝叶斯优化（支持 Pruner + 两阶段搜索）"""
        from core.search_space import SearchSpace
        
        if self.sampler_type == SamplerType.TPE:
            sampler = TPESampler(seed=self.random_state, multivariate=True)
        elif self.sampler_type == SamplerType.CMAES:
            sampler = CmaEsSampler(seed=self.random_state)
        else:
            sampler = RandomSampler(seed=self.random_state)
        
        pruner = None
        if self.pruner_type == 'median':
            # 更早开始剪枝：2 个 trial 后就开始评估，第 1 折就报告
            pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=0)
        elif self.pruner_type == 'hyperband':
            pruner = HyperbandPruner(min_resource=1, reduction_factor=3)
        
        # Early stopping: stop if no improvement for N consecutive complete trials
        early_stop_patience = 10
        best_value_history = []
        
        def early_stop_callback(study, trial):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            current_best = study.best_value if study.direction == optuna.study.StudyDirection.MAXIMIZE else study.best_value
            best_value_history.append(current_best)
            if len(best_value_history) > early_stop_patience:
                recent = best_value_history[-early_stop_patience:]
                if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                    if max(recent) <= best_value_history[-early_stop_patience - 1]:
                        study.stop()
                else:
                    if min(recent) >= best_value_history[-early_stop_patience - 1]:
                        study.stop()
        
        # ========== 阶段 1: 粗筛（快速排除差配置）==========
        n_coarse = max(3, self.n_trials // 3)
        n_fine = self.n_trials - n_coarse
        
        log_info(f"[BayesianOptimizer] {model_key} two-stage search: coarse={n_coarse}, fine={n_fine}")
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # 粗筛阶段：少 fold（2折），DL 模型限制 epoch=10
        coarse_cv_folds = min(2, self.cv_folds)
        original_cv_folds = self.cv_folds
        
        def objective(trial, fixed_params: Optional[Dict] = None, cv_folds: int = None, dl_max_epochs: int = 20):
            if fixed_params is not None:
                # 精筛阶段：固定使用给定参数
                params = deepcopy(fixed_params)
            else:
                params = search_space.to_optuna(trial) if isinstance(search_space, SearchSpace) else self._suggest_params(trial, search_space)
            params.update(deepcopy(spec.default_params))
            
            # 限制危险超参组合
            if self._is_dangerous_config(params, model_key):
                return float('-inf') if self.direction == 'maximize' else float('inf')
            
            try:
                model = ModelLibrary.create_model(model_key, task_type, **params)
                
                # 临时修改 cv_folds
                if cv_folds is not None:
                    self.cv_folds = cv_folds
                
                # 逐 fold 报告，让 Pruner 真正工作
                def step_callback(interim_score, step):
                    if pruner and hasattr(trial, 'report'):
                        trial.report(interim_score, step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                
                score = self._evaluate_model(model, X, y, task_type, metric, step_callback=step_callback)
                
                return score
            except optuna.TrialPruned:
                raise
            except TimeoutError:
                return float('-inf') if self.direction == 'maximize' else float('inf')
            except Exception:
                return float('-inf') if self.direction == 'maximize' else float('inf')
            finally:
                self.cv_folds = original_cv_folds
        
        # 粗筛
        if n_coarse > 0:
            log_info(f"[BayesianOptimizer] {model_key} coarse screening ({coarse_cv_folds}-fold)")
            # TPE 采样器不支持并行 trial，固定 n_jobs=1，并行留给 cross_val_score
            study.optimize(
                lambda trial: objective(trial, cv_folds=coarse_cv_folds, dl_max_epochs=10),
                n_trials=n_coarse,
                timeout=self.timeout // 2 if self.timeout else None,
                n_jobs=1,
                show_progress_bar=False,
                callbacks=[early_stop_callback]
            )
        
        # 阶段 2: 精筛（只对 top 配置用完整资源验证）
        if n_fine > 0 and len(study.trials) > 0:
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                top_trials = sorted(completed, key=lambda t: t.value, reverse=(self.direction == 'maximize'))[:5]
                top_params = [t.params for t in top_trials]
                log_info(f"[BayesianOptimizer] {model_key} fine tuning top-{len(top_params)} configs ({original_cv_folds}-fold)")
                
                # 精筛：用 enqueue_trial 固定插入 top 配置，让 Optuna 在附近搜索
                for tp in top_params[:n_fine]:
                    study.enqueue_trial(tp)
                
                study.optimize(
                    lambda trial: objective(trial, fixed_params=None, cv_folds=original_cv_folds, dl_max_epochs=15),
                    n_trials=n_fine,
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[early_stop_callback]
                )
        
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                })
        
        best_params = deepcopy(spec.default_params)
        best_score = 0.0
        if len(history) > 0:
            best_params.update(study.best_params)
            best_score = study.best_value
        
        return OptimizationResult(
            model_key=model_key,
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=len(history),
            sampler_type=self.sampler_type
        )
    
    def _is_dangerous_config(self, params: Dict[str, Any], model_key: str) -> bool:
        """检测危险超参组合（会导致训练极慢或崩溃）"""
        # 深度学习模型：避免过大网络
        if model_key in ('torch_mlp', 'torch_cnn1d', 'torch_lstm'):
            hidden = params.get('hidden_dims')
            if hidden:
                total_neurons = sum(hidden) if isinstance(hidden, (list, tuple)) else hidden
                if total_neurons > 512:
                    return True
        # GBDT：n_estimators * max_depth 乘积过大时训练极慢（大数据下）
        if model_key in ('gbdt', 'gradient_boosting'):
            n_est = params.get('n_estimators', 200)
            max_d = params.get('max_depth', 3)
            if n_est * max_d > 1000:  # 如 200*5=1000，300*4=1200
                return True
        return False
    
    def _random_search_optimize(self, model_key: str, spec, search_space,
                                 X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                                 metric: Optional[str]) -> OptimizationResult:
        """随机搜索回退"""
        from core.search_space import SearchSpace
        rng = np.random.RandomState(self.random_state)
        history = []
        best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        best_params = {}
        
        for trial in progress_range(self.n_trials, desc=f"优化 {model_key}", disable=not self.verbose):
            params = search_space.sample(rng=rng) if isinstance(search_space, SearchSpace) else {}
            if not params and isinstance(search_space, dict):
                for key, values in search_space.items():
                    params[key] = rng.choice(values)
            
            try:
                full_params = deepcopy(spec.default_params)
                full_params.update(params)
                model = ModelLibrary.create_model(model_key, task_type, **full_params)
                score = self._evaluate_model(model, X, y, task_type, metric)
                
                history.append({
                    'number': trial,
                    'params': params,
                    'value': score
                })
                
                is_better = (score > best_score) if self.direction == 'maximize' else (score < best_score)
                if is_better:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                continue
        
        final_params = deepcopy(spec.default_params)
        final_params.update(best_params)
        
        return OptimizationResult(
            model_key=model_key,
            best_params=final_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=len(history),
            sampler_type='random_fallback'
        )
    
    def _suggest_params(self, trial, search_space) -> Dict:
        """为 Optuna trial 建议参数（向后兼容列表格式）"""
        params = {}
        for key, values in search_space.items():
            if all(isinstance(v, bool) for v in values):
                params[key] = trial.suggest_categorical(key, values)
            elif all(isinstance(v, int) for v in values):
                params[key] = trial.suggest_categorical(key, values)
            elif all(isinstance(v, float) for v in values):
                params[key] = trial.suggest_categorical(key, values)
            else:
                params[key] = trial.suggest_categorical(key, values)
        return params
    
    def plot_optimization_history(self, result: OptimizationResult, save_path: Optional[str] = None):
        """绘制优化历史"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if not result.optimization_history:
                return
            
            values = [h['value'] for h in result.optimization_history if h['value'] is not None]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(values)), values, 'b-', alpha=0.5, label='Trial score')
            
            if self.direction == 'maximize':
                best_so_far = np.maximum.accumulate(values)
            else:
                best_so_far = np.minimum.accumulate(values)
            ax.plot(range(len(values)), best_so_far, 'r-', linewidth=2, label='Best so far')
            
            ax.set_xlabel('Trial')
            ax.set_ylabel('Score')
            ax.set_title(f'Optimization History: {result.model_key}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                wm = get_workspace_manager()
                safe_path = wm.safe_path(save_path, 'reports')
                fig.savefig(safe_path, dpi=150, bbox_inches='tight')
                log_info(f"优化历史图已保存: {safe_path}")
            
            plt.close(fig)
        except ImportError:
            log_warning("matplotlib 未安装，跳过绘图")


# 向后兼容别名
HyperparameterOptimizer = BayesianOptimizer


def quick_optimize(model_key: str, X, y, task_type='classification',
                   n_trials: int = 20, **kwargs) -> OptimizationResult:
    """快速超参数优化"""
    optimizer = BayesianOptimizer(n_trials=n_trials, **kwargs)
    return optimizer.optimize(model_key, X, y, task_type)
