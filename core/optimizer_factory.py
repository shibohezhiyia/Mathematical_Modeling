"""
优化器工厂与附加优化策略

统一入口：OptimizerFactory.create(strategy, **kwargs) -> BaseOptimizer
支持策略：bayesian, tpe, cmaes, rl, random, hyperband, genetic
"""

import random
import time
import copy
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.modeling_engine import ModelLibrary, TaskType
from core.optimizer_base import BaseOptimizer, OptimizationResult
from utils.helpers import log_info
from core.search_space import SearchSpace
from core.adaptive_search_space import AdaptiveSearchSpace
from core.progress_bar import progress_range


# =============================================================================
# RandomSearchOptimizer
# =============================================================================

class RandomSearchOptimizer(BaseOptimizer):
    """纯随机搜索优化器 — 简单基线，也作为其他优化器失败时的 fallback"""
    
    def __init__(self, n_trials: int = 30, cv_folds: int = 3, random_state: int = 42, **kwargs: Any) -> None:
        super().__init__(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state,
                         verbose=kwargs.get('verbose', True),
                         trial_timeout=kwargs.get('trial_timeout', 120))
        self.rng = np.random.RandomState(random_state)
    
    def optimize(self,
                 model_key: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: Union[str, TaskType],
                 metric: Optional[str] = None,
                 custom_search_space: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        search_space = self._get_search_space(model_key, task_type, custom_search_space)
        models = ModelLibrary.get_models(task_type, [model_key])
        spec = models[model_key]
        
        if not search_space:
            return OptimizationResult(
                model_key=model_key,
                best_params=copy.deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                sampler_type='random'
            )
        
        start_time = time.time()
        history = []
        best_score = float('-inf')
        best_params = {}
        
        # 自适应搜索空间支持
        is_adaptive = isinstance(search_space, AdaptiveSearchSpace)
        
        for trial in progress_range(self.n_trials, desc=f"随机搜索 {model_key}", disable=not self.verbose):
            params = search_space.sample(rng=self.rng)
            
            try:
                # 优化：使用 dict.copy() 替代 deepcopy，因为 default_params 只包含简单类型
                full_params = spec.default_params.copy()
                full_params.update(params)
                model = ModelLibrary.create_model(model_key, task_type, **full_params)
                score = self._evaluate_model(model, X, y, task_type, metric)
                
                # 报告 trial 分数（用于早停）
                self._report_trial_score(score)
                
                history.append({'trial': trial + 1, 'params': params, 'score': score})
                if score > best_score:
                    best_score = score
                    best_params = copy.deepcopy(params)
                
                # 自适应搜索空间更新
                if is_adaptive:
                    search_space.update_history(params, score)
                    if search_space.should_adapt():
                        report = search_space.adapt(direction='maximize')
                        if self.verbose and report.get('adapted_params'):
                            log_info(f"[AdaptiveSpace] {model_key} 调整: {report['adapted_params']}")
                
                # Trial 级早停检查
                if self._check_trial_early_stop(score):
                    break
                    
            except Exception:
                continue
        
        final_params = copy.deepcopy(spec.default_params)
        final_params.update(best_params)
        
        return OptimizationResult(
            model_key=model_key,
            best_params=final_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=len(history),
            optimize_time=time.time() - start_time,
            sampler_type='random'
        )


# =============================================================================
# HyperbandOptimizer
# =============================================================================

class HyperbandOptimizer(BaseOptimizer):
    """
    Hyperband 多保真度优化器
    
    核心思想：用少量资源（如少折CV、少迭代）快速筛除差配置，
    将更多资源留给有潜力的配置。
    
    对树模型通过减少 n_estimators/max_depth 作为保真度维度；
    对神经网络通过减少 epochs 作为保真度维度。
    """
    
    def __init__(self, n_trials: int = 81, cv_folds: int = 3, random_state: int = 42,
                 eta: int = 3, max_resource: int = 27, **kwargs: Any) -> None:
        super().__init__(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state,
                         verbose=kwargs.get('verbose', True),
                         trial_timeout=kwargs.get('trial_timeout', 120))
        self.eta = eta
        self.max_resource = max_resource
        self.rng = np.random.RandomState(random_state)
    
    def optimize(self,
                 model_key: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: Union[str, TaskType],
                 metric: Optional[str] = None,
                 custom_search_space: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        search_space = self._get_search_space(model_key, task_type, custom_search_space)
        models = ModelLibrary.get_models(task_type, [model_key])
        spec = models[model_key]
        
        if not search_space:
            return OptimizationResult(
                model_key=model_key,
                best_params=copy.deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                sampler_type='hyperband'
            )
        
        start_time = time.time()
        history = []
        best_score = float('-inf')
        best_params = {}
        
        # Hyperband 算法
        max_iter = int(np.log(self.max_resource) / np.log(self.eta))
        
        for s in reversed(range(max_iter + 1)):
            n = int(np.ceil((max_iter + 1) / (s + 1)) * self.eta ** s)
            r = self.max_resource * self.eta ** (-s)
            
            # SuccessiveHalving 内层循环
            candidates = []
            for i in range(n):
                params = search_space.sample(rng=self.rng)
                candidates.append(params)
            
            for i in range(s + 1):
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)
                
                scores = []
                for params in candidates[:n_i]:
                    try:
                        # 优化：使用 dict.copy() 替代 deepcopy，因为 default_params 只包含简单类型
                        full_params = spec.default_params.copy()
                        full_params.update(params)
                        # 通过资源维度控制保真度
                        full_params = self._apply_resource(full_params, r_i, model_key)
                        model = ModelLibrary.create_model(model_key, task_type, **full_params)
                        score = self._evaluate_model(model, X, y, task_type, metric)
                        scores.append((score, params))
                        history.append({'trial': len(history) + 1, 'params': params, 'score': score, 'resource': r_i})
                    except Exception:
                        scores.append((float('-inf'), params))
                
                scores.sort(key=lambda x: x[0], reverse=True)
                candidates = [p for _, p in scores]
                
                if scores[0][0] > best_score:
                    best_score = scores[0][0]
                    best_params = copy.deepcopy(scores[0][1])
        
        # 优化：使用 dict.copy() 替代 deepcopy
        final_params = spec.default_params.copy()
        final_params.update(best_params)
        
        return OptimizationResult(
            model_key=model_key,
            best_params=final_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=len(history),
            optimize_time=time.time() - start_time,
            sampler_type='hyperband'
        )
    
    def _apply_resource(self, params: Dict[str, Any], resource: int, model_key: str) -> Dict[str, Any]:
        """将资源维度应用到参数中
        
        优化：直接修改输入字典，避免 deepcopy，调用方应使用 dict.copy() 传递副本。
        """
        # 树模型：减少迭代次数
        if 'n_estimators' in params:
            params['n_estimators'] = max(10, min(resource * 10, params['n_estimators']))
        if 'max_iter' in params:
            params['max_iter'] = max(10, min(resource * 10, params['max_iter']))
        if 'iterations' in params:
            params['iterations'] = max(10, min(resource * 10, params['iterations']))
        # 神经网络：减少 epochs
        if 'epochs' in params:
            params['epochs'] = max(5, min(resource, params['epochs']))
        if 'max_epochs' in params:
            params['max_epochs'] = max(5, min(resource, params['max_epochs']))
        return params


# =============================================================================
# GeneticAlgorithmOptimizer
# =============================================================================

class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    遗传算法优化器
    
    染色体编码：参数向量（混合离散/连续）
    选择：锦标赛选择
    交叉：单点交叉
    变异：高斯变异（连续）/ 随机替换（离散）
    """
    
    def __init__(self, n_trials: int = 50, cv_folds: int = 3, random_state: int = 42,
                 population_size: int = 20, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2, elitism: int = 2, **kwargs: Any) -> None:
        super().__init__(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state,
                         verbose=kwargs.get('verbose', True),
                         trial_timeout=kwargs.get('trial_timeout', 120))
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.rng = np.random.RandomState(random_state)
    
    def optimize(self,
                 model_key: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: Union[str, TaskType],
                 metric: Optional[str] = None,
                 custom_search_space: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        search_space = self._get_search_space(model_key, task_type, custom_search_space)
        models = ModelLibrary.get_models(task_type, [model_key])
        spec = models[model_key]
        
        if not search_space:
            return OptimizationResult(
                model_key=model_key,
                best_params=copy.deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                sampler_type='genetic'
            )
        
        start_time = time.time()
        history = []
        best_score = float('-inf')
        best_params = {}
        
        param_names = list(search_space.keys())
        
        # 初始化种群
        population = [self._random_chromosome(search_space) for _ in range(self.population_size)]
        
        # 为遗传算法构建离散候选（从 SearchSpace 中提取）
        param_candidates = search_space.build_candidates(n=16) if hasattr(search_space, 'build_candidates') else search_space
        
        generations = self.n_trials // self.population_size
        
        for gen in progress_range(generations, desc=f"遗传算法 {model_key}", disable=not self.verbose):
            # 评估种群
            fitness = []
            for chrom in population:
                params = self._chromosome_to_params(chrom, search_space)
                try:
                    # 优化：使用 dict.copy() 替代 deepcopy，因为 default_params 只包含简单类型
                    full_params = spec.default_params.copy()
                    full_params.update(params)
                    model = ModelLibrary.create_model(model_key, task_type, **full_params)
                    score = self._evaluate_model(model, X, y, task_type, metric)
                    fitness.append(score)
                    history.append({'trial': len(history) + 1, 'params': params, 'score': score, 'generation': gen})
                    if score > best_score:
                        best_score = score
                        best_params = copy.deepcopy(params)
                except Exception:
                    fitness.append(float('-inf'))
            
            # 精英保留
            sorted_idx = np.argsort(fitness)[::-1]
            new_population = [population[i] for i in sorted_idx[:self.elitism]]
            
            # 生成新一代
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)
                
                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    # 优化：使用 list.copy() 替代 deepcopy，染色体是简单值列表
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutate(child1, search_space)
                child2 = self._mutate(child2, search_space)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # 优化：使用 dict.copy() 替代 deepcopy
        final_params = spec.default_params.copy()
        final_params.update(best_params)
        
        return OptimizationResult(
            model_key=model_key,
            best_params=final_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=len(history),
            optimize_time=time.time() - start_time,
            sampler_type='genetic'
        )
    
    def _random_chromosome(self, search_space: SearchSpace) -> List[float]:
        """将搜索空间编码为一维染色体"""
        if hasattr(search_space, 'build_candidates'):
            cand = search_space.build_candidates(n=16)
            chrom = []
            for key in search_space.keys():
                values = cand.get(key, [])
                if values:
                    idx = self.rng.randint(0, len(values))
                else:
                    idx = 0
                chrom.append(idx)
            return chrom
        else:
            chrom = []
            for key, values in search_space.items():
                idx = self.rng.randint(0, len(values))
                chrom.append(idx)
            return chrom
    
    def _chromosome_to_params(self, chrom: List[float], search_space: SearchSpace) -> Dict[str, Any]:
        """将染色体解码为参数"""
        if hasattr(search_space, 'build_candidates'):
            cand = search_space.build_candidates(n=16)
            params = {}
            for i, key in enumerate(search_space.keys()):
                values = cand.get(key, [])
                if values:
                    idx = int(chrom[i]) % len(values)
                    params[key] = values[idx]
            return params
        else:
            params = {}
            for i, (key, values) in enumerate(search_space.items()):
                idx = int(chrom[i]) % len(values)
                params[key] = values[idx]
            return params
    
    def _tournament_select(self, population: List[List[float]], fitness: List[float], k: int = 3) -> List[float]:
        """锦标赛选择"""
        selected = self.rng.choice(len(population), k, replace=False)
        best = selected[0]
        for idx in selected[1:]:
            if fitness[idx] > fitness[best]:
                best = idx
        return population[best]
    
    def _crossover(self, p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        """单点交叉 - 列表切片，O(k) 其中 k=point"""
        point = self.rng.randint(1, len(p1))
        # 切片创建新列表，避免逐个元素复制
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    
    def _mutate(self, chrom: List[float], search_space: SearchSpace) -> List[float]:
        """变异：随机替换某个基因
        
        优化：使用 list.copy() 替代 deepcopy，染色体是简单值列表。
        """
        chrom = chrom.copy()
        if hasattr(search_space, 'build_candidates'):
            cand = search_space.build_candidates(n=16)
            gene_keys = list(search_space.keys())
            for i, key in enumerate(gene_keys):
                values = cand.get(key, [])
                if values and self.rng.random() < self.mutation_rate:
                    chrom[i] = self.rng.randint(0, len(values))
        else:
            for i, (key, values) in enumerate(search_space.items()):
                if self.rng.random() < self.mutation_rate:
                    chrom[i] = self.rng.randint(0, len(values))
        return chrom


# =============================================================================
# OptimizerFactory
# =============================================================================

class OptimizerFactory:
    """
    优化器工厂 — 统一创建各种超参数优化策略
    
    使用方式：
        optimizer = OptimizerFactory.create('bayesian', n_trials=50)
        optimizer = OptimizerFactory.create('rl', n_trials=30)
        optimizer = OptimizerFactory.create('hyperband')
    """
    
    STRATEGIES = {
        'bayesian': 'core.hyperparameter_optimizer.BayesianOptimizer',
        'tpe': 'core.hyperparameter_optimizer.BayesianOptimizer',
        'cmaes': 'core.hyperparameter_optimizer.BayesianOptimizer',
        'rl': 'core.reinforcement_learning.RLOptimizer',
        'random': 'core.optimizer_factory.RandomSearchOptimizer',
        'hyperband': 'core.optimizer_factory.HyperbandOptimizer',
        'genetic': 'core.optimizer_factory.GeneticAlgorithmOptimizer',
        'both': 'core.hyperparameter_optimizer.BayesianOptimizer',  # 回退到贝叶斯
    }
    
    @classmethod
    def create(cls, strategy: str, **kwargs) -> BaseOptimizer:
        """
        创建优化器实例
        
        Args:
            strategy: 策略名称 ('bayesian', 'tpe', 'cmaes', 'rl', 'random', 'hyperband', 'genetic')
            **kwargs: 传递给优化器构造函数的参数
            
        Returns:
            BaseOptimizer 实例
        """
        strategy = strategy.lower().strip()
        
        if strategy not in cls.STRATEGIES:
            available = ', '.join(sorted(set(cls.STRATEGIES.keys())))
            raise ValueError(f"未知优化策略: '{strategy}'。可用: {available}")
        
        module_path, class_name = cls.STRATEGIES[strategy].rsplit('.', 1)
        
        # 特殊处理 cmaes：设置 sampler 参数
        if strategy == 'cmaes':
            kwargs['sampler'] = 'cmaes'
        elif strategy == 'tpe':
            kwargs['sampler'] = 'tpe'
        elif strategy == 'bayesian' and 'sampler' not in kwargs:
            kwargs['sampler'] = 'tpe'
        
        # 动态导入
        import importlib
        module = importlib.import_module(module_path)
        optimizer_class = getattr(module, class_name)
        
        return optimizer_class(**kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有可用策略"""
        return sorted(set(cls.STRATEGIES.keys()))
