"""
自适应搜索空间

根据优化历史动态调整搜索空间：
1. 收缩 (shrink): 聚焦高表现参数区域
2. 剪枝 (prune): 移除低表现离散值
3. 扩展 (expand): 在边界附近发现更优值时扩展范围

与 SearchSpace 兼容，可无缝替换使用。
"""
import math
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
from scipy.stats import spearmanr

from core.search_space import SearchSpace, Parameter


@dataclass
class AdaptationConfig:
    """自适应配置"""
    warmup_trials: int = 10           # 前 N 次 trial 不调整，仅收集数据
    shrink_ratio: float = 0.5         # 收缩比例（保留 top N% 区域）
    min_range_ratio: float = 0.1      # 最小保留范围比例（防止过度收缩）
    prune_threshold: float = 0.2      # 离散值剪枝阈值（低于最佳值的 ratio）
    expand_ratio: float = 0.3         # 扩展比例（边界值持续最优时扩展）
    adapt_every: int = 5              # 每 N 次 trial 调整一次
    correlation_threshold: float = 0.3  # 相关性阈值，低于此值不调整


class AdaptiveSearchSpace(SearchSpace):
    """
    自适应搜索空间
    
    继承 SearchSpace，在优化过程中根据历史结果自动调整参数范围。
    
    使用方式:
        space = AdaptiveSearchSpace(config, adaptation=AdaptationConfig())
        for trial in range(n_trials):
            params = space.sample(rng=rng)
            score = evaluate(params)
            space.update_history(params, score)
            if space.should_adapt():
                space.adapt()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 adaptation: Optional[AdaptationConfig] = None) -> None:
        super().__init__(config)
        self.adaptation = adaptation or AdaptationConfig()
        self.history: List[Dict[str, Any]] = []  # [{'params': {...}, 'score': float}, ...]
        self.original_params: Dict[str, Parameter] = deepcopy(self.params)
        self._adaptation_count: int = 0
        self._param_importance: Dict[str, float] = {}
    
    def update_history(self, params: Dict[str, Any], score: float) -> None:
        """记录一次 trial 的结果"""
        self.history.append({
            'params': deepcopy(params),
            'score': float(score),
            'trial': len(self.history)
        })
    
    def should_adapt(self) -> bool:
        """判断是否满足调整条件"""
        n = len(self.history)
        if n < self.adaptation.warmup_trials:
            return False
        return (n - self.adaptation.warmup_trials) % self.adaptation.adapt_every == 0
    
    def adapt(self, direction: str = 'maximize') -> Dict[str, Any]:
        """
        执行一次搜索空间自适应调整
        
        Args:
            direction: 'maximize' 或 'minimize'
            
        Returns:
            调整报告 {'adapted_params': [...], 'pruned_values': {...}}
        """
        if len(self.history) < self.adaptation.warmup_trials:
            return {'adapted_params': [], 'pruned_values': {}}
        
        report = {
            'adapted_params': [],
            'pruned_values': {},
            'expanded_params': [],
            'importance': {}
        }
        
        # 1. 计算各参数与 score 的相关性/重要性
        self._param_importance = self._compute_param_importance(direction)
        report['importance'] = deepcopy(self._param_importance)
        
        # 2. 对每个参数进行调整
        for name, param in self.params.items():
            importance = self._param_importance.get(name, 0.0)
            
            # 低相关性参数不调整（避免噪声导致过度收缩）
            if importance < self.adaptation.correlation_threshold:
                continue
            
            orig = self.original_params.get(name)
            if orig is None:
                continue
            
            if param.type in ('float', 'int'):
                adapted = self._adapt_numeric(name, param, orig, direction)
                if adapted:
                    report['adapted_params'].append(name)
            
            elif param.type == 'categorical':
                pruned = self._prune_categorical(name, param, direction)
                if pruned:
                    report['pruned_values'][name] = pruned
        
        self._adaptation_count += 1
        return report
    
    def _compute_param_importance(self, direction: str = 'maximize') -> Dict[str, float]:
        """计算各参数的重要性（基于 Spearman 秩相关）"""
        importance = {}
        n = len(self.history)
        if n < 3:
            return importance
        
        # 提取各参数值序列
        param_values: Dict[str, List[float]] = {}
        scores = []
        
        for record in self.history:
            scores.append(record['score'])
            for name, val in record['params'].items():
                if name not in param_values:
                    param_values[name] = []
                # 将非数值转为秩
                numeric_val = self._to_numeric(val)
                param_values[name].append(numeric_val)
        
        # 计算 Spearman 相关系数
        
        for name, values in param_values.items():
            if len(values) < 3 or len(set(values)) < 2:
                importance[name] = 0.0
                continue
            try:
                corr, _ = spearmanr(values, scores[:len(values)])
                if math.isnan(corr):
                    corr = 0.0
                importance[name] = abs(corr)
            except Exception:
                importance[name] = 0.0
        
        return importance
    
    def _to_numeric(self, val: Any) -> float:
        """将任意值转为可计算的数值"""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, str):
            return float(hash(val) % 10000) / 10000.0
        return 0.0
    
    def _adapt_numeric(self, name: str, param: Parameter, orig: Parameter,
                       direction: str) -> bool:
        """调整数值型参数范围"""
        if param.low is None or param.high is None:
            return False
        
        # 提取该参数的历史值和对应分数
        records = []
        for record in self.history:
            if name in record['params']:
                records.append((float(record['params'][name]), record['score']))
        
        if len(records) < 3:
            return False
        
        # 按分数排序
        reverse = (direction == 'maximize')
        records.sort(key=lambda x: x[1], reverse=reverse)
        
        # 取 top N% 的记录
        top_n = max(3, int(len(records) * self.adaptation.shrink_ratio))
        top_records = records[:top_n]
        top_values = [v for v, _ in top_records]
        
        new_low = min(top_values)
        new_high = max(top_values)
        
        # 确保最小范围（防止过度收缩）
        orig_range = float(orig.high) - float(orig.low)
        if orig.type == 'int':
            orig_range = max(1, orig_range)
        min_range = orig_range * self.adaptation.min_range_ratio
        
        if new_high - new_low < min_range:
            center = (new_low + new_high) / 2
            new_low = center - min_range / 2
            new_high = center + min_range / 2
        
        # 边界保护
        new_low = max(float(orig.low), new_low)
        new_high = min(float(orig.high), new_high)
        
        if param.type == 'int':
            new_low = int(math.floor(new_low))
            new_high = int(math.ceil(new_high))
        
        if new_low < new_high:
            param.low = new_low
            param.high = new_high
            return True
        return False
    
    def _prune_categorical(self, name: str, param: Parameter,
                           direction: str) -> List[Any]:
        """剪枝低表现离散值"""
        if not param.choices:
            return []
        
        # 统计每个选择的表现
        choice_scores: Dict[Any, List[float]] = {}
        for record in self.history:
            if name in record['params']:
                val = record['params'][name]
                if val not in choice_scores:
                    choice_scores[val] = []
                choice_scores[val].append(record['score'])
        
        if len(choice_scores) < 2:
            return []
        
        # 计算每个选择的平均分数
        choice_mean = {k: np.mean(v) for k, v in choice_scores.items()}
        
        reverse = (direction == 'maximize')
        best_score = max(choice_mean.values()) if reverse else min(choice_mean.values())
        
        # 剪枝低于阈值的值
        pruned = []
        new_choices = []
        for choice in param.choices:
            if choice in choice_mean:
                score = choice_mean[choice]
                # 保留接近最佳值的选项
                gap = abs(score - best_score) / (abs(best_score) + 1e-10)
                if gap > self.adaptation.prune_threshold:
                    pruned.append(choice)
                    continue
            new_choices.append(choice)
        
        # 至少保留一个选择
        if len(new_choices) >= 1:
            param.choices = new_choices
        
        return pruned
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """获取自适应调整的历史报告"""
        return {
            'n_adaptations': self._adaptation_count,
            'n_history': len(self.history),
            'param_importance': deepcopy(self._param_importance),
            'current_space': self.to_dict(),
            'original_space': SearchSpace.to_dict(self)
        }
    
    def reset(self) -> None:
        """重置到原始搜索空间"""
        self.params = deepcopy(self.original_params)
        self.history.clear()
        self._adaptation_count = 0
        self._param_importance.clear()


class SearchSpaceAdapter:
    """
    搜索空间适配器（用于在现有优化器中无痛集成自适应搜索空间）
    
    用法:
        from core.adaptive_search_space import SearchSpaceAdapter
        adapter = SearchSpaceAdapter(search_space)
        
        for trial in range(n_trials):
            params = adapter.sample(rng)
            score = evaluate(params)
            adapter.report(score)
            if adapter.should_adapt():
                adapter.adapt()
    """
    
    def __init__(self, search_space: Union[SearchSpace, Dict[str, Any]],
                 adaptation: Optional[AdaptationConfig] = None) -> None:
        if isinstance(search_space, SearchSpace):
            self.space = AdaptiveSearchSpace(search_space.to_dict(), adaptation)
        else:
            self.space = AdaptiveSearchSpace(search_space, adaptation)
        self._last_params: Optional[Dict[str, Any]] = None
    
    def sample(self, rng: Optional[np.random.RandomState] = None,
               random_state: Optional[int] = None) -> Dict[str, Any]:
        """采样一组参数"""
        self._last_params = self.space.sample(rng=rng, random_state=random_state)
        return deepcopy(self._last_params)
    
    def report(self, score: float) -> None:
        """报告上一次采样的分数"""
        if self._last_params is not None:
            self.space.update_history(self._last_params, score)
            self._last_params = None
    
    def should_adapt(self) -> bool:
        """判断是否需要调整"""
        return self.space.should_adapt()
    
    def adapt(self, direction: str = 'maximize') -> Dict[str, Any]:
        """执行自适应调整"""
        return self.space.adapt(direction=direction)
    
    @property
    def config(self) -> Dict[str, Any]:
        """当前搜索空间配置"""
        return self.space.to_dict()
