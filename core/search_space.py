"""
丰富格式的超参数搜索空间定义与采样

支持格式:
  1. 向后兼容 — 列表: ['l1', 'l2']
  2. 向后兼容 — 单值: 42
  3. 丰富格式 — dict:
     {
       'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log',
       'condition': {'param': 'booster', 'values': ['gbtree']}
     }
     {
       'type': 'categorical', 'choices': ['gbtree', 'dart'],
       'labels': ['梯度提升树', 'Dropouts']
     }
     {'type': 'int', 'low': 3, 'high': 10}
     {'type': 'bool'}

使用方式:
    space = SearchSpace({'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'scale': 'log'}})
    params = space.sample(rng=np.random.RandomState(42))
    # params => {'lr': 0.0032}
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
import math
import numpy as np


@dataclass
class Parameter:
    """单个超参数定义"""
    name: str
    type: str = 'float'  # 'float', 'int', 'categorical', 'bool'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    labels: Optional[List[str]] = None
    scale: str = 'linear'  # 'linear', 'log'
    condition: Optional[Dict[str, Any]] = None
    # condition = {'param': 'parent_name', 'values': [val1, val2]}
    
    def is_active(self, current_params: Optional[Dict[str, Any]] = None) -> bool:
        """检查在当前配置下此参数是否激活"""
        if self.condition is None:
            return True
        if current_params is None:
            return False
        parent_val = current_params.get(self.condition['param'])
        return parent_val in self.condition['values']
    
    def sample(self, rng: np.random.RandomState,
               current_params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """从参数空间中采样一个值；条件不满足时返回 None"""
        if not self.is_active(current_params):
            return None
        
        if self.type == 'float':
            lo, hi = float(self.low), float(self.high)
            if self.scale == 'log':
                if lo <= 0:
                    lo = 1e-10
                log_lo, log_hi = math.log(lo), math.log(hi)
                return float(math.exp(rng.uniform(log_lo, log_hi)))
            else:
                return float(rng.uniform(lo, hi))
        
        elif self.type == 'int':
            return int(rng.randint(int(self.low), int(self.high) + 1))
        
        elif self.type == 'categorical':
            choices = self.choices if self.choices is not None else []
            if not choices:
                return None
            return rng.choice(choices)
        
        elif self.type == 'bool':
            return bool(rng.choice([True, False]))
        
        else:
            raise ValueError(f"[Parameter] 未知参数类型: {self.type}")
    
    def build_candidates(self, n: int = 8) -> List[Any]:
        """为离散化优化器（RL/GA）生成候选值列表"""
        if self.type == 'float':
            lo, hi = float(self.low), float(self.high)
            if self.scale == 'log':
                if lo <= 0:
                    lo = 1e-10
                log_vals = np.linspace(math.log(lo), math.log(hi), n)
                vals = [float(math.exp(v)) for v in log_vals]
            else:
                vals = [float(v) for v in np.linspace(lo, hi, n)]
            # 去重（log scale 可能导致边界重复）
            seen = set()
            unique = []
            for v in vals:
                key = round(v, 10)
                if key not in seen:
                    seen.add(key)
                    unique.append(v)
            return unique
        
        elif self.type == 'int':
            lo, hi = int(self.low), int(self.high)
            total = hi - lo + 1
            if total <= n:
                return list(range(lo, hi + 1))
            step = max(1, total // n)
            vals = list(range(lo, hi + 1, step))
            if len(vals) > n:
                vals = vals[:n]
            return vals
        
        elif self.type == 'categorical':
            return list(self.choices) if self.choices else []
        
        elif self.type == 'bool':
            return [True, False]
        
        else:
            return []
    
    def to_optuna(self, trial: Any, current_params: Optional[Dict[str, Any]] = None) -> Any:
        """为 Optuna trial 建议参数值"""
        if not self.is_active(current_params):
            return None
        
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna 未安装")
        
        if self.type == 'float':
            lo, hi = float(self.low), float(self.high)
            if self.scale == 'log':
                if lo <= 0:
                    lo = 1e-10
                return trial.suggest_float(self.name, lo, hi, log=True)
            else:
                return trial.suggest_float(self.name, lo, hi)
        
        elif self.type == 'int':
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        
        elif self.type == 'categorical':
            choices = list(self.choices) if self.choices else []
            return trial.suggest_categorical(self.name, choices)
        
        elif self.type == 'bool':
            return trial.suggest_categorical(self.name, [True, False])
        
        else:
            raise ValueError(f"[Parameter] 未知参数类型: {self.type}")


class SearchSpace:
    """
    超参数搜索空间
    
    统一解析多种配置格式，提供一致的采样接口。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            config: 搜索空间配置字典。每个键对应一个参数，值可以是:
                - List[Any]: 离散候选值列表（向后兼容）
                - Dict: 丰富格式参数定义
                - Any: 固定值
        """
        self.config = config or {}
        self.params: Dict[str, Parameter] = {}
        self._parse_config(self.config)
    
    def _parse_config(self, config: Dict[str, Any]) -> None:
        """解析配置为 Parameter 对象"""
        for name, spec in config.items():
            if isinstance(spec, list):
                self.params[name] = Parameter(
                    name=name,
                    type='categorical',
                    choices=list(spec)
                )
            elif isinstance(spec, dict):
                self.params[name] = self._parse_dict_spec(name, spec)
            else:
                # 单个固定值
                self.params[name] = Parameter(
                    name=name,
                    type='categorical',
                    choices=[spec]
                )
    
    def _parse_dict_spec(self, name: str, spec: Dict[str, Any]) -> Parameter:
        """解析 dict 格式的参数定义"""
        ptype = spec.get('type', 'float')
        
        param = Parameter(
            name=name,
            type=ptype,
            condition=spec.get('condition')
        )
        
        if ptype == 'float':
            param.low = spec.get('low')
            param.high = spec.get('high')
            param.scale = spec.get('scale', 'linear')
        
        elif ptype == 'int':
            param.low = spec.get('low')
            param.high = spec.get('high')
        
        elif ptype == 'categorical':
            param.choices = list(spec.get('choices', []))
            param.labels = spec.get('labels')
        
        elif ptype == 'bool':
            pass
        
        else:
            raise ValueError(f"[SearchSpace] 未知参数类型 '{ptype}' for '{name}'")
        
        return param
    
    # -------------------------------------------------------------------------
    # 核心接口
    # -------------------------------------------------------------------------
    
    def sample(self, rng: Optional[np.random.RandomState] = None,
               random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        采样一组完整参数（自动处理条件依赖）
        
        按定义顺序逐个采样，确保条件参数能正确判断父参数值。
        """
        if rng is None:
            rng = np.random.RandomState(random_state)
        
        result = {}
        for name, param in self.params.items():
            val = param.sample(rng, result)
            if val is not None:
                result[name] = val
        return result
    
    def sample_many(self, n: int, rng: Optional[np.random.RandomState] = None,
                    random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """采样 n 组参数"""
        if rng is None:
            rng = np.random.RandomState(random_state)
        return [self.sample(rng=rng) for _ in range(n)]
    
    def build_candidates(self, n: int = 8) -> Dict[str, List[Any]]:
        """
        为每个参数生成离散候选值列表（用于 RL/GA 等离散优化器）
        
        条件参数也会被包含，由调用方根据实际配置过滤。
        """
        candidates = {}
        for name, param in self.params.items():
            cand = param.build_candidates(n)
            if cand:
                candidates[name] = cand
        return candidates
    
    def to_optuna(self, trial: Any, current_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        为 Optuna trial 建议所有参数值
        
        内部自动维护 current_params 以处理条件依赖。
        """
        result = current_params or {}
        for name, param in self.params.items():
            if name in result:
                continue
            val = param.to_optuna(trial, result)
            if val is not None:
                result[name] = val
        return result
    
    def get_active_params(self, current_params: Dict[str, Any]) -> List[str]:
        """获取在当前配置下激活的参数名列表"""
        return [name for name, p in self.params.items() if p.is_active(current_params)]
    
    def get_param(self, name: str) -> Optional[Parameter]:
        """获取指定参数定义"""
        return self.params.get(name)
    
    def get(self, name: str, default: Any = None) -> Any:
        """获取参数定义（兼容 dict.get 接口）"""
        param = self.params.get(name)
        if param is None:
            return default
        # 返回类似 dict 的格式，便于兼容旧代码
        return param.to_dict() if hasattr(param, 'to_dict') else param
    
    # -------------------------------------------------------------------------
    # 向后兼容接口
    # -------------------------------------------------------------------------
    
    def __contains__(self, key: str) -> bool:
        return key in self.params
    
    def __getitem__(self, key: str) -> Parameter:
        return self.params[key]
    
    def keys(self) -> Any:
        return self.params.keys()
    
    def items(self) -> Any:
        return self.params.items()
    
    def values(self) -> Any:
        return self.params.values()
    
    def __bool__(self) -> bool:
        return len(self.params) > 0
    
    def __len__(self) -> int:
        return len(self.params)
    
    def __repr__(self) -> str:
        lines = ["SearchSpace("]
        for name, p in self.params.items():
            cond = f", condition={p.condition}" if p.condition else ""
            if p.type == 'float':
                lines.append(f"  {name}: float({p.low}, {p.high}, scale={p.scale}){cond}")
            elif p.type == 'int':
                lines.append(f"  {name}: int({p.low}, {p.high}){cond}")
            elif p.type == 'categorical':
                labels = f", labels={p.labels}" if p.labels else ""
                lines.append(f"  {name}: categorical({p.choices}){labels}{cond}")
            elif p.type == 'bool':
                lines.append(f"  {name}: bool{cond}")
        lines.append(")")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为配置字典"""
        result = {}
        for name, p in self.params.items():
            if p.type == 'float':
                d = {'type': 'float', 'low': p.low, 'high': p.high}
                if p.scale != 'linear':
                    d['scale'] = p.scale
            elif p.type == 'int':
                d = {'type': 'int', 'low': p.low, 'high': p.high}
            elif p.type == 'categorical':
                d = {'type': 'categorical', 'choices': p.choices}
                if p.labels:
                    d['labels'] = p.labels
            elif p.type == 'bool':
                d = {'type': 'bool'}
            else:
                continue
            if p.condition:
                d['condition'] = p.condition
            result[name] = d
        return result
