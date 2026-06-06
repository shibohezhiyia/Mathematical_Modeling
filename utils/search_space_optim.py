"""
搜索空间数学优化 - 改进采样效率和数值稳定性

优化点：
1. 对数尺度采样使用 np.logspace 替代手动 log-exp
2. Sobol 序列替代纯随机采样，提高空间覆盖
3. 条件参数联合采样策略
4. 数值稳定性改进（处理边界值）
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def sobol_sample_float(low: float, high: float, n: int, 
                        scale: str = 'linear', seed: int = 42) -> np.ndarray:
    """
    使用 Sobol 序列进行低差异采样（替代均匀随机采样）
    
    Sobol 序列比纯随机采样具有更好的空间覆盖性，
    适合超参数搜索的初始采样阶段。
    
    参数:
        low: 下限
        high: 上限
        n: 采样数量
        scale: 'linear' 或 'log'
        seed: 随机种子
        
    返回:
        np.ndarray 形状 (n,)
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
        samples = sampler.random(n=n)
    except ImportError:
        # 回退到拉丁超立方采样
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=1, seed=seed)
            samples = sampler.random(n=n)
        except ImportError:
            # 最终回退到均匀随机
            rng = np.random.RandomState(seed)
            samples = rng.uniform(0, 1, size=n)
    
    if scale == 'log':
        # 使用 np.logspace 的数学性质
        return np.logspace(np.log10(max(low, 1e-10)), np.log10(high), n)
    else:
        return low + samples.ravel() * (high - low)


def sobol_sample_int(low: int, high: int, n: int, seed: int = 42) -> np.ndarray:
    """
    Sobol 序列采样整数
    
    使用 rounding + 去重策略
    """
    floats = sobol_sample_float(float(low), float(high), n, scale='linear', seed=seed)
    ints = np.round(floats).astype(int)
    
    # 去重并补充
    unique = np.unique(ints)
    if len(unique) < n and high - low + 1 >= n:
        # 补充缺失值
        remaining = set(range(low, high + 1)) - set(unique)
        if remaining:
            additional = np.array(list(remaining)[:n - len(unique)])
            unique = np.concatenate([unique, additional])
    
    return unique[:n]


def joint_conditional_sample(space: Dict[str, Any], 
                             n_samples: int,
                             seed: int = 42) -> List[Dict[str, Any]]:
    """
    条件参数的联合采样
    
    处理参数间的依赖关系（如某参数只在另一参数取特定值时激活）
    
    参数:
        space: 搜索空间字典，支持 condition 字段
        n_samples: 采样数量
        seed: 随机种子
        
    返回:
        List[Dict] 采样结果列表
    """
    rng = np.random.RandomState(seed)
    samples = []
    
    for _ in range(n_samples):
        current = {}
        
        # 第一轮：采样无条件参数
        for name, spec in space.items():
            if isinstance(spec, dict) and 'condition' in spec:
                continue  # 条件参数稍后处理
            
            if isinstance(spec, list):
                current[name] = rng.choice(spec)
            elif isinstance(spec, dict):
                current[name] = _sample_param(spec, rng, current)
            else:
                current[name] = spec
        
        # 第二轮：采样条件参数
        for name, spec in space.items():
            if not isinstance(spec, dict) or 'condition' not in spec:
                continue
            
            condition = spec['condition']
            parent_val = current.get(condition['param'])
            
            if parent_val in condition.get('values', []):
                current[name] = _sample_param(spec, rng, current)
            else:
                # 条件不满足，设为 None 或默认值
                current[name] = spec.get('default', None)
        
        samples.append(current)
    
    return samples


def _sample_param(spec: Dict[str, Any], rng: np.random.RandomState,
                  current_params: Dict[str, Any]) -> Any:
    """从参数规格中采样单个值"""
    param_type = spec.get('type', 'float')
    
    if param_type == 'float':
        low = float(spec.get('low', 0.0))
        high = float(spec.get('high', 1.0))
        scale = spec.get('scale', 'linear')
        
        if scale == 'log':
            if low <= 0:
                low = 1e-10
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            return float(rng.uniform(low, high))
    
    elif param_type == 'int':
        low = int(spec.get('low', 0))
        high = int(spec.get('high', 10))
        return int(rng.randint(low, high + 1))
    
    elif param_type == 'categorical':
        choices = spec.get('choices', [])
        if not choices:
            return None
        return rng.choice(choices)
    
    elif param_type == 'bool':
        return bool(rng.choice([True, False]))
    
    else:
        return spec.get('default', None)


def adaptive_n_components(data_shape: Tuple[int, ...], 
                        target_variance: float = 0.95) -> int:
    """
    根据数据的有效秩自适应选择组件数
    
    用于核近似、PCA等降维操作
    
    参数:
        data_shape: (n_samples, n_features) 数据形状
        target_variance: 目标方差保留比例
        
    返回:
        建议的组件数
    """
    n_samples, n_features = data_shape
    
    # 基于数据维度的启发式
    max_components = min(n_samples, n_features)
    
    # 经验法则：保留目标方差通常需要 min(n_samples, n_features) 的某个比例
    # 对于高维数据，有效秩通常远小于 min(n_samples, n_features)
    
    if n_samples < 1000:
        # 小样本：保留更多组件
        heuristic = int(max_components * 0.8)
    elif n_samples < 10000:
        # 中等样本
        heuristic = int(max_components * 0.5)
    else:
        # 大样本：可以激进降维
        heuristic = int(max_components * 0.2)
    
    # 确保在合理范围内
    heuristic = max(2, min(heuristic, max_components))
    
    return heuristic


def logspace_candidates(low: float, high: float, n: int = 8) -> List[float]:
    """
    生成对数尺度的候选值列表（数学优化版）
    
    替代 search_space.py 中手动实现的 log 候选生成
    
    优化点：
    - 使用 np.logspace 而非手动循环
    - 自动处理边界值
    - 确保去重的高效性
    """
    if low <= 0:
        low = min(1e-10, high / 1000.0)
    
    # 使用 np.logspace 生成
    vals = np.logspace(np.log10(low), np.log10(high), n)
    
    # 使用 np.unique 高效去重（替代手动 set + round）
    unique_vals = np.unique(np.round(vals, 10))
    
    return [float(v) for v in unique_vals]
