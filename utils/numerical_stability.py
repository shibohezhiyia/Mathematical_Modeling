"""
数值稳定性优化模块

解决核心数学计算中的数值稳定性问题：
1. 矩阵求逆/求解的稳定性
2. 除零保护
3. 对数/指数计算的边界处理
4. 权重计算的平滑化
"""

import numpy as np
from typing import Optional, Tuple


def stable_solve(K: np.ndarray, b: np.ndarray, 
                 reg: float = 1e-6,
                 cond_threshold: float = 1e12) -> Tuple[np.ndarray, bool]:
    """
    稳定的线性方程组求解
    
    替代 np.linalg.solve，添加：
    - 正则化（处理奇异矩阵）
    - 条件数检查
    - 自动回退到最小二乘
    
    参数:
        K: 系数矩阵 (n, n)
        b: 右侧向量 (n,) 或 (n, m)
        reg: 正则化强度
        cond_threshold: 条件数阈值
        
    返回:
        (solution, is_stable)
        solution: 解向量
        is_stable: 是否稳定求解
    """
    K = np.asarray(K, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    n = K.shape[0]
    
    # 添加正则化
    K_reg = K + reg * np.eye(n)
    
    try:
        # 检查条件数
        s = np.linalg.svd(K_reg, compute_uv=False)
        if s[-1] > 0:
            cond = s[0] / s[-1]
            is_stable = cond < cond_threshold
        else:
            is_stable = False
        
        if is_stable:
            # 条件良好，使用直接求解
            solution = np.linalg.solve(K_reg, b)
        else:
            # 条件数过高，使用最小二乘
            solution = np.linalg.lstsq(K_reg, b, rcond=None)[0]
        
        return solution, is_stable
    
    except np.linalg.LinAlgError:
        # 完全奇异，回退到伪逆
        K_pinv = np.linalg.pinv(K_reg)
        solution = K_pinv @ b
        return solution, False


def smooth_inverse(x: np.ndarray, epsilon: float = 1e-6,
                   smoothness: float = 0.1) -> np.ndarray:
    """
    平滑化的倒数计算
    
    替代 1.0 / (x + epsilon)，提供更平滑的过渡
    
    数学公式：
        f(x) = 1 / (x + epsilon) * sigmoid((x - epsilon) / smoothness)
        
    这样在 x 接近 0 时不会产生突变
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 基础倒数
    inv = np.zeros_like(x)
    mask = x > epsilon
    inv[mask] = 1.0 / x[mask]
    
    # 平滑过渡区
    transition = (x > 0) & (x <= epsilon)
    if np.any(transition):
        # 使用线性插值平滑过渡
        inv[transition] = 1.0 / epsilon * (x[transition] / epsilon)
    
    return inv


def softmax_with_temperature(x: np.ndarray, 
                             temperature: float = 1.0,
                             axis: int = -1) -> np.ndarray:
    """
    带温度参数的 softmax
    
    温度参数控制分布的"尖锐"程度：
    - T > 1: 更平滑（探索性）
    - T < 1: 更尖锐（贪婪）
    - T = 1: 标准 softmax
    
    数值稳定性：减去最大值防止溢出
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 数值稳定性：减去最大值
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # 应用温度
    if temperature != 1.0:
        x_shifted = x_shifted / temperature
    
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    return exp_x / sum_exp


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    稳定的 log-sum-exp 计算
    
    计算 log(sum(exp(x)))，避免数值溢出
    
    公式：log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def stable_ensemble_weights(scores: np.ndarray,
                          correlations: Optional[np.ndarray] = None,
                          diversity_penalty: float = 0.3) -> np.ndarray:
    """
    稳定的集成权重计算（考虑模型多样性）
    
    替代简单的分数归一化，添加：
    - 负相关奖励（多样性）
    - softmax 平滑化
    - 最小权重保护
    
    参数:
        scores: (n_models,) 各模型 CV 分数
        correlations: (n_models, n_models) 预测相关性矩阵
        diversity_penalty: 多样性惩罚强度
        
    返回:
        (n_models,) 归一化权重
    """
    scores = np.asarray(scores, dtype=np.float64)
    
    # 确保分数为正
    scores = np.maximum(scores, 1e-6)
    
    # 基础权重：softmax 平滑化
    weights = softmax_with_temperature(scores, temperature=0.5)
    
    # 如果提供了相关性矩阵，应用多样性调整（向量化避免 O(n²) Python 循环）
    if correlations is not None and correlations.ndim == 2:
        n = len(scores)
        # 创建高权重掩码
        high_weight_mask = weights > 0.1
        if np.any(high_weight_mask):
            # 向量化：对每一行，计算与所有高权重模型的平均相关性
            # 只考虑上三角（排除对角线自相关）
            corr_abs = np.abs(correlations)
            np.fill_diagonal(corr_abs, 0.0)
            # 只保留高权重模型的列
            masked_corr = np.where(high_weight_mask, corr_abs, 0.0)
            # 计算每个模型与其他高权重模型的平均相关性
            avg_corr = masked_corr.sum(axis=1) / np.maximum(high_weight_mask.sum(), 1)
            # 高相关性降低权重
            weights *= (1.0 - diversity_penalty * avg_corr)
            
            # 重新归一化
            weights = np.maximum(weights, 0.01)  # 最小权重保护
            weights = weights / weights.sum()
    
    return weights


def stable_variance(x: np.ndarray, ddof: int = 1) -> float:
    """
    稳定的方差计算
    
    小数组（<=10000）使用 numpy 高效计算；大数组使用 Welford 算法保证数值稳定性。
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n <= ddof:
        return 0.0
    
    # 小数组直接用 numpy（C 级实现，更快）
    if n <= 10000:
        return float(np.var(x, ddof=ddof))
    
    # 大数组使用 Welford 算法（避免数值溢出）
    mean = 0.0
    M2 = 0.0
    
    for i, val in enumerate(x):
        delta = val - mean
        mean += delta / (i + 1)
        delta2 = val - mean
        M2 += delta * delta2
    
    return M2 / (n - ddof)
