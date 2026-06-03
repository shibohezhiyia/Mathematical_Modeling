"""
向量化工具集 - 替代低效循环和 apply 操作

提供高性能的向量化替代方案，用于替换：
- np.apply_along_axis
- pd.DataFrame.apply(lambda...)
- df.iterrows() / df.iteritems()
"""

import numpy as np
import pandas as pd
from typing import List, Callable, Any, Dict, Optional


# ===================== 分类投票向量化 =====================

def hard_vote_vectorized(preds: np.ndarray) -> np.ndarray:
    """
    硬投票的向量化实现
    
    替代:
        np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)
    
    参数:
        preds: (n_samples, n_models) 的预测标签数组
        
    返回:
        (n_samples,) 的多数投票结果
    """
    if preds.ndim == 1:
        return preds
    
    n_samples, n_models = preds.shape
    # 获取所有可能的类别
    n_classes = int(preds.max()) + 1
    
    # 使用 one-hot 编码 + 求和
    # shape: (n_samples, n_models, n_classes)
    one_hot = np.zeros((n_samples, n_models, n_classes), dtype=np.int32)
    rows = np.arange(n_samples)[:, None]
    cols = np.arange(n_models)[None, :]
    one_hot[rows, cols, preds] = 1
    
    # 每个样本的每个类别的票数
    votes = one_hot.sum(axis=1)  # (n_samples, n_classes)
    
    return votes.argmax(axis=1)


def hard_vote_scipy(preds: np.ndarray) -> np.ndarray:
    """
    使用 scipy.stats.mode 的硬投票（更简洁）
    
    需要 scipy 可用，否则回退到 hard_vote_vectorized
    """
    try:
        from scipy import stats
        mode_result = stats.mode(preds, axis=1, keepdims=False)
        return mode_result.mode
    except ImportError:
        return hard_vote_vectorized(preds)


# ===================== 编码映射向量化 =====================

def fast_encode_map(values: pd.Series, mapping: Dict[Any, Any], 
                     default: Any = -1) -> np.ndarray:
    """
    快速编码映射 - 替代 pd.apply(lambda...)
    
    替代:
        X_out[col] = values.apply(lambda v: enc.transform([v])[0] if v in known else -1)
    
    参数:
        values: pd.Series 待编码值
        mapping: 编码映射字典
        default: 未知值的默认值
        
    返回:
        np.ndarray 编码后的值
    """
    # 使用 map + fillna 替代 apply(lambda...)
    # 比 apply 快 10-100x
    return values.map(mapping).fillna(default).values.astype(np.int32)


def batch_transform_encoder(X: pd.DataFrame, encoders: Dict[str, Any],
                            unknown_value: int = -1) -> pd.DataFrame:
    """
    批量编码转换 - 向量化处理多列
    
    替代逐列 apply(lambda...)
    """
    X_out = X.copy()
    
    for col, enc in encoders.items():
        if col not in X_out.columns:
            continue
        
        # 获取已知的类别
        if hasattr(enc, 'categories_'):
            known = set()
            for cats in enc.categories_:
                known.update(cats)
        elif hasattr(enc, 'classes_'):
            known = set(enc.classes_)
        else:
            known = set(X_out[col].dropna().unique())
        
        # 构建映射字典
        mapping = {}
        for val in known:
            try:
                transformed = enc.transform([val])
                if hasattr(transformed, '__len__') and len(transformed) > 0:
                    mapping[val] = transformed[0] if not hasattr(transformed[0], '__len__') else transformed[0][0]
            except Exception:
                pass
        
        # 使用向量化映射
        X_out[col] = fast_encode_map(X_out[col], mapping, default=unknown_value)
    
    return X_out


# ===================== DataFrame 遍历优化 =====================

def iterrows_to_dict(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    将 DataFrame 转换为字典列表 - 比 iterrows() 快
    
    iterrows() 每行创建 Series，开销大。
    to_dict('records') 一次性创建所有字典。
    """
    return df.to_dict('records')


def vectorized_row_op(df: pd.DataFrame, columns: List[str],
                      op: Callable[..., Any]) -> np.ndarray:
    """
    对 DataFrame 多列执行向量化操作
    
    替代:
        for idx, row in df.iterrows():
            result[idx] = op(row[col1], row[col2], ...)
    
    参数:
        df: DataFrame
        columns: 参与运算的列名
        op: 接受 numpy arrays 的函数
        
    返回:
        np.ndarray 结果
    """
    arrays = [df[col].values for col in columns]
    return op(*arrays)


# ===================== 数值稳定性工具 =====================

def safe_inverse(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    安全的倒数计算，避免除零和溢出
    
    替代:
        1.0 / (x + 1e-6)
    """
    x_safe = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x_safe)
    mask = x_safe > epsilon
    result[mask] = 1.0 / x_safe[mask]
    # 对于接近0的值，使用 epsilon 的倒数
    result[~mask] = 1.0 / epsilon
    return result


def safe_logspace(low: float, high: float, n: int) -> np.ndarray:
    """
    安全的对数空间采样
    
    处理 low <= 0 的情况，自动调整到有效范围
    """
    low = float(low)
    high = float(high)
    
    if low <= 0:
        # 使用相对尺度：如果 high > 1，从 1e-10 开始；否则从 high/1000 开始
        low = min(1e-10, high / 1000.0)
    
    return np.logspace(np.log10(low), np.log10(high), n)


def check_matrix_condition(K: np.ndarray, threshold: float = 1e12) -> bool:
    """
    检查矩阵条件数，判断是否需要正则化
    
    返回 True 表示矩阵条件良好
    """
    try:
        # 使用 SVD 估计条件数（比 np.linalg.cond 更稳定）
        s = np.linalg.svd(K, compute_uv=False)
        if len(s) == 0 or s[-1] == 0:
            return False
        cond = s[0] / s[-1]
        return cond < threshold
    except Exception:
        return False


# ===================== 缓存优化工具 =====================

def cache_unique(y: np.ndarray) -> tuple:
    """
    缓存 np.unique 结果，避免重复计算
    
    返回: (unique_values, counts, n_classes)
    """
    unique_vals, counts = np.unique(y, return_counts=True)
    return unique_vals, counts, len(unique_vals)


# ===================== 快速互信息计算 =====================

def fast_mutual_info(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    快速互信息估计（使用直方图法）
    
    比 sklearn 的 mutual_info_regression 更快，适合大批量特征筛选
    """
    # 将连续变量分箱
    x_edges = np.histogram_bin_edges(X, bins=n_bins)
    y_edges = np.histogram_bin_edges(y, bins=n_bins)
    x_binned = np.digitize(X, x_edges)
    y_binned = np.digitize(y, y_edges)
    
    # 联合直方图
    joint_hist, _, _ = np.histogram2d(x_binned, y_binned, bins=n_bins)
    joint_prob = joint_hist / joint_hist.sum()
    
    # 边缘概率
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)
    
    # 互信息 - 向量化计算（避免 Python 嵌套循环）
    # 只计算非零项
    mask = joint_prob > 0
    x_prob_safe = np.where(x_prob > 0, x_prob, 1.0)
    y_prob_safe = np.where(y_prob > 0, y_prob, 1.0)
    
    # 广播得到联合分母
    denom = np.outer(x_prob_safe, y_prob_safe)
    denom = np.where(denom > 0, denom, 1.0)
    
    mi_vals = np.zeros_like(joint_prob)
    mi_vals[mask] = joint_prob[mask] * np.log(joint_prob[mask] / denom[mask])
    mi = mi_vals.sum()
    
    return max(0.0, mi)
