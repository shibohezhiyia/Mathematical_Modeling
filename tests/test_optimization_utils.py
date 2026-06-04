import pytest
import numpy as np
import pandas as pd
from utils.vectorization_utils import (
    hard_vote_vectorized, fast_encode_map, safe_inverse,
    safe_logspace, check_matrix_condition, cache_unique
)
from utils.search_space_optim import (
    sobol_sample_float, logspace_candidates, adaptive_n_components
)
from utils.numerical_stability import (
    stable_solve, smooth_inverse, softmax_with_temperature,
    stable_ensemble_weights
)


# ===================== 向量化工具测试 =====================

def test_hard_vote_vectorized():
    """测试硬投票向量化"""
    preds = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 2],
        [2, 2, 1]
    ])
    result = hard_vote_vectorized(preds)
    
    assert len(result) == 4
    assert result[0] == 0  # [0,0,1] -> 0
    assert result[1] == 1  # [1,1,1] -> 1
    assert result[2] in [0, 1, 2]  # [0,1,2] -> tie
    
    # 测试一维输入
    result_1d = hard_vote_vectorized(np.array([0, 1, 2]))
    assert np.array_equal(result_1d, np.array([0, 1, 2]))


def test_fast_encode_map():
    """测试快速编码映射"""
    values = pd.Series(['a', 'b', 'c', 'a', 'd'])
    mapping = {'a': 0, 'b': 1, 'c': 2}
    
    result = fast_encode_map(values, mapping, default=-1)
    
    assert len(result) == 5
    assert result[0] == 0  # 'a' -> 0
    assert result[1] == 1  # 'b' -> 1
    assert result[2] == 2  # 'c' -> 2
    assert result[3] == 0  # 'a' -> 0
    assert result[4] == -1  # 'd' -> unknown


def test_safe_inverse():
    """测试安全倒数"""
    x = np.array([1.0, 2.0, 0.0, 1e-7, 1e-5])
    result = safe_inverse(x, epsilon=1e-6)
    
    assert result[0] == 1.0  # 1/1 = 1
    assert result[1] == 0.5  # 1/2 = 0.5
    assert result[2] == 1e6  # 1/1e-6 = 1e6 (处理0值)
    assert result[3] == 1e6  # 1/1e-6 = 1e6 (接近0)
    assert result[4] == 1e5  # 1/1e-5 = 1e5


def test_safe_logspace():
    """测试安全对数空间采样"""
    # 正常范围
    result = safe_logspace(1e-5, 1.0, 5)
    assert len(result) == 5
    assert result[0] >= 1e-5
    assert result[-1] <= 1.0
    
    # low <= 0 的情况
    result_neg = safe_logspace(-1.0, 1.0, 5)
    assert len(result_neg) == 5
    assert result_neg[0] > 0


def test_check_matrix_condition():
    """测试矩阵条件数检查"""
    # 良好条件的矩阵
    good = np.eye(5)
    assert check_matrix_condition(good, threshold=1e12) == True
    
    # 接近奇异的矩阵
    bad = np.array([[1, 1], [1, 1.0001]])
    # 条件数约为 40000，应该通过
    assert check_matrix_condition(bad, threshold=1e12) == True


def test_cache_unique():
    """测试唯一值缓存"""
    y = np.array([0, 1, 1, 2, 0, 2, 2])
    unique, counts, n_classes = cache_unique(y)
    
    assert n_classes == 3
    assert len(unique) == 3
    assert len(counts) == 3
    assert np.array_equal(unique, np.array([0, 1, 2]))


# ===================== 搜索空间优化测试 =====================

def test_logspace_candidates():
    """测试对数尺度候选值生成"""
    candidates = logspace_candidates(1e-5, 1.0, 8)
    
    assert len(candidates) >= 2
    assert candidates[0] >= 1e-5
    assert candidates[-1] <= 1.0
    
    # 测试 low <= 0 的情况
    candidates_neg = logspace_candidates(-1.0, 1.0, 8)
    assert len(candidates_neg) >= 2
    assert candidates_neg[0] > 0


def test_adaptive_n_components():
    """测试自适应组件数选择"""
    # 小样本
    n1 = adaptive_n_components((100, 50), target_variance=0.95)
    assert 2 <= n1 <= 50
    
    # 大样本
    n2 = adaptive_n_components((100000, 1000), target_variance=0.95)
    assert 2 <= n2 <= 1000


# ===================== 数值稳定性测试 =====================

def test_stable_solve():
    """测试稳定求解"""
    # 简单方程组
    K = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([3, 3], dtype=float)
    
    solution, is_stable = stable_solve(K, b)
    
    assert is_stable == True
    assert np.allclose(solution, np.array([1.0, 1.0]), rtol=1e-5)


def test_smooth_inverse():
    """测试平滑倒数"""
    x = np.array([1.0, 0.5, 0.0, 1e-7])
    result = smooth_inverse(x, epsilon=1e-6)
    
    assert result[0] == 1.0
    assert result[1] == 2.0
    assert result[2] < 1e6  # 平滑处理，不会突变到 1/epsilon
    assert result[3] < 1e6


def test_softmax_with_temperature():
    """测试温度 softmax"""
    x = np.array([1.0, 2.0, 3.0])
    
    # 标准温度
    s1 = softmax_with_temperature(x, temperature=1.0)
    assert np.allclose(s1.sum(), 1.0)
    
    # 高温（更平滑）
    s2 = softmax_with_temperature(x, temperature=2.0)
    assert np.allclose(s2.sum(), 1.0)
    assert s2.max() < s1.max()  # 更平滑
    
    # 低温（更尖锐）
    s3 = softmax_with_temperature(x, temperature=0.5)
    assert np.allclose(s3.sum(), 1.0)
    assert s3.max() > s1.max()  # 更尖锐


def test_stable_ensemble_weights():
    """测试稳定集成权重"""
    scores = np.array([0.8, 0.9, 0.7])
    
    # 无相关性矩阵
    weights = stable_ensemble_weights(scores)
    assert np.allclose(weights.sum(), 1.0)
    assert len(weights) == 3
    
    # 有相关性矩阵（高相关性惩罚）
    corr = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0]
    ])
    weights_div = stable_ensemble_weights(scores, correlations=corr, diversity_penalty=0.5)
    assert np.allclose(weights_div.sum(), 1.0)
    # 模型0和1高度相关，权重应该比不相关时更低
    assert weights_div[2] > weights[2]  # 模型2与其他不相关，权重应相对更高
