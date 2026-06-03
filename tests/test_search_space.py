import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_space import SearchSpace, Parameter
import numpy as np


class TestSearchSpace:
    """测试搜索空间"""

    def test_parameter_float(self):
        """测试浮点参数"""
        p = Parameter(name='lr', type='float', low=0.01, high=0.1, scale='log')
        val = p.sample(np.random.RandomState(42))
        assert 0.01 <= val <= 0.1, "Should be in range"

    def test_parameter_int(self):
        """测试整数参数"""
        p = Parameter(name='n', type='int', low=1, high=10)
        val = p.sample(np.random.RandomState(42))
        assert 1 <= val <= 10, "Should be in range"
        assert isinstance(val, int), "Should be integer"

    def test_parameter_categorical(self):
        """测试分类参数"""
        p = Parameter(name='model', type='categorical', choices=['a', 'b', 'c'])
        val = p.sample(np.random.RandomState(42))
        assert val in ['a', 'b', 'c'], "Should be in choices"

    def test_search_space_sample(self):
        """测试搜索空间采样"""
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'n': {'type': 'int', 'low': 1, 'high': 10}
        })
        params = space.sample(random_state=42)
        assert 'lr' in params, "Should contain lr"
        assert 'n' in params, "Should contain n"

    def test_search_space_sample_sobol(self):
        """测试Sobol采样"""
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'n': {'type': 'int', 'low': 1, 'high': 10}
        })
        params_list = space.sample_sobol(n=5, random_state=42)
        assert len(params_list) == 5, "Should return 5 samples"

    def test_search_space_build_candidates(self):
        """测试构建候选值"""
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 0.01, 'high': 0.1},
            'n': {'type': 'int', 'low': 1, 'high': 10}
        })
        candidates = space.build_candidates(n=5)
        assert 'lr' in candidates, "Should contain lr candidates"
        assert len(candidates['lr']) <= 5, "Should not exceed n"

    def test_condition_param(self):
        """测试条件参数"""
        space = SearchSpace({
            'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart']},
            'sample_type': {'type': 'categorical', 'choices': ['uniform', 'weighted'],
                           'condition': {'param': 'booster', 'values': ['dart']}}
        })
        params = space.sample(random_state=42)
        # If booster is dart, sample_type should be present
        if params.get('booster') == 'dart':
            assert 'sample_type' in params, "Should contain sample_type when booster=dart"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
