import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parallel_modeling import ModelRegistry, ParallelModelingEngine
import pandas as pd
import numpy as np


class TestParallelModeling:
    """测试并行建模引擎"""

    def test_model_registry(self):
        """测试模型注册表"""
        registry = ModelRegistry()
        models = registry.list_models()
        assert isinstance(models, list), "Should return list of models"

    def test_parallel_engine_classification(self):
        """测试并行分类"""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        df['target'] = y
        
        engine = ParallelModelingEngine(n_jobs=1, verbose=False)
        result = engine.fit(df, 'target')
        assert result is not None, "Should return modeling result"

    def test_parallel_engine_regression(self):
        """测试并行回归"""
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        df['target'] = y
        
        engine = ParallelModelingEngine(n_jobs=1, verbose=False)
        result = engine.fit(df, 'target')
        assert result is not None, "Should return modeling result"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
