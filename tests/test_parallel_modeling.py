import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parallel_modeling import ModelRegistry, ParallelModelingEngine
from core.performance_scheduler import ExecutionPlan
import pandas as pd
import numpy as np


def _make_engine(task_type: str) -> ParallelModelingEngine:
    """按当前 API 构造引擎(n_jobs=1 走串行训练)"""
    plan = ExecutionPlan(n_jobs=1, cv_folds=2, hyperparameter_trials=0, use_gpu=False)
    return ParallelModelingEngine(task_type=task_type, plan=plan, verbose=False)


class TestParallelModeling:
    """测试并行建模引擎"""

    def test_model_registry(self):
        """测试模型注册表"""
        models = ModelRegistry.get_available_models('classification')
        assert isinstance(models, dict), "Should return dict of models"
        assert len(models) > 0, "Should have at least one model registered"

    def test_parallel_engine_classification(self):
        """测试并行分类"""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])

        engine = _make_engine('classification')
        result = engine.fit(df, y)
        assert result is not None, "Should return modeling result"
        assert len(engine.results) > 0, "Should have trained at least one model"

    def test_parallel_engine_regression(self):
        """测试并行回归"""
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])

        engine = _make_engine('regression')
        result = engine.fit(df, y)
        assert result is not None, "Should return modeling result"
        assert len(engine.results) > 0, "Should have trained at least one model"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
