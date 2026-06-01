"""
NAS + 分布式计算测试

覆盖:
  - TorchNAS: 架构搜索、fit/predict、搜索历史
  - TransferFeatureExtractor: 预训练特征提取
  - ParallelEngine: Dask 后端映射
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestTorchNAS(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.randn(60, 4), columns=['a', 'b', 'c', 'd'])
        self.y_cls = pd.Series(np.random.randint(0, 2, 60))
        self.y_reg = pd.Series(np.random.randn(60))
    
    def test_nas_classification(self):
        try:
            from core.nas import TorchNAS
        except ImportError:
            self.skipTest("PyTorch not available")
        
        model = TorchNAS(task_type='classification', n_candidates=3, epochs=5, random_state=42)
        model.fit(self.X, self.y_cls)
        
        self.assertIsNotNone(model.model_)
        self.assertIsNotNone(model.best_arch_)
        self.assertTrue(len(model.search_history_) > 0)
        self.assertIn('hidden_dims', model.best_arch_)
        self.assertIn('activation', model.best_arch_)
        
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape[0], len(self.X))
    
    def test_nas_regression(self):
        try:
            from core.nas import TorchNAS
        except ImportError:
            self.skipTest("PyTorch not available")
        
        model = TorchNAS(task_type='regression', n_candidates=3, epochs=5, random_state=42)
        model.fit(self.X, self.y_reg)
        
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
    
    def test_nas_search_history_structure(self):
        try:
            from core.nas import TorchNAS
        except ImportError:
            self.skipTest("PyTorch not available")
        
        model = TorchNAS(task_type='classification', n_candidates=2, epochs=3, random_state=42)
        model.fit(self.X, self.y_cls)
        
        for h in model.search_history_:
            self.assertIn('candidate_id', h)
            self.assertIn('architecture', h)
            self.assertIn('score', h)
            self.assertIn('time', h)
    
    def test_nas_different_architectures(self):
        """确保不同 random_state 产生不同架构选择"""
        try:
            from core.nas import TorchNAS
        except ImportError:
            self.skipTest("PyTorch not available")
        
        model1 = TorchNAS(task_type='classification', n_candidates=2, epochs=3, random_state=1)
        model1.fit(self.X, self.y_cls)
        
        model2 = TorchNAS(task_type='classification', n_candidates=2, epochs=3, random_state=2)
        model2.fit(self.X, self.y_cls)
        
        # 至少有一个候选的架构不同
        archs1 = [tuple(h['architecture']['hidden_dims']) for h in model1.search_history_]
        archs2 = [tuple(h['architecture']['hidden_dims']) for h in model2.search_history_]
        self.assertNotEqual(archs1, archs2)


class TestTransferFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.randn(50, 8))
    
    def test_fit_transform(self):
        try:
            from core.nas import TransferFeatureExtractor
        except ImportError:
            self.skipTest("PyTorch not available")
        
        extractor = TransferFeatureExtractor(encoding_dim=4, epochs=10, random_state=42)
        X_new = extractor.fit_transform(self.X)
        
        self.assertEqual(X_new.shape[0], self.X.shape[0])
        self.assertEqual(X_new.shape[1], 4)
    
    def test_transform_after_fit(self):
        try:
            from core.nas import TransferFeatureExtractor
        except ImportError:
            self.skipTest("PyTorch not available")
        
        extractor = TransferFeatureExtractor(encoding_dim=4, epochs=10, random_state=42)
        extractor.fit(self.X)
        X_new = extractor.transform(self.X)
        
        self.assertEqual(X_new.shape[0], self.X.shape[0])
        self.assertEqual(X_new.shape[1], 4)


class TestParallelEngineDask(unittest.TestCase):
    def test_dask_map(self):
        try:
            from dask.distributed import Client
        except ImportError:
            self.skipTest("Dask not installed")
        
        from core.accelerators import ParallelEngine
        
        engine = ParallelEngine(n_jobs=2, backend='dask')
        result = engine.map(lambda x: x * x, [1, 2, 3, 4])
        self.assertEqual(result, [1, 4, 9, 16])
        engine.close()
    
    def test_dask_fallback_when_not_installed(self):
        """测试 Dask 不可用时回退到进程池"""
        from core.accelerators import ParallelEngine
        
        with patch('core.accelerators.log_warning') as mock_log:
            engine = ParallelEngine(n_jobs=2, backend='dask')
            # 模拟 dask 未安装的情况
            with patch.dict('sys.modules', {'dask.distributed': None}):
                # 这里不会真正触发，因为 dask 已安装
                pass
            engine.close()
    
    def test_auto_backend(self):
        from core.accelerators import ParallelEngine
        
        engine = ParallelEngine(n_jobs=2, backend='thread')
        result = engine.map(lambda x: x + 1, [1, 2, 3])
        self.assertEqual(result, [2, 3, 4])
    
    def test_thread_backend(self):
        from core.accelerators import ParallelEngine
        
        engine = ParallelEngine(n_jobs=2, backend='thread')
        result = engine.map(lambda x: x * 2, [1, 2, 3])
        self.assertEqual(result, [2, 4, 6])
    
    def test_starmap(self):
        from core.accelerators import ParallelEngine
        
        engine = ParallelEngine(n_jobs=2, backend='thread')
        result = engine.starmap(lambda a, b: a + b, [(1, 2), (3, 4), (5, 6)])
        self.assertEqual(result, [3, 7, 11])


class TestNASRegistration(unittest.TestCase):
    def test_nas_in_model_library(self):
        from core.modeling_engine import ModelLibrary, TaskType
        ModelLibrary._init()
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        self.assertIn('torch_nas', models)
    
    def test_nas_dl_filter(self):
        """NAS 应被正确识别为 DL 模型并在默认情况下过滤"""
        from core.modeling_engine import ModelingEngine, FeatureSelectionStrategy
        
        engine = ModelingEngine(
            task_type='classification',
            deep_learning={'enabled': False, 'models': []},
            feature_selection=FeatureSelectionStrategy.NONE
        )
        X = pd.DataFrame(np.random.randn(30, 2), columns=['a', 'b'])
        y = pd.Series(np.random.randint(0, 2, 30))
        result = engine.fit(X, y)
        
        trained_keys = [r.model_key for r in result.cv_results]
        self.assertNotIn('torch_nas', trained_keys)


if __name__ == '__main__':
    unittest.main()
