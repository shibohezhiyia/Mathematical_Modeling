"""
覆盖率提升测试 - 集中覆盖多个模块的边缘情况
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataModuleExtended(unittest.TestCase):
    """提升 data_module.py 覆盖率"""
    
    def setUp(self):
        from core.data_module import DataLoader, DataModule
        from core.workspace_manager import get_workspace_manager
        self.loader = DataLoader()
        wm = get_workspace_manager()
        wm.set_allow_disk_write(True)
        self.test_dir = wm.create_temp_dir(prefix='test_cov')
    
    def test_load_multiple_files(self):
        from core.data_module import DataLoader
        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'a': [3, 4]})
        p1 = os.path.join(self.test_dir, 'f1.csv')
        p2 = os.path.join(self.test_dir, 'f2.csv')
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        result = self.loader.load_multiple([p1, p2])
        self.assertEqual(len(result), 4)
    
    def test_load_json(self):
        import json
        from core.data_module import DataLoader
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        path = os.path.join(self.test_dir, 'test.json')
        with open(path, 'w') as f:
            json.dump(data, f)
        result = self.loader.load(path)
        self.assertEqual(len(result), 2)
    
    def test_load_parquet(self):
        from core.data_module import DataLoader
        df = pd.DataFrame({'a': [1, 2, 3]})
        path = os.path.join(self.test_dir, 'test.parquet')
        df.to_parquet(path)
        result = self.loader.load(path)
        self.assertEqual(len(result), 3)
    
    def test_data_module_full_pipeline(self):
        from core.data_module import DataModule
        df = pd.DataFrame({
            'num': [1.0, 2.0, np.nan, 4.0],
            'cat': ['A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1]
        })
        path = os.path.join(self.test_dir, 'data.csv')
        df.to_csv(path, index=False)
        dm = DataModule()
        dm.load(path)
        dm.analyze()
        dm.clean(target_col='target')
        summary = dm.get_summary()
        self.assertIn('total_columns', summary)
    
    def test_type_detector_datetime(self):
        from core.data_module import TypeDetector, DataType
        detector = TypeDetector()
        s = pd.Series(pd.date_range('2020-01-01', periods=10))
        dtype, profile = detector.detect(s, 'dt_col')
        self.assertEqual(dtype, DataType.DATETIME)
    
    def test_type_detector_text(self):
        from core.data_module import TypeDetector, DataType
        detector = TypeDetector(text_length_threshold=30)
        s = pd.Series([f'This is a long text entry number {i} with sufficient length to exceed the threshold.' for i in range(10)])
        dtype, profile = detector.detect(s, 'txt_col')
        self.assertEqual(dtype, DataType.TEXT)
    
    def test_data_cleaner_outliers(self):
        from core.data_module import DataCleaner
        cleaner = DataCleaner()
        df = pd.DataFrame({'a': [1, 2, 3, 1000]})
        result = cleaner.clean(df)
        self.assertEqual(len(result), 4)


class TestHyperparameterOptimizerExtended(unittest.TestCase):
    """提升 hyperparameter_optimizer.py 覆盖率"""
    
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(40, 4))
        self.y = pd.Series(np.random.randint(0, 2, 40))
    
    def test_bayesian_with_cmaes(self):
        try:
            import cmaes
        except ImportError:
            self.skipTest('cmaes package not installed')
        from core.hyperparameter_optimizer import BayesianOptimizer
        opt = BayesianOptimizer(n_trials=3, sampler='cmaes')
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result.best_params)
    
    def test_bayesian_with_random(self):
        from core.hyperparameter_optimizer import BayesianOptimizer
        opt = BayesianOptimizer(n_trials=3, sampler='random')
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result.best_params)
    
    def test_bayesian_timeout(self):
        from core.hyperparameter_optimizer import BayesianOptimizer
        opt = BayesianOptimizer(n_trials=10, timeout=1)
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result)
    
    def test_bayesian_pruner_median(self):
        from core.hyperparameter_optimizer import BayesianOptimizer
        opt = BayesianOptimizer(n_trials=3, pruner='median')
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result)
    
    def test_bayesian_regression(self):
        from core.hyperparameter_optimizer import BayesianOptimizer
        y_reg = pd.Series(np.random.randn(40))
        opt = BayesianOptimizer(n_trials=3)
        result = opt.optimize('ridge', self.X, y_reg, 'regression')
        self.assertIsNotNone(result.best_params)
    
    def test_bayesian_multiclass(self):
        from core.hyperparameter_optimizer import BayesianOptimizer
        y_multi = pd.Series(np.random.randint(0, 3, 40))
        opt = BayesianOptimizer(n_trials=3)
        result = opt.optimize('lr', self.X, y_multi, 'classification')
        self.assertIsNotNone(result)


class TestDeepLearningExtended(unittest.TestCase):
    """提升 deep_learning.py 覆盖率"""
    
    def setUp(self):
        np.random.seed(42)
        self.X_cls = pd.DataFrame(np.random.randn(30, 4))
        self.y_cls = pd.Series(np.random.randint(0, 2, 30))
        self.X_reg = pd.DataFrame(np.random.randn(30, 4))
        self.y_reg = pd.Series(np.random.randn(30))
    
    def test_torch_mlp_regression(self):
        from core.deep_learning import TorchMLP
        model = TorchMLP(task_type='regression', epochs=2, batch_size=10)
        model.fit(self.X_reg, self.y_reg)
        preds = model.predict(self.X_reg)
        self.assertEqual(len(preds), 30)
    
    def test_torch_cnn1d_classification(self):
        from core.deep_learning import TorchCNN1D
        model = TorchCNN1D(task_type='classification', epochs=2, batch_size=10)
        model.fit(self.X_cls, self.y_cls)
        preds = model.predict(self.X_cls)
        self.assertEqual(len(preds), 30)
    
    def test_torch_lstm_regression(self):
        from core.deep_learning import TorchLSTM
        model = TorchLSTM(task_type='regression', epochs=2, batch_size=10)
        model.fit(self.X_reg, self.y_reg)
        preds = model.predict(self.X_reg)
        self.assertEqual(len(preds), 30)
    
    def test_torch_gru_classification(self):
        from core.deep_learning import TorchGRU
        model = TorchGRU(task_type='classification', epochs=2, batch_size=10)
        model.fit(self.X_cls, self.y_cls)
        preds = model.predict(self.X_cls)
        self.assertEqual(len(preds), 30)
    
    def test_autoencoder_transform(self):
        from core.deep_learning import TorchAutoEncoder
        model = TorchAutoEncoder(encoding_dim=2, epochs=2)
        model.fit(self.X_cls)
        encoded = model.transform(self.X_cls)
        self.assertEqual(encoded.shape[1], 2)
    
    def test_autoencoder_fit_transform(self):
        from core.deep_learning import TorchAutoEncoder
        model = TorchAutoEncoder(encoding_dim=2, epochs=2)
        encoded = model.fit_transform(self.X_cls)
        self.assertEqual(encoded.shape[1], 2)


class TestModelingEngineMore(unittest.TestCase):
    """进一步提升 modeling_engine.py 覆盖率"""
    
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'a': np.random.randn(50),
            'b': np.random.randn(50),
            'c': np.random.choice(['X', 'Y'], 50),
            'target': np.random.randint(0, 2, 50)
        })
    
    def test_modeling_with_best_single_ensemble(self):
        from core.modeling_engine import ModelingEngine, EnsembleMethod
        engine = ModelingEngine(ensemble=EnsembleMethod.BEST_SINGLE, n_splits=3, auto_sample=False)
        result = engine.fit(self.df.drop(columns=['target']), self.df['target'])
        self.assertIsNotNone(result.leaderboard)
    
    def test_modeling_with_voting_soft(self):
        from core.modeling_engine import ModelingEngine, EnsembleMethod
        engine = ModelingEngine(ensemble=EnsembleMethod.VOTING_SOFT, n_splits=3, auto_sample=False)
        result = engine.fit(self.df.drop(columns=['target']), self.df['target'])
        self.assertIsNotNone(result.leaderboard)
    
    def test_modeling_auto_task_detection(self):
        from core.modeling_engine import ModelingEngine
        engine = ModelingEngine(n_splits=3, auto_sample=False)
        result = engine.fit(self.df.drop(columns=['target']), self.df['target'])
        self.assertIsNotNone(result.leaderboard)
    
    def test_modeling_with_dim_reduction(self):
        from core.modeling_engine import ModelingEngine
        engine = ModelingEngine(dim_reduction='pca', n_splits=3, auto_sample=False)
        result = engine.fit(self.df.drop(columns=['target']), self.df['target'])
        self.assertIsNotNone(result.leaderboard)


class TestAcceleratorsMore(unittest.TestCase):
    """进一步提升 accelerators.py 覆盖率"""
    
    def test_optimize_memory_unsigned_int(self):
        from core.accelerators import optimize_memory
        df = pd.DataFrame({'a': [0, 1, 2, 3]})
        result = optimize_memory(df, verbose=False)
        self.assertEqual(result['a'].dtype, np.uint8)
    
    def test_optimize_memory_signed_int(self):
        from core.accelerators import optimize_memory
        df = pd.DataFrame({'a': [-1, 0, 1, 2]})
        result = optimize_memory(df, verbose=False)
        self.assertEqual(result['a'].dtype, np.int8)
    
    def test_optimize_memory_no_reduction(self):
        from core.accelerators import optimize_memory
        df = pd.DataFrame({'a': [0, 1000, 2000, 3000]})
        result = optimize_memory(df, verbose=False)
        self.assertTrue(pd.api.types.is_integer_dtype(result['a']))


class TestEvaluationEngineExtended(unittest.TestCase):
    """提升 evaluation_engine.py 覆盖率"""
    
    def test_auto_decision_balanced(self):
        from core.evaluation_engine import AutoDecisionEngine, DecisionMode, ModelScore
        engine = AutoDecisionEngine(mode=DecisionMode.BALANCED)
        scores = [
            ModelScore(model_key='model_a', model_name='Model A', accuracy_score=80, speed_score=50, stability_score=60, simplicity_score=70, generalization_score=65),
            ModelScore(model_key='model_b', model_name='Model B', accuracy_score=70, speed_score=90, stability_score=60, simplicity_score=70, generalization_score=65),
        ]
        decision = engine.decide(scores)
        self.assertIsNotNone(decision.recommended_model)
    
    def test_auto_decision_empty(self):
        from core.evaluation_engine import AutoDecisionEngine, DecisionMode
        engine = AutoDecisionEngine(mode=DecisionMode.BALANCED)
        decision = engine.decide([])
        self.assertEqual(decision.recommended_model, "")


if __name__ == '__main__':
    unittest.main()
