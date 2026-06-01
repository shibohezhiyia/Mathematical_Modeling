"""
Unit tests for core/integrated_pipeline.py
Uses mocks for heavy operations to keep tests fast.
"""
import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.integrated_pipeline import IntegratedPipeline, PipelineResult, quick_run
from core.modeling_engine import ModelingResult, TaskType, CVResult
from core.performance_scheduler import ExecutionPlan, StrategyLevel


class MockMissingReport:
    pass


def make_mock_plan(strategy='fast'):
    """Create a mock ExecutionPlan"""
    strategy_map = {
        'fast': StrategyLevel.FAST,
        'standard': StrategyLevel.STANDARD,
        'ultra': StrategyLevel.ULTRA,
    }
    return ExecutionPlan(
        strategy=strategy_map.get(strategy, StrategyLevel.FAST),
        n_jobs=2,
        use_gpu=False,
        missing_sample_size=500,
        missing_structural_threshold=0.9,
        cv_folds=3,
        max_models=2,
        hyperparameter_trials=0
    )


def make_mock_modeling_result(task_type_val='classification', with_predictions=True):
    """Create a mock ModelingResult"""
    task_type = TaskType(task_type_val)
    cv_result = CVResult(
        model_key='lr',
        model_name='LogisticRegression',
        mean_scores={'accuracy': 0.9, 'f1_weighted': 0.88},
        std_scores={'accuracy': 0.02, 'f1_weighted': 0.03},
        oof_pred=np.array([0, 1, 0]),
        fitted_models=[MagicMock()],
        train_time=0.5
    )
    ensemble_result = None
    if with_predictions:
        ensemble_result = {
            'test': np.array([0.6, 0.4, 0.7, 0.3]),
            'weights': {'lr': 1.0}
        }

    decision_report = MagicMock()
    decision_report.recommended_model = 'lr'
    decision_report.recommended_name = 'LogisticRegression'
    decision_report.confidence = 0.9

    return ModelingResult(
        task_type=task_type,
        cv_results=[cv_result],
        ensemble_result=ensemble_result,
        best_model_key='lr',
        best_cv_result=cv_result,
        leaderboard=pd.DataFrame({
            'rank': [1],
            'model': ['LogisticRegression'],
            'key': ['lr'],
            'train_time': [0.5],
            'accuracy_mean': [0.9],
            'accuracy_std': [0.02]
        }),
        decision_report=decision_report,
        preprocessing_info={
            'original_features': 5,
            'encoded_features': 7,
            'selected_features': 5
        },
        encoding_report=pd.DataFrame({
            'column': ['cat'],
            'strategy': ['onehot'],
            'encoder_type': ['OneHotEncoder']
        }),
        train_time=1.0
    )


class TestIntegratedPipelineStrategies(unittest.TestCase):
    """Test IntegratedPipeline with different strategies"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    def _make_classification_df(self, n=100):
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'f3': np.random.choice(['A', 'B', 'C'], n),
            'target': np.nan
        })
        n_train = int(n * 0.75)
        df.iloc[:n_train, df.columns.get_loc('target')] = np.random.choice([0, 1], n_train)
        return df

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_strategy_fast(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test with fast strategy"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        mock_missing.run.return_value = (
            pd.DataFrame({'f1': [1.0], 'target': [0]}),
            pd.DataFrame({'f1': [1.0]}),
            MockMissingReport()
        )
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = self._make_classification_df(80)
        pipeline = IntegratedPipeline(strategy_preference='fast', target_col='target')
        result = pipeline.run(df)

        self.assertEqual(result.strategy, 'fast')
        self.assertIsNotNone(result.predictions)
        self.assertIsNotNone(result.leaderboard)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_strategy_standard(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test with standard strategy"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('standard')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        mock_missing.run.return_value = (
            pd.DataFrame({'f1': [1.0], 'target': [0]}),
            pd.DataFrame({'f1': [1.0]}),
            MockMissingReport()
        )
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = self._make_classification_df(80)
        pipeline = IntegratedPipeline(strategy_preference='standard', target_col='target')
        result = pipeline.run(df)

        self.assertEqual(result.strategy, 'standard')

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_accuracy_first_mode(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test with accuracy_first decision mode"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('standard')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        mock_missing.run.return_value = (
            pd.DataFrame({'f1': [1.0], 'target': [0]}),
            pd.DataFrame({'f1': [1.0]}),
            MockMissingReport()
        )
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = self._make_classification_df(80)
        pipeline = IntegratedPipeline(
            target_col='target',
            auto_decision_mode='accuracy_first'
        )
        result = pipeline.run(df)
        self.assertEqual(result.task_type, 'classification')


class TestIntegratedPipelineTaskTypes(unittest.TestCase):
    """Test IntegratedPipeline with different task types"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_binary_classification(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test binary classification"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_result = make_mock_modeling_result('classification')
        mock_result.ensemble_result = {
            'test': np.array([0.6]),
            'weights': {'lr': 1.0}
        }
        mock_engine.fit.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'target': np.nan
        })
        df.iloc[:76, df.columns.get_loc('target')] = np.random.choice([0, 1], 76)

        pipeline = IntegratedPipeline(target_col='target')
        result = pipeline.run(df)

        self.assertIsNotNone(result.train_df)
        self.assertIsNotNone(result.test_df)
        self.assertIsNotNone(result.predictions)
        self.assertEqual(len(result.predictions), 1)  # test_df has 1 row in mock

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_multiclass_classification(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test multi-class classification"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'target': [0, 1, 2]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_result = make_mock_modeling_result('classification')
        mock_engine.fit.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        np.random.seed(42)
        n = 120
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'target': np.nan
        })
        df.loc[:90, 'target'] = np.random.choice([0, 1, 2], 91)

        pipeline = IntegratedPipeline(target_col='target')
        result = pipeline.run(df)
        self.assertEqual(result.task_type, 'classification')

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_regression(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test regression task"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0, 2.0], 'target': [1.5, 2.5]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        reg_result = make_mock_modeling_result('regression')
        mock_engine.fit.return_value = reg_result
        mock_engine_cls.return_value = mock_engine

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'target': np.nan
        })
        df.loc[:75, 'target'] = np.random.randn(76)

        pipeline = IntegratedPipeline(target_col='target', task_type='regression')
        result = pipeline.run(df)
        self.assertEqual(result.task_type, 'regression')

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_clustering(self, mock_engine_cls, mock_scheduler_cls):
        """Test clustering mode (no target)"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_engine = MagicMock()
        clust_result = make_mock_modeling_result('clustering')
        mock_engine.fit.return_value = clust_result
        mock_engine_cls.return_value = mock_engine

        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(80),
            'f2': np.random.randn(80),
        })

        pipeline = IntegratedPipeline(task_type='clustering')
        result = pipeline.run(df)

        self.assertIsNone(result.target_col)
        self.assertEqual(result.task_type, 'clustering')
        self.assertIsNone(result.test_df)


class TestIntegratedPipelineFeatures(unittest.TestCase):
    """Test specific IntegratedPipeline features"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_auto_target_detection(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test automatic target column detection"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0, 2.0], 'target_col': [0, 1]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'target_col': np.nan
        })
        # Set missing rate around 25% so it should be detected as target
        df.loc[:75, 'target_col'] = np.random.choice([0, 1], 76)

        pipeline = IntegratedPipeline()  # No target_col specified
        result = pipeline.run(df)
        self.assertEqual(result.target_col, 'target_col')

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_user_override_model(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test user_override_model parameter"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_result = make_mock_modeling_result('classification')
        mock_engine.fit.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        pipeline = IntegratedPipeline(
            target_col='target',
            user_override_model='xgb'
        )
        result = pipeline.run(df)
        self.assertIsNotNone(result.predictions)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_deep_learning_config(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test deep_learning configuration"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        pipeline = IntegratedPipeline(
            target_col='target',
            deep_learning={'enabled': True, 'models': ['torch_mlp']}
        )
        result = pipeline.run(df)
        self.assertIsNotNone(result.predictions)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_with_test_set(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test pipeline when test set is present"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]})
        test_df = pd.DataFrame({'f1': [3.0, 4.0]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_result = make_mock_modeling_result('classification')
        mock_result.ensemble_result = {
            'test': np.array([0.2, 0.8]),
            'weights': {'lr': 1.0}
        }
        mock_engine.fit.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0],
            'target': [0, 1, np.nan, np.nan]
        })

        pipeline = IntegratedPipeline(target_col='target')
        result = pipeline.run(df)
        self.assertIsNotNone(result.test_df)
        self.assertEqual(len(result.predictions), 2)

    def test_print_summary(self):
        """Test print_summary method"""
        pipeline = IntegratedPipeline(target_col='target')
        pipeline.result = PipelineResult()
        pipeline.result.strategy = 'fast'
        pipeline.result.task_type = 'classification'
        pipeline.result.target_col = 'target'
        pipeline.result.total_time = 1.5
        pipeline.result.execution_plan = make_mock_plan('fast')
        pipeline.result.train_df = pd.DataFrame({'a': [1, 2]})
        pipeline.result.test_df = pd.DataFrame({'a': [3]})
        pipeline.result.leaderboard = pd.DataFrame({'model': ['lr']})
        pipeline.result.ensemble_weights = {'lr': 1.0}
        pipeline.result.feature_importance = pd.DataFrame({
            'feature': ['f1'],
            'importance': [0.5]
        })
        pipeline.result.modeling_result = make_mock_modeling_result('classification')

        with patch('builtins.print') as mock_print:
            pipeline.print_summary()
            self.assertTrue(mock_print.called)

    def test_print_summary_no_result(self):
        """Test print_summary with minimal result"""
        pipeline = IntegratedPipeline(target_col='target')
        pipeline.result = PipelineResult()
        pipeline.result.strategy = 'fast'
        pipeline.result.task_type = 'classification'
        pipeline.result.target_col = None
        pipeline.result.total_time = 0.5
        pipeline.result.execution_plan = None
        pipeline.result.train_df = None
        pipeline.result.test_df = None
        pipeline.result.leaderboard = None
        pipeline.result.ensemble_weights = None
        pipeline.result.feature_importance = None
        pipeline.result.modeling_result = None

        with patch('builtins.print') as mock_print:
            pipeline.print_summary()
            self.assertTrue(mock_print.called)

    @patch('core.integrated_pipeline.get_workspace_manager')
    def test_export_predictions(self, mock_get_wm):
        """Test export_predictions"""
        mock_wm = MagicMock()
        mock_wm.save_dataframe.return_value = '/path/to/predictions.csv'
        mock_get_wm.return_value = mock_wm

        pipeline = IntegratedPipeline(target_col='target')
        pipeline.result = PipelineResult()
        pipeline.result.predictions = np.array([0, 1, 0])
        pipeline.result.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'f1': [1.0, 2.0, 3.0]
        })
        pipeline.task_type = 'classification'

        path = pipeline.export_predictions('preds.csv', id_col='id')
        self.assertEqual(path, '/path/to/predictions.csv')
        mock_wm.save_dataframe.assert_called_once()

    @patch('core.integrated_pipeline.get_workspace_manager')
    def test_export_predictions_no_id_col(self, mock_get_wm):
        """Test export_predictions without id_col"""
        mock_wm = MagicMock()
        mock_wm.save_dataframe.return_value = '/path/to/predictions.csv'
        mock_get_wm.return_value = mock_wm

        pipeline = IntegratedPipeline(target_col='target')
        pipeline.result = PipelineResult()
        pipeline.result.predictions = np.array([0.5, 0.6, 0.7])
        pipeline.result.test_df = pd.DataFrame({'f1': [1.0, 2.0, 3.0]})
        pipeline.task_type = 'regression'

        path = pipeline.export_predictions('preds.csv')
        self.assertEqual(path, '/path/to/predictions.csv')

    def test_export_predictions_no_predictions(self):
        """Test export_predictions raises when no predictions"""
        pipeline = IntegratedPipeline(target_col='target')
        pipeline.result = PipelineResult()
        pipeline.result.predictions = None

        with self.assertRaises(ValueError):
            pipeline.export_predictions('preds.csv')

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_no_target_all_train(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test when all rows have target values (no test set)"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0, 2.0], 'target': [0, 1]})
        mock_missing.run.return_value = (train_df, None, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_result = make_mock_modeling_result('classification', with_predictions=False)
        mock_engine.fit.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })

        pipeline = IntegratedPipeline(target_col='target')
        result = pipeline.run(df)
        self.assertIsNone(result.test_df)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_encoding_strategies(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test different encoding strategy strings"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        for enc in ['onehot', 'label', 'target', 'none']:
            pipeline = IntegratedPipeline(target_col='target', encoding=enc)
            result = pipeline.run(df)
            self.assertIsNotNone(result)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_feature_selection_strategies(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test different feature selection strategy strings"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        for fs in ['variance', 'rfe', 'model_based', 'correlation', 'pca', 'none']:
            pipeline = IntegratedPipeline(target_col='target', feature_selection=fs)
            result = pipeline.run(df)
            self.assertIsNotNone(result)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_ensemble_strategies(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test different ensemble strategy strings"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        for ens in ['voting_hard', 'voting_soft', 'stacking', 'best_single']:
            pipeline = IntegratedPipeline(target_col='target', ensemble=ens)
            result = pipeline.run(df)
            self.assertIsNotNone(result)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_visualization_true(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test visualization enabled path"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        # Mock visualization imports
        with patch.dict('sys.modules', {'core.visualization': MagicMock()}):
            pipeline = IntegratedPipeline(target_col='target', visualization=True)
            result = pipeline.run(df)
            self.assertIsNotNone(result)

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_automl_recommendation(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test optimizer='auto' triggers AutoML path"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        # Mock automl modules - they'll fail and be caught
        pipeline = IntegratedPipeline(target_col='target', optimizer='auto')
        result = pipeline.run(df)
        self.assertIsNotNone(result)


class TestQuickRun(unittest.TestCase):
    """Test quick_run convenience function"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    @patch('core.integrated_pipeline.PerformanceScheduler')
    @patch('core.integrated_pipeline.AutoMissingPipeline')
    @patch('core.integrated_pipeline.ModelingEngine')
    def test_quick_run(self, mock_engine_cls, mock_missing_cls, mock_scheduler_cls):
        """Test quick_run function"""
        mock_scheduler = MagicMock()
        mock_scheduler.schedule.return_value = make_mock_plan('fast')
        mock_scheduler_cls.return_value = mock_scheduler

        mock_missing = MagicMock()
        train_df = pd.DataFrame({'f1': [1.0], 'target': [0]})
        test_df = pd.DataFrame({'f1': [1.5]})
        mock_missing.run.return_value = (train_df, test_df, MockMissingReport())
        mock_missing_cls.return_value = mock_missing

        mock_engine = MagicMock()
        mock_engine.fit.return_value = make_mock_modeling_result('classification')
        mock_engine_cls.return_value = mock_engine

        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'target': [0, 1, np.nan]
        })

        with patch('builtins.print'):
            result = quick_run(df, target_col='target')
            self.assertIsNotNone(result.predictions)


if __name__ == '__main__':
    unittest.main()
