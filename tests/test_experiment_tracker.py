"""
Unit tests for core/experiment_tracker.py
"""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import core.experiment_tracker as et


class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        # Use a temporary DB for isolation
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test_experiments.db')
        self._orig_db_path = et.DB_PATH
        et.DB_PATH = self.db_path
        et._init_db()

    def tearDown(self):
        et.DB_PATH = self._orig_db_path
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.tmpdir)

    def test_log_and_list_experiments(self):
        config = {'model_keys': ['lr', 'xgb'], 'ensemble': 'stacking', 'feature_engineering': True}
        result = {
            'task_type': 'classification',
            'leaderboard': [
                {'model_name': 'xgb', 'accuracy_mean': 0.92}
            ]
        }
        exp_id = et.log_experiment(config, result, duration=12.5, dataset_name='iris')
        self.assertIsInstance(exp_id, int)
        self.assertGreater(exp_id, 0)

        rows = et.list_experiments(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['task_type'], 'classification')
        self.assertEqual(rows[0]['best_model'], 'xgb')
        self.assertAlmostEqual(rows[0]['best_score'], 0.92, places=4)
        self.assertEqual(rows[0]['duration'], 12.5)
        self.assertEqual(rows[0]['dataset_name'], 'iris')

    def test_list_with_task_filter(self):
        et.log_experiment({}, {'task_type': 'classification', 'leaderboard': []}, 1.0)
        et.log_experiment({}, {'task_type': 'regression', 'leaderboard': []}, 2.0)
        cls_rows = et.list_experiments(task_type='classification')
        self.assertEqual(len(cls_rows), 1)
        self.assertEqual(cls_rows[0]['task_type'], 'classification')

    def test_get_experiment(self):
        config = {'optimize_hyperparams': True, 'hyperparam_trials': 20}
        result = {'task_type': 'regression', 'leaderboard': [{'model_key': 'ridge', 'rmse_mean': 0.3}]}
        exp_id = et.log_experiment(config, result, 5.0)
        exp = et.get_experiment(exp_id)
        self.assertIsNotNone(exp)
        self.assertEqual(exp['id'], exp_id)
        self.assertEqual(exp['config']['hyperparam_trials'], 20)
        self.assertEqual(exp['result']['task_type'], 'regression')

    def test_get_nonexistent_experiment(self):
        self.assertIsNone(et.get_experiment(99999))

    def test_compare_experiments(self):
        e1 = et.log_experiment({'ensemble': 'voting'}, {'task_type': 'classification', 'leaderboard': [{'model_name': 'lr', 'accuracy_mean': 0.8}]}, 3.0)
        e2 = et.log_experiment({'ensemble': 'stacking'}, {'task_type': 'classification', 'leaderboard': [{'model_name': 'xgb', 'accuracy_mean': 0.9}]}, 6.0)
        comp = et.compare_experiments([e1, e2])
        self.assertEqual(len(comp), 2)
        ids = [c['id'] for c in comp]
        self.assertIn(e1, ids)
        self.assertIn(e2, ids)

    def test_delete_experiment(self):
        exp_id = et.log_experiment({}, {'task_type': 'clustering', 'leaderboard': []}, 1.0)
        self.assertTrue(et.delete_experiment(exp_id))
        self.assertIsNone(et.get_experiment(exp_id))
        self.assertFalse(et.delete_experiment(exp_id))

    def test_summarize_config(self):
        s = et._summarize_config({'model_keys': ['a', 'b'], 'ensemble': 'stacking', 'optimize_hyperparams': True, 'hyperparam_trials': 30, 'feature_engineering': True, 'pseudo_labeling': True})
        self.assertIn('models=2', s)
        self.assertIn('ens=stacking', s)
        self.assertIn('hpo=30', s)
        self.assertIn('fe=on', s)
        self.assertIn('pl=on', s)

    def test_empty_leaderboard(self):
        exp_id = et.log_experiment({}, {'task_type': 'clustering', 'leaderboard': []}, 2.0)
        row = et.list_experiments(limit=1)[0]
        self.assertEqual(row['best_score'], 0.0)
        self.assertEqual(row['best_model'], '')


if __name__ == '__main__':
    unittest.main()
