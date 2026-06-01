"""
Unit tests for core/shap_explainer.py
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from core.shap_explainer import explain_model


class TestSHAPExplainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame({
            'a': np.random.randn(50),
            'b': np.random.randn(50),
            'c': np.random.randn(50)
        })
        self.y = pd.Series(np.random.randint(0, 2, 50))

    def test_tree_explainer(self):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        result = explain_model(model, self.X, task_type='classification', sample_size=30)
        self.assertEqual(len(result['feature_importance']), 3)
        self.assertTrue(all('feature' in f and 'importance' in f for f in result['feature_importance']))
        self.assertEqual(len(result['instance_explanations']), 5)
        self.assertTrue(all(len(inst['top_features']) <= 8 for inst in result['instance_explanations']))

    def test_linear_explainer(self):
        model = LogisticRegression(max_iter=200)
        model.fit(self.X, self.y)
        result = explain_model(model, self.X, task_type='classification', sample_size=30)
        self.assertEqual(len(result['feature_importance']), 3)
        self.assertEqual(len(result['instance_explanations']), 5)

    def test_feature_importance_sorted(self):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        result = explain_model(model, self.X, sample_size=30)
        importances = [f['importance'] for f in result['feature_importance']]
        self.assertEqual(importances, sorted(importances, reverse=True))


if __name__ == '__main__':
    unittest.main()
