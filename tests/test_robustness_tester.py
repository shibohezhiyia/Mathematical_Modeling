"""
Unit tests for core/robustness_tester.py
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from core.robustness_tester import evaluate_robustness


class TestRobustnessTester(unittest.TestCase):
    def test_classification_robustness(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randint(0, 2, 60))
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = evaluate_robustness(model, X, y, task_type='classification')
        self.assertIn('baseline', result)
        self.assertTrue(len(result['tests']) >= 6)
        self.assertIn('robustness_score', result)
        self.assertTrue(0 <= result['robustness_score'] <= 100)
        self.assertTrue(len(result['feature_sensitivity']) > 0)


if __name__ == '__main__':
    unittest.main()
