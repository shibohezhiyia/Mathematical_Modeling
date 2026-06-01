"""
Unit tests for core/incremental_learning.py
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from core.incremental_learning import partial_fit_model, supports_incremental


class TestIncrementalLearning(unittest.TestCase):
    def test_supports_incremental(self):
        self.assertTrue(supports_incremental('sgd'))
        self.assertFalse(supports_incremental('xgb'))

    def test_partial_fit_sgd(self):
        model = SGDClassifier(max_iter=5, random_state=42)
        X1 = pd.DataFrame(np.random.randn(20, 3))
        y1 = pd.Series(np.random.randint(0, 2, 20))
        model.partial_fit(X1, y1, classes=[0, 1])
        X2 = pd.DataFrame(np.random.randn(10, 3))
        y2 = pd.Series(np.random.randint(0, 2, 10))
        updated = partial_fit_model(model, X2, y2)
        self.assertIs(updated, model)

    def test_partial_fit_no_support(self):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=3)
        X = pd.DataFrame(np.random.randn(10, 2))
        y = pd.Series(np.random.randint(0, 2, 10))
        with self.assertRaises(ValueError):
            partial_fit_model(model, X, y)


if __name__ == '__main__':
    unittest.main()
