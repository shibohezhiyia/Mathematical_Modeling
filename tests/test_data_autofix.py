"""
Unit tests for core/data_autofix.py
"""
import unittest

import numpy as np
import pandas as pd

from core.data_autofix import autofix_dataframe
from core.data_quality import generate_data_quality_report


class TestDataAutofix(unittest.TestCase):
    def test_drop_duplicates(self):
        df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [4, 5, 5, 6]})
        fixed, fixes = autofix_dataframe(df, drop_duplicates=True)
        self.assertEqual(len(fixed), 3)
        self.assertTrue(any('duplicate' in f for f in fixes))

    def test_fill_missing_numeric(self):
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0], 'target': [0, 1, 0, 1]})
        report = generate_data_quality_report(df, target_col='target')
        fixed, fixes = autofix_dataframe(df, report=report, target_col='target')
        self.assertEqual(fixed['a'].isnull().sum(), 0)
        self.assertTrue(any('median' in f for f in fixes))

    def test_fill_missing_categorical(self):
        df = pd.DataFrame({'cat': ['a', 'b', None, 'a'], 'target': [0, 1, 0, 1]})
        report = generate_data_quality_report(df, target_col='target')
        fixed, fixes = autofix_dataframe(df, report=report, target_col='target')
        self.assertEqual(fixed['cat'].isnull().sum(), 0)
        self.assertTrue(any('mode' in f for f in fixes))

    def test_drop_high_missing(self):
        df = pd.DataFrame({
            'a': [1, np.nan, np.nan, np.nan],  # 75% missing
            'b': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        report = generate_data_quality_report(df, target_col='target')
        fixed, fixes = autofix_dataframe(df, report=report, target_col='target', missing_threshold=50.0)
        self.assertNotIn('a', fixed.columns)
        self.assertTrue(any('Dropped' in f for f in fixes))

    def test_clip_outliers(self):
        x = list(range(1, 16)) + [200]  # 15 normal values + 1 outlier = 16
        y = [0] * 8 + [1] * 8
        df = pd.DataFrame({'x': x, 'target': y})
        report = generate_data_quality_report(df, target_col='target')
        fixed, fixes = autofix_dataframe(df, report=report, target_col='target', fix_outliers=True)
        self.assertLess(fixed['x'].max(), 200)  # outlier should be clipped
        self.assertTrue(any('outlier' in f.lower() or 'clip' in f.lower() for f in fixes))


if __name__ == '__main__':
    unittest.main()
