"""
Unit tests for core/data_quality.py
"""
import unittest

import numpy as np
import pandas as pd

from core.data_quality import generate_data_quality_report


class TestDataQualityReport(unittest.TestCase):
    def test_classification_report(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [1, 1, 1, 2, 2, 2, 3, 3, 3, 100],  # outlier in b
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        report = generate_data_quality_report(df, target_col='target', task_type='classification')
        self.assertEqual(report['n_rows'], 10)
        self.assertEqual(report['n_columns'], 3)
        self.assertEqual(report['duplicates']['duplicate_rows'], 0)
        self.assertEqual(report['missing_values']['total_missing_cells'], 0)
        self.assertEqual(report['target']['n_classes'], 2)
        self.assertEqual(report['target']['imbalance_ratio'], 1.0)
        self.assertEqual(report['outliers']['columns_with_outliers'], 1)
        self.assertEqual(report['outliers']['details'][0]['column'], 'b')

    def test_regression_report(self):
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'y': np.random.randn(50)
        })
        report = generate_data_quality_report(df, target_col='y', task_type='regression')
        self.assertEqual(report['n_rows'], 50)
        self.assertEqual(report['target']['type'], 'regression')
        self.assertIn('mean', report['target'])
        self.assertIn('std', report['target'])

    def test_missing_values(self):
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [np.nan, np.nan, 3, 4, 5]
        })
        report = generate_data_quality_report(df)
        self.assertEqual(report['missing_values']['total_missing_cells'], 3)
        self.assertEqual(report['missing_values']['columns_with_missing'], 2)

    def test_duplicates(self):
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 5, 6]
        })
        report = generate_data_quality_report(df)
        self.assertEqual(report['duplicates']['duplicate_rows'], 1)

    def test_high_correlation(self):
        x = np.random.randn(100)
        df = pd.DataFrame({
            'a': x,
            'b': x * 2 + 0.001,  # nearly perfectly correlated
            'c': np.random.randn(100)
        })
        report = generate_data_quality_report(df)
        pairs = report['correlations']['high_correlation_pairs']
        self.assertTrue(any(p['col1'] in ('a', 'b') and p['col2'] in ('a', 'b') for p in pairs))

    def test_no_numeric(self):
        df = pd.DataFrame({
            'cat': ['a', 'b', 'c'],
            'target': ['x', 'y', 'z']
        })
        report = generate_data_quality_report(df)
        self.assertEqual(report['outliers']['columns_with_outliers'], 0)
        self.assertEqual(len(report['correlations']['high_correlation_pairs']), 0)

    def test_constant_columns(self):
        df = pd.DataFrame({
            'a': [1, 1, 1],
            'b': [1, 2, 3],
            'target': [0, 1, 0]
        })
        report = generate_data_quality_report(df, target_col='target')
        self.assertEqual(report['constant_columns']['count'], 1)
        self.assertEqual(report['constant_columns']['columns'][0]['column'], 'a')

    def test_high_cardinality(self):
        df = pd.DataFrame({
            'id': ['id1', 'id2', 'id3'],
            'cat': ['a', 'a', 'b'],
            'target': [0, 1, 0]
        })
        report = generate_data_quality_report(df, target_col='target')
        self.assertEqual(report['high_cardinality']['count'], 1)
        self.assertEqual(report['high_cardinality']['columns'][0]['column'], 'id')

    def test_target_leakage(self):
        df = pd.DataFrame({
            'leak': [1, 2, 3, 4],
            'target': [1, 2, 3, 4]
        })
        report = generate_data_quality_report(df, target_col='target')
        self.assertEqual(report['target_leakage']['count'], 1)
        self.assertEqual(report['target_leakage']['columns'][0]['column'], 'leak')


if __name__ == '__main__':
    unittest.main()
