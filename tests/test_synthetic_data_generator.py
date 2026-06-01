"""
Unit tests for core/synthetic_data_generator.py
"""
import unittest

from core.synthetic_data_generator import generate_from_description, generate_synthetic_data


class TestSyntheticDataGenerator(unittest.TestCase):
    def test_classification(self):
        df = generate_synthetic_data('classification', n_samples=100, n_features=5, n_classes=2)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 6)
        self.assertIn('target', df.columns)
        self.assertEqual(df['target'].nunique(), 2)

    def test_regression(self):
        df = generate_synthetic_data('regression', n_samples=50, n_features=3)
        self.assertEqual(len(df), 50)
        self.assertIn('target', df.columns)
        self.assertTrue(df['target'].dtype.kind in 'iufc')

    def test_clustering(self):
        df = generate_synthetic_data('clustering', n_samples=80, n_features=4, n_classes=3)
        self.assertEqual(len(df), 80)
        self.assertIn('cluster', df.columns)
        self.assertEqual(df['cluster'].nunique(), 3)

    def test_time_series(self):
        df = generate_synthetic_data('time_series', n_samples=200, n_features=3)
        self.assertEqual(len(df), 200)
        self.assertIn('timestamp', df.columns)
        self.assertIn('target', df.columns)

    def test_from_description(self):
        df = generate_from_description('regression with 200 samples and 5 features')
        self.assertEqual(len(df), 200)
        self.assertEqual(len(df.columns), 6)


if __name__ == '__main__':
    unittest.main()
