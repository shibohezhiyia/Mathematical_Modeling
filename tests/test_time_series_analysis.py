"""
Unit tests for core/time_series_analysis.py
"""
import unittest

import numpy as np
import pandas as pd

from core.time_series_analysis import analyze_time_series


class TestTimeSeriesAnalysis(unittest.TestCase):
    def test_basic_analysis(self):
        t = pd.date_range('2024-01-01', periods=100, freq='D')
        y = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
        series = pd.Series(y, index=t)
        result = analyze_time_series(series, freq='D')
        self.assertIn('n_obs', result)
        self.assertIn('adf_pvalue', result)
        self.assertIn('acf', result)


if __name__ == '__main__':
    unittest.main()
