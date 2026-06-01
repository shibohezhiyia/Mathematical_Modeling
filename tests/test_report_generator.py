"""
Unit tests for core/report_generator.py
"""
import os
import tempfile
import unittest

from core.report_generator import generate_html_report, generate_word_report


class TestReportGenerator(unittest.TestCase):
    def test_html_report(self):
        html = generate_html_report(
            modeling_result={'accuracy': 0.92, 'leaderboard': [{'model': 'xgb', 'score': 0.92}]},
            data_info={'shape': [100, 5], 'columns': ['a', 'b', 'c', 'd', 'target']},
        )
        self.assertIn('Modeling Report', html)
        self.assertIn('Data Summary', html)
        self.assertIn('xgb', html)

    def test_word_report_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'report.docx')
            result = generate_word_report({'accuracy': 0.9}, output_path=path)
            self.assertTrue(os.path.exists(result))


if __name__ == '__main__':
    unittest.main()
