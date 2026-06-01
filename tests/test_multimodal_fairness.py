"""
多模态 + 公平性测试

覆盖:
  - ImageResNet: 图像分类 fit/predict
  - TextBERT: 文本分类 fit/predict
  - FairnessEngine: 公平性指标计算、敏感属性检测、约束训练
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestImageResNet(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'feature_a': [1, 2, 3],
            'image_path': ['fake1.jpg', 'fake2.jpg', 'fake3.jpg'],
        })
        self.y = pd.Series([0, 1, 0])
    
    def test_image_model_initialization(self):
        try:
            from core.multimodal import ImageResNet
        except ImportError:
            self.skipTest("torchvision not available")
        
        model = ImageResNet(task_type='classification', image_col='image_path', epochs=1)
        self.assertEqual(model.image_col, 'image_path')
        self.assertEqual(model.task_type, 'classification')
    
    def test_image_model_missing_column(self):
        try:
            from core.multimodal import ImageResNet
        except ImportError:
            self.skipTest("torchvision not available")
        
        model = ImageResNet(task_type='classification', image_col='missing_col', epochs=1)
        df_bad = pd.DataFrame({'a': [1, 2, 3]})
        with self.assertRaises(ValueError):
            model.fit(df_bad, self.y)


class TestTextBERT(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'feature_a': [1, 2, 3],
            'text': ['good product', 'bad service', 'excellent quality'],
        })
        self.y = pd.Series([1, 0, 1])
    
    def test_text_model_initialization(self):
        try:
            from core.multimodal import TextBERT
        except ImportError:
            self.skipTest("transformers not available")
        
        model = TextBERT(task_type='classification', text_col='text', epochs=1)
        self.assertEqual(model.text_col, 'text')
        self.assertEqual(model.task_type, 'classification')
    
    def test_text_model_fit_predict(self):
        try:
            from core.multimodal import TextBERT
        except ImportError:
            self.skipTest("transformers not available")
        
        model = TextBERT(task_type='classification', text_col='text', epochs=1, freeze_backbone=True)
        model.fit(self.df, self.y)
        
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))
        
        proba = model.predict_proba(self.df)
        self.assertEqual(proba.shape[0], len(self.df))


class TestFairnessEngine(unittest.TestCase):
    def setUp(self):
        from sklearn.linear_model import LogisticRegression
        self.X = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50],
            'income': [3000, 4000, 5000, 6000, 7000, 8000],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        })
        self.y = pd.Series([0, 1, 0, 1, 1, 0])
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X[['age', 'income']], self.y)
    
    def test_detect_sensitive_attributes(self):
        try:
            from core.fairness import FairnessEngine
        except ImportError:
            self.skipTest("Fairlearn not available")
        
        engine = FairnessEngine()
        attrs = engine.detect_sensitive_attributes(self.X)
        self.assertIn('gender', attrs)
    
    def test_analyze_classification(self):
        try:
            from core.fairness import FairnessEngine
        except ImportError:
            self.skipTest("Fairlearn not available")
        
        engine = FairnessEngine(fairness_threshold=0.1)
        report = engine.analyze(
            self.model, self.X, self.y,
            sensitive_attr='gender',
            task_type='classification'
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(report.sensitive_attr, 'gender')
        self.assertIsNotNone(report.demographic_parity_diff)
        self.assertIsNotNone(report.equalized_odds_diff)
        self.assertIn('group_metrics', dir(report))
        self.assertTrue(len(report.group_metrics) > 0)
    
    def test_analyze_auto_detect(self):
        try:
            from core.fairness import FairnessEngine
        except ImportError:
            self.skipTest("Fairlearn not available")
        
        engine = FairnessEngine()
        report = engine.analyze(
            self.model, self.X, self.y,
            sensitive_attr=None,
            task_type='classification'
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(report.sensitive_attr, 'gender')
    
    def test_fairness_report_structure(self):
        try:
            from core.fairness import FairnessEngine
        except ImportError:
            self.skipTest("Fairlearn not available")
        
        engine = FairnessEngine()
        report = engine.analyze(self.model, self.X, self.y, sensitive_attr='gender', task_type='classification')
        
        self.assertTrue(hasattr(report, 'is_fair'))
        self.assertTrue(hasattr(report, 'recommendations'))
        self.assertTrue(hasattr(report, 'analysis_time'))
        self.assertGreaterEqual(report.analysis_time, 0)
    
    def test_group_metrics_content(self):
        try:
            from core.fairness import FairnessEngine
        except ImportError:
            self.skipTest("Fairlearn not available")
        
        engine = FairnessEngine()
        report = engine.analyze(self.model, self.X, self.y, sensitive_attr='gender', task_type='classification')
        
        for group, metrics in report.group_metrics.items():
            self.assertIn('count', metrics)
            self.assertGreaterEqual(metrics['count'], 1)


class TestMultimodalRegistration(unittest.TestCase):
    def test_models_in_library(self):
        from core.modeling_engine import ModelLibrary, TaskType
        ModelLibrary._init()
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        
        # 多模态模型应已注册
        self.assertIn('image_resnet', models)
        self.assertIn('text_bert', models)


if __name__ == '__main__':
    unittest.main()
