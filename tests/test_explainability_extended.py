"""
Extended unit tests for core/explainability.py
Maximizes line coverage for ExplainabilityEngine and edge cases.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.explainability import ExplainabilityEngine, ExplanationResult
from core.modeling_engine import TaskType


class TestExplainabilityEngineExtended(unittest.TestCase):
    """Extended tests for ExplainabilityEngine"""

    def _make_data(self, n=50, n_features=5):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(n, n_features), columns=[f'f{i}' for i in range(n_features)])
        y = pd.Series(np.random.choice([0, 1], n))
        return X, y

    def test_explain_model_tree_shap(self):
        """Test explain_model with tree model triggers TreeSHAP"""
        from sklearn.tree import DecisionTreeClassifier
        X, y = self._make_data()
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(X, y)

        mock_shap_vals = np.random.randn(len(X), X.shape[1])

        with patch('core.explainability.shap') as mock_shap_mod:
            mock_shap_mod.sample.return_value = X.iloc[:10]
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = mock_shap_vals
            mock_shap_mod.TreeExplainer.return_value = mock_explainer

            engine = ExplainabilityEngine(use_shap=True, use_lime=False)
            result = engine.explain_model(model, X, y, model_key='dt', task_type=TaskType.CLASSIFICATION)

            self.assertEqual(result.method, 'tree_shap')
            self.assertIsNotNone(result.shap_values)
            self.assertIsNotNone(result.global_importance)
            mock_shap_mod.TreeExplainer.assert_called_once_with(model)

    def test_explain_model_kernel_shap_fallback(self):
        """Test explain_model falls back to KernelSHAP for non-tree models"""
        from sklearn.neighbors import KNeighborsClassifier
        X, y = self._make_data()
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        mock_shap_vals = np.random.randn(len(X), X.shape[1])

        with patch('core.explainability.shap') as mock_shap_mod:
            mock_shap_mod.sample.return_value = X.iloc[:10]
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = mock_shap_vals
            mock_shap_mod.KernelExplainer.return_value = mock_explainer

            engine = ExplainabilityEngine(use_shap=True, use_lime=False)
            result = engine.explain_model(model, X, y, model_key='knn', task_type=TaskType.CLASSIFICATION)

            self.assertEqual(result.method, 'kernel_shap')
            self.assertIsNotNone(result.shap_values)
            mock_shap_mod.KernelExplainer.assert_called_once()

    def test_explain_instance_lime(self):
        """Test explain_instance uses LIME when available"""
        from sklearn.tree import DecisionTreeClassifier
        X, y = self._make_data()
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(X, y)

        mock_lime_exp = MagicMock()
        mock_lime_exp.as_list.return_value = [('f0', 0.5), ('f1', -0.2)]
        mock_lime_exp.intercept = 0.1
        mock_lime_exp.score = 0.95

        mock_lime_explainer = MagicMock()
        mock_lime_explainer.explain_instance.return_value = mock_lime_exp

        with patch('core.explainability.LimeTabularExplainer', return_value=mock_lime_explainer) as mock_lime_cls:
            engine = ExplainabilityEngine(use_shap=False, use_lime=True)
            exp = engine.explain_instance(model, X, instance_index=0, task_type=TaskType.CLASSIFICATION)

            self.assertIn('lime', exp)
            self.assertEqual(exp['lime']['score'], 0.95)
            mock_lime_cls.assert_called_once()

    def test_explain_instance_kernel_shap(self):
        """Test explain_instance extracts KernelSHAP values for a single instance"""
        from sklearn.neighbors import KNeighborsClassifier
        X, y = self._make_data()
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        mock_shap_vals = np.random.randn(len(X), X.shape[1])

        with patch('core.explainability.shap') as mock_shap_mod:
            mock_shap_mod.sample.return_value = X.iloc[:10]
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = mock_shap_vals
            mock_shap_mod.KernelExplainer.return_value = mock_explainer

            engine = ExplainabilityEngine(use_shap=True, use_lime=False)
            exp = engine.explain_instance(model, X, instance_index=0, model_key='knn', task_type=TaskType.CLASSIFICATION)

            self.assertIn('top_positive', exp)
            self.assertIn('top_negative', exp)
            self.assertTrue(all(isinstance(t, tuple) for t in exp['top_positive']))

    def test_explain_multiple_instances(self):
        """Test explain_multiple_instances returns a list of explanations"""
        from sklearn.tree import DecisionTreeClassifier
        X, y = self._make_data(n=30, n_features=3)
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(X, y)

        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        results = engine.explain_multiple_instances(model, X, indices=[0, 1, 2], task_type=TaskType.CLASSIFICATION)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn('prediction', r)
            self.assertIn('features', r)

    def test_generate_report(self):
        """Test generate_report returns formatted text"""
        result = ExplanationResult(
            model_key='test_model',
            method='builtin',
            global_importance=pd.DataFrame({
                'feature': ['a', 'b'],
                'importance': [0.8, 0.2]
            })
        )
        engine = ExplainabilityEngine()
        report = engine.generate_report(result)
        self.assertIn('test_model', report)
        self.assertIn('解释方法: builtin', report)

    def test_generate_report_with_output_path(self):
        """Test generate_report writes to disk via workspace manager"""
        mock_wm = MagicMock()
        mock_wm.write_text.return_value = os.path.join('workspace', 'reports', 'report.txt')

        result = ExplanationResult(
            model_key='test_model',
            method='builtin',
            global_importance=pd.DataFrame({
                'feature': ['a'],
                'importance': [1.0]
            })
        )
        with patch('core.explainability.get_workspace_manager', return_value=mock_wm):
            engine = ExplainabilityEngine()
            report = engine.generate_report(result, output_path='report.txt')
            self.assertIn('test_model', report)
            mock_wm.write_text.assert_called_once()

    def test_explain_model_shap_raises(self):
        """Test explain_model gracefully handles SHAP computation failure"""
        from sklearn.tree import DecisionTreeClassifier
        X, y = self._make_data()
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(X, y)

        engine = ExplainabilityEngine(use_shap=True, use_lime=False)
        with patch.object(engine, '_compute_shap', side_effect=Exception("SHAP error")):
            result = engine.explain_model(model, X, y, model_key='dt', task_type=TaskType.CLASSIFICATION)
            self.assertIsNotNone(result.global_importance)
            self.assertEqual(result.method, 'builtin')

    def test_compute_shap_no_explainer(self):
        """Test _compute_shap returns None when no explainer can be created"""
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))

        X = pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c'])
        engine = ExplainabilityEngine(use_shap=True, use_lime=False)
        with patch('core.explainability.shap') as mock_shap_mod:
            mock_shap_mod.sample.return_value = X.iloc[:5]
            mock_shap_mod.KernelExplainer.side_effect = Exception("init fail")
            result = engine._compute_shap(DummyModel(), X, ['a', 'b', 'c'], TaskType.CLASSIFICATION, 'dummy')
            self.assertIsNone(result)


class TestExplainabilityEdgeCases(unittest.TestCase):
    """Edge case tests for explainability"""

    def test_empty_dataset(self):
        """Test explain_model with empty dataset (0 rows)"""
        from sklearn.tree import DecisionTreeClassifier
        # Train on real data
        X_train = pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c'])
        y_train = pd.Series(np.random.choice([0, 1], 10))
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Explain empty DataFrame with same columns
        X_empty = pd.DataFrame(columns=['a', 'b', 'c'])
        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        result = engine.explain_model(model, X_empty, model_key='dt', task_type=TaskType.CLASSIFICATION)
        self.assertIsNotNone(result.global_importance)
        self.assertEqual(len(result.global_importance), 3)

    def test_single_feature(self):
        """Test explain_model with a single feature"""
        from sklearn.tree import DecisionTreeClassifier
        X = pd.DataFrame({'a': np.random.randn(50)})
        y = pd.Series(np.random.choice([0, 1], 50))
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        result = engine.explain_model(model, X, model_key='dt', task_type=TaskType.CLASSIFICATION)
        self.assertEqual(len(result.global_importance), 1)
        self.assertEqual(result.global_importance.iloc[0]['feature'], 'a')

    def test_model_without_importance_or_coef(self):
        """Test model without feature_importances_ or coef_ returns None from builtin"""
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))

        X = pd.DataFrame(np.random.randn(30, 3), columns=['a', 'b', 'c'])
        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        result = engine.explain_model(DummyModel(), X, y=None, model_key='dummy', task_type=TaskType.CLASSIFICATION)
        self.assertIsNone(result.global_importance)
        self.assertEqual(result.method, 'builtin')

    def test_get_builtin_importance_none(self):
        """Test _get_builtin_importance directly for models with no attributes"""
        class Dummy:
            pass
        engine = ExplainabilityEngine()
        importance = engine._get_builtin_importance(Dummy(), ['a', 'b'])
        self.assertIsNone(importance)

    def test_explain_instance_lime_raises(self):
        """Test explain_instance handles LIME failure gracefully"""
        from sklearn.tree import DecisionTreeClassifier
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.choice([0, 1], 30))
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        engine = ExplainabilityEngine(use_shap=False, use_lime=True)
        with patch.object(engine, '_lime_explain', side_effect=Exception("LIME error")):
            exp = engine.explain_instance(model, X, 0, task_type=TaskType.CLASSIFICATION)
            self.assertNotIn('lime', exp)


if __name__ == '__main__':
    unittest.main()
