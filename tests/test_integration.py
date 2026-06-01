"""
Comprehensive integration tests exercising multiple modules together
in realistic end-to-end scenarios.
"""
import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from core.data_module import DataModule
from core.integrated_pipeline import IntegratedPipeline
from core.auto_pipeline import AutoMissingPipeline, PipelineConfig
from core.sampling_engine import AutoSampler
from core.explainability import ExplainabilityEngine
from core.fairness import FairnessEngine, FAIRLEARN_AVAILABLE
from core.performance_scheduler import PerformanceScheduler, DataScaleEvaluator, StrategyLevel
from core.modeling_engine import ModelLibrary, TaskType, ModelingEngine

# Trigger registration of deep-learning / multimodal models
# (core.__init__ imports this automatically when any core submodule is loaded)
# ModelLibrary._init() is called lazily by get_models / create_model.


class TestEndToEndClassification(unittest.TestCase):
    """End-to-end classification workflow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "cls.csv")
        rng = np.random.RandomState(42)
        df_train = pd.DataFrame({
            "f0": rng.randn(80),
            "f1": rng.randn(80),
            "f2": rng.randn(80),
            "f3": rng.randn(80),
            "f4": rng.randn(80),
            "target": rng.randint(0, 2, 80),
        })
        df_test = pd.DataFrame({
            "f0": rng.randn(20),
            "f1": rng.randn(20),
            "f2": rng.randn(20),
            "f3": rng.randn(20),
            "f4": rng.randn(20),
            "target": np.nan,
        })
        df = pd.concat([df_train, df_test], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_pipeline(self):
        dm = DataModule()
        dm.load(self.csv_path).analyze().clean(target_col="target")
        self.assertIsNotNone(dm.cleaned_data)
        self.assertIn("target", dm.cleaned_data.columns)

        # Use the raw merged data for the pipeline so NaN targets form a test split
        raw_df = dm.raw_data.copy()

        pipeline = IntegratedPipeline(
            strategy_preference="fast",
            target_col="target",
            allow_disk_write=False,
            n_splits=3,
            model_keys=["lr", "dt"],
            ensemble="weighted",
        )
        result = pipeline.run(raw_df)

        self.assertIsNotNone(result.leaderboard)
        self.assertFalse(result.leaderboard.empty)
        self.assertIsNotNone(result.modeling_result)
        self.assertTrue(len(result.modeling_result.cv_results) > 0)
        self.assertIsNotNone(result.decision_report)

        # predictions exist because we provided a test split (NaN targets)
        self.assertIsNotNone(result.predictions)
        exported = pipeline.export_predictions("pred_cls.csv")
        # allow_disk_write=False -> None, but should not raise
        self.assertIsNone(exported)


class TestEndToEndRegression(unittest.TestCase):
    """End-to-end regression workflow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "reg.csv")
        rng = np.random.RandomState(42)
        df_train = pd.DataFrame({
            "f0": rng.randn(80),
            "f1": rng.randn(80),
            "f2": rng.randn(80),
            "f3": rng.randn(80),
            "f4": rng.randn(80),
            "target": rng.randn(80) * 10 + 5,
        })
        df_test = pd.DataFrame({
            "f0": rng.randn(20),
            "f1": rng.randn(20),
            "f2": rng.randn(20),
            "f3": rng.randn(20),
            "f4": rng.randn(20),
            "target": np.nan,
        })
        df = pd.concat([df_train, df_test], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_regression_pipeline(self):
        dm = DataModule()
        dm.load(self.csv_path).analyze().clean(target_col="target")

        pipeline = IntegratedPipeline(
            strategy_preference="standard",
            target_col="target",
            allow_disk_write=False,
            n_splits=3,
            model_keys=["linear", "ridge", "dt"],
            ensemble="best_single",
        )
        result = pipeline.run(dm.cleaned_data)

        self.assertIsNotNone(result.leaderboard)
        self.assertFalse(result.leaderboard.empty)
        # Regression metrics should be present
        cols = list(result.leaderboard.columns)
        self.assertTrue(any("rmse" in c for c in cols) or any("r2" in c for c in cols))
        self.assertEqual(result.task_type, "regression")


class TestEndToEndClustering(unittest.TestCase):
    """End-to-end clustering workflow."""

    def test_clustering_pipeline(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f0": rng.randn(100),
            "f1": rng.randn(100),
            "f2": rng.randn(100),
            "f3": rng.randn(100),
        })

        pipeline = IntegratedPipeline(
            task_type="clustering",
            allow_disk_write=False,
            model_keys=["kmeans", "gmm"],
            n_splits=3,
        )
        result = pipeline.run(df)

        self.assertEqual(result.task_type, "clustering")
        self.assertIsNotNone(result.leaderboard)
        self.assertFalse(result.leaderboard.empty)
        self.assertTrue(len(result.modeling_result.cv_results) > 0)
        # Clustering metrics should exist in at least one result
        scores = result.modeling_result.cv_results[0].mean_scores
        self.assertIn("silhouette", scores)


class TestEndToEndWithOptimization(unittest.TestCase):
    """Small dataset with hyperparameter optimization."""

    def test_optimization_history_populated(self):
        rng = np.random.RandomState(42)
        df_train = pd.DataFrame({
            "f0": rng.randn(40),
            "f1": rng.randn(40),
            "f2": rng.randn(40),
            "target": rng.randint(0, 2, 40),
        })
        df_test = pd.DataFrame({
            "f0": rng.randn(10),
            "f1": rng.randn(10),
            "f2": rng.randn(10),
            "target": np.nan,
        })
        df = pd.concat([df_train, df_test], ignore_index=True)

        pipeline = IntegratedPipeline(
            strategy_preference="fast",
            target_col="target",
            allow_disk_write=False,
            optimize_hyperparams=True,
            hyperparam_trials=5,
            model_keys=["lr"],
            n_splits=3,
            ensemble="best_single",
        )
        result = pipeline.run(df)

        self.assertIsNotNone(result.modeling_result)
        self.assertIsNotNone(result.modeling_result.optimization_history)
        self.assertTrue(len(result.modeling_result.optimization_history) > 0)
        # At least one model should have history entries
        histories = result.modeling_result.optimization_history
        self.assertTrue(any(len(v) > 0 for v in histories.values()))


class TestEndToEndWithMissingValues(unittest.TestCase):
    """Data with various missing patterns handled by AutoMissingPipeline."""

    def test_missing_pipeline(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f_num": rng.randn(100),
            "f_cat": rng.choice(["A", "B", "C"], 100),
            "target": rng.randint(0, 2, 100),
        })
        # MCAR: random missing
        mcar_idx = df.sample(10, random_state=42).index
        df.loc[mcar_idx, "f_num"] = np.nan
        # MAR: missing in f_cat when target == 0
        mar_idx = df[df["target"] == 0].sample(5, random_state=43).index
        df.loc[mar_idx, "f_cat"] = np.nan
        # MNAR: missing in target when f_num is high
        high_idx = df[df["f_num"] > df["f_num"].quantile(0.8)].sample(8, random_state=44).index
        df.loc[high_idx, "target"] = np.nan

        config = PipelineConfig(target_col="target", allow_disk_write=False)
        pipeline = AutoMissingPipeline(config)
        train_df, test_df, report = pipeline.run(df)

        self.assertIsNotNone(report)
        self.assertIsNotNone(train_df)
        self.assertGreater(len(report.column_profiles), 0)
        # Imputed data should have fewer NaNs than original (excluding target test split)
        self.assertTrue(train_df.isnull().sum().sum() <= df[df["target"].notna()].isnull().sum().sum())


class TestEndToEndWithSampling(unittest.TestCase):
    """Large dataset triggers auto-sampling."""

    def test_sampling_report(self):
        rng = np.random.RandomState(42)
        df_train = pd.DataFrame({
            "f0": rng.randn(4000),
            "f1": rng.randn(4000),
            "f2": rng.randn(4000),
            "target": rng.randint(0, 2, 4000),
        })
        df_test = pd.DataFrame({
            "f0": rng.randn(1000),
            "f1": rng.randn(1000),
            "f2": rng.randn(1000),
            "target": np.nan,
        })
        df = pd.concat([df_train, df_test], ignore_index=True)

        pipeline = IntegratedPipeline(
            strategy_preference="fast",
            target_col="target",
            allow_disk_write=False,
            auto_sample=True,
            max_samples=1000,
            n_splits=3,
            model_keys=["lr", "dt"],
            ensemble="best_single",
        )
        result = pipeline.run(df)

        self.assertIsNotNone(result.modeling_result)
        self.assertIsNotNone(result.modeling_result.sampling_report)
        report = result.modeling_result.sampling_report
        self.assertLess(report.sample_ratio, 1.0)
        self.assertEqual(report.sampled_n, 1000)


class TestEndToEndWithExplainer(unittest.TestCase):
    """Train a model and explain it."""

    def test_explain_model(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(60, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randint(0, 2, 60))
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        exp = engine.explain_model(model, X, y, model_key="rf", task_type="classification")

        self.assertIsNotNone(exp.global_importance)
        self.assertFalse(exp.global_importance.empty)
        self.assertIn("feature", exp.global_importance.columns)
        self.assertIn("importance", exp.global_importance.columns)


class TestEndToEndWithFairness(unittest.TestCase):
    """Fairness analysis on data with a sensitive attribute."""

    def test_fairness_report_structure(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "f0": rng.randn(60),
            "f1": rng.randn(60),
            "gender": rng.choice(["M", "F"], 60),
        })
        y = pd.Series(rng.randint(0, 2, 60))
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X.drop(columns=["gender"]), y)

        engine = FairnessEngine()
        report = engine.analyze(
            model, X, y,
            sensitive_attr="gender",
            task_type="classification"
        )

        if FAIRLEARN_AVAILABLE:
            self.assertIsNotNone(report)
            self.assertEqual(report.sensitive_attr, "gender")
            self.assertIn("group_metrics", report.__dict__)
            self.assertTrue(len(report.group_metrics) > 0)
        else:
            # Graceful fallback when fairlearn not installed
            self.assertIsNone(report)


class TestEndToEndMultimodal(unittest.TestCase):
    """Mock image/text data and verify multimodal model initialization."""

    def test_image_resnet_init(self):
        from core.multimodal import ImageResNet, TORCHVISION_AVAILABLE
        if not TORCHVISION_AVAILABLE:
            self.skipTest("torchvision not available")

        # Mock the backbone download/build to avoid network I/O
        with patch("core.multimodal.models.resnet18") as mock_resnet:
            mock_backbone = MagicMock()
            mock_backbone.fc = MagicMock()
            mock_backbone.fc.in_features = 512
            mock_resnet.return_value = mock_backbone
            model = ImageResNet(
                task_type="classification",
                image_col="image_path",
                epochs=1,
                freeze_backbone=True,
                random_state=42,
            )
            self.assertEqual(model.image_col, "image_path")
            self.assertEqual(model.task_type, "classification")

    def test_text_bert_init(self):
        from core.multimodal import TextBERT, TRANSFORMERS_AVAILABLE
        if not TRANSFORMERS_AVAILABLE:
            self.skipTest("transformers not available")

        with patch("core.multimodal.DistilBertModel.from_pretrained") as mock_bert, \
             patch("core.multimodal.DistilBertTokenizer.from_pretrained") as mock_tok:
            mock_bert.return_value = MagicMock()
            mock_tok.return_value = MagicMock()
            model = TextBERT(
                task_type="classification",
                text_col="text",
                epochs=1,
                freeze_backbone=True,
                random_state=42,
            )
            self.assertEqual(model.text_col, "text")
            self.assertEqual(model.task_type, "classification")

    def test_model_library_creation(self):
        # Ensure ModelLibrary knows about multimodal keys
        # (registration happens in deep_learning.py imported via core.__init__)
        ModelLibrary._init()
        cls_models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        self.assertIn("image_resnet", cls_models)
        self.assertIn("text_bert", cls_models)

        reg_models = ModelLibrary.get_models(TaskType.REGRESSION)
        self.assertIn("image_resnet", reg_models)
        self.assertIn("text_bert", reg_models)


class TestEndToEndPerformanceScheduler(unittest.TestCase):
    """PerformanceScheduler with different data sizes."""

    def test_small_data_plan(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(100, 5))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        self.assertIn(plan.strategy, [StrategyLevel.STANDARD, StrategyLevel.FAST])
        self.assertGreaterEqual(plan.n_jobs, 1)
        self.assertIsNotNone(plan.reason)

    def test_large_data_plan(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(500_000, 20))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        # Large data should trigger FAST or ULTRA
        self.assertIn(plan.strategy, [StrategyLevel.FAST, StrategyLevel.ULTRA])
        self.assertIsNotNone(plan.sample_size)
        self.assertIsNotNone(plan.missing_sample_size)

    def test_user_preference_override(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(100, 5))
        scheduler = PerformanceScheduler(user_preference=StrategyLevel.ULTRA)
        plan = scheduler.schedule(df)
        self.assertEqual(plan.strategy, StrategyLevel.ULTRA)


if __name__ == "__main__":
    unittest.main()
