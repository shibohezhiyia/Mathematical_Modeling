"""
Tests for core/config_manager.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from core.config_manager import ConfigManager, get_config


class TestConfigManagerDefault(unittest.TestCase):
    """默认配置相关测试"""

    def test_default_config_structure(self):
        cm = ConfigManager()
        self.assertIn("data", cm.config)
        self.assertIn("modeling", cm.config)
        self.assertIn("deep_learning", cm.config)
        self.assertIn("performance", cm.config)
        self.assertIn("cache", cm.config)

    def test_default_values(self):
        cm = ConfigManager()
        self.assertEqual(cm.get("data.encoding"), "utf-8")
        self.assertEqual(cm.get("data.chunk_threshold_mb"), 100)
        self.assertEqual(cm.get("modeling.n_splits"), 5)
        self.assertEqual(cm.get("modeling.optimizer"), "bayesian")
        self.assertEqual(cm.get("deep_learning.enabled"), False)
        self.assertEqual(cm.get("deep_learning.use_amp"), False)
        self.assertEqual(cm.get("performance.n_jobs"), -1)
        self.assertEqual(cm.get("performance.enable_kernel_approximation"), True)
        self.assertEqual(cm.get("performance.enable_precomputed_kernel_cache"), True)
        self.assertEqual(cm.get("cache.ttl_seconds"), 604800)


class TestConfigManagerLoadSave(unittest.TestCase):
    """加载与保存测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _sample_config(self):
        return {
            "data": {"encoding": "gbk", "chunk_threshold_mb": 50},
            "modeling": {"n_splits": 3, "optimizer": "random"},
            "custom": {"flag": True},
        }

    def test_load_save_yaml(self):
        path = os.path.join(self.tmpdir, "cfg.yaml")
        data = self._sample_config()
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        cm = ConfigManager().load(path)
        self.assertEqual(cm.get("data.encoding"), "gbk")
        self.assertEqual(cm.get("modeling.n_splits"), 3)
        self.assertEqual(cm.get("custom.flag"), True)

        cm.set("modeling.optimizer", "bayesian")
        saved = cm.save(path)
        self.assertEqual(saved, path)

        cm2 = ConfigManager().load(path)
        self.assertEqual(cm2.get("modeling.optimizer"), "bayesian")

    def test_load_save_json(self):
        path = os.path.join(self.tmpdir, "cfg.json")
        data = self._sample_config()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        cm = ConfigManager().load(path)
        self.assertEqual(cm.get("data.encoding"), "gbk")
        self.assertEqual(cm.get("modeling.n_splits"), 3)

        cm.set("data.chunk_threshold_mb", 200)
        saved = cm.save(path)
        self.assertEqual(saved, path)

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(raw["data"]["chunk_threshold_mb"], 200)

    def test_load_unsupported_format(self):
        path = os.path.join(self.tmpdir, "cfg.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("hello")
        with self.assertRaises(ValueError) as ctx:
            ConfigManager().load(path)
        self.assertIn("Unsupported config format", str(ctx.exception))

    def test_save_unsupported_format(self):
        cm = ConfigManager()
        path = os.path.join(self.tmpdir, "cfg.txt")
        with self.assertRaises(ValueError) as ctx:
            cm.save(path)
        self.assertIn("Unsupported config format", str(ctx.exception))

    def test_save_without_path(self):
        cm = ConfigManager()
        with self.assertRaises(ValueError):
            cm.save()


class TestConfigManagerDotPath(unittest.TestCase):
    """点号路径 get/set 测试"""

    def test_get_existing(self):
        cm = ConfigManager()
        self.assertEqual(cm.get("modeling.n_splits"), 5)

    def test_get_missing_with_default(self):
        cm = ConfigManager()
        self.assertIsNone(cm.get("modeling.nonexistent"))
        self.assertEqual(cm.get("modeling.nonexistent", "default"), "default")

    def test_get_top_level(self):
        cm = ConfigManager()
        self.assertIsInstance(cm.get("data"), dict)

    def test_set_existing(self):
        cm = ConfigManager()
        cm.set("modeling.n_splits", 10)
        self.assertEqual(cm.get("modeling.n_splits"), 10)

    def test_set_create_nested(self):
        cm = ConfigManager()
        cm.set("extra.level1.level2", 42)
        self.assertEqual(cm.get("extra.level1.level2"), 42)

    def test_set_overwrite_non_dict(self):
        cm = ConfigManager()
        cm.set("data.encoding", {"type": "auto"})
        self.assertEqual(cm.get("data.encoding"), {"type": "auto"})


class TestConfigManagerMerge(unittest.TestCase):
    """深度合并测试"""

    def test_merge_override_existing(self):
        cm = ConfigManager()
        cm.merge({"modeling": {"n_splits": 3}})
        self.assertEqual(cm.get("modeling.n_splits"), 3)

    def test_merge_add_new_keys(self):
        cm = ConfigManager()
        cm.merge({"modeling": {"new_key": "new_value"}})
        self.assertEqual(cm.get("modeling.new_key"), "new_value")
        self.assertEqual(cm.get("modeling.optimizer"), "bayesian")

    def test_merge_top_level(self):
        cm = ConfigManager()
        cm.merge({"custom_section": {"a": 1}})
        self.assertEqual(cm.get("custom_section.a"), 1)

    def test_merge_does_not_mutate_argument(self):
        cm = ConfigManager()
        other = {"modeling": {"n_splits": 3}}
        cm.merge(other)
        self.assertEqual(other["modeling"]["n_splits"], 3)  # unchanged


class TestConfigManagerValidation(unittest.TestCase):
    """配置验证测试"""

    def test_validate_default_pass(self):
        cm = ConfigManager()
        self.assertTrue(cm.validate())

    def test_validate_custom_schema(self):
        cm = ConfigManager()
        cm.config = {"name": "test", "value": 123}
        schema = {"name": str, "value": int}
        self.assertTrue(cm.validate(schema))

    def test_validate_missing_key(self):
        cm = ConfigManager()
        del cm.config["modeling"]
        with self.assertRaises(ValueError) as ctx:
            cm.validate()
        self.assertIn("缺少必需字段", str(ctx.exception))

    def test_validate_wrong_type(self):
        cm = ConfigManager()
        cm.config["modeling"]["n_splits"] = "five"
        with self.assertRaises(TypeError) as ctx:
            cm.validate()
        self.assertIn("类型错误", str(ctx.exception))

    def test_validate_root_not_dict(self):
        cm = ConfigManager()
        cm.config = [1, 2, 3]
        with self.assertRaises(ValueError):
            cm.validate()


class TestConfigManagerFromPipeline(unittest.TestCase):
    """从 Pipeline 导出配置测试"""

    def test_from_pipeline_mock(self):
        class FakePipeline:
            strategy_preference = "fast"
            target_col = "target"
            user_task_type = "classification"
            model_keys = ["lr", "xgb"]
            allow_disk_write = True
            encoding = "onehot"
            feature_selection = "rfe"
            ensemble = "voting_soft"
            n_splits = 3
            optimize_hyperparams = True
            hyperparam_trials = 30
            hyperparam_sampler = "random"
            explainability = True
            auto_decision_mode = "accuracy_first"
            user_override_model = "xgb"
            visualization = True
            auto_sample = False
            max_samples = 10000
            deep_learning = {"enabled": True, "models": ["torch_mlp", "tabnet"]}
            optimizer = "rl"
            dim_reduction = "pca"

        pipeline = FakePipeline()
        cm = ConfigManager.from_pipeline(pipeline)

        self.assertEqual(cm.get("modeling.n_splits"), 3)
        self.assertEqual(cm.get("modeling.optimizer"), "rl")
        self.assertEqual(cm.get("modeling.hyperparam_trials"), 30)
        self.assertEqual(cm.get("modeling.auto_sample"), False)
        self.assertEqual(cm.get("modeling.max_samples"), 10000)
        self.assertEqual(cm.get("modeling.feature_selection"), "rfe")
        self.assertEqual(cm.get("modeling.ensemble"), "voting_soft")
        self.assertEqual(cm.get("modeling.encoding"), "onehot")
        self.assertEqual(cm.get("deep_learning.enabled"), True)
        self.assertEqual(cm.get("deep_learning.use_amp"), False)
        self.assertEqual(cm.get("deep_learning.models"), ["torch_mlp", "tabnet"])
        self.assertEqual(cm.get("performance.strategy_preference"), "fast")
        self.assertEqual(cm.get("performance.enable_kernel_approximation"), True)
        self.assertEqual(cm.get("performance.enable_precomputed_kernel_cache"), True)
        self.assertEqual(cm.get("pipeline.target_col"), "target")
        self.assertEqual(cm.get("pipeline.task_type"), "classification")
        self.assertEqual(cm.get("pipeline.visualization"), True)
        self.assertEqual(cm.get("pipeline.dim_reduction"), "pca")

    def test_from_pipeline_no_deep_learning(self):
        class FakePipeline:
            strategy_preference = None
            target_col = None
            user_task_type = None
            model_keys = None
            allow_disk_write = True
            encoding = "auto"
            feature_selection = "mi"
            ensemble = "weighted"
            n_splits = 5
            optimize_hyperparams = False
            hyperparam_trials = 20
            hyperparam_sampler = "tpe"
            explainability = False
            auto_decision_mode = "balanced"
            user_override_model = None
            visualization = False
            auto_sample = True
            max_samples = 50000
            deep_learning = None
            optimizer = "bayesian"
            dim_reduction = "none"

        cm = ConfigManager.from_pipeline(FakePipeline())
        self.assertEqual(cm.get("deep_learning.enabled"), False)
        self.assertEqual(cm.get("deep_learning.use_amp"), False)
        self.assertEqual(cm.get("deep_learning.models"), ["torch_mlp"])


class TestConfigManagerSingleton(unittest.TestCase):
    """全局单例测试"""

    def tearDown(self):
        # 重置单例，避免影响其他测试
        import core.config_manager as cm_mod
        cm_mod._config_manager = None

    def test_singleton_same_instance(self):
        c1 = get_config()
        c2 = get_config()
        self.assertIs(c1, c2)

    def test_singleton_reload_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"custom": {"x": 1}}, f)
            c1 = get_config()
            c2 = get_config(path)
            self.assertIsNot(c1, c2)  # 提供路径时会创建新实例
            self.assertEqual(c2.get("custom.x"), 1)
            self.assertIs(get_config(), c2)  # 再次获取应返回最新实例


if __name__ == "__main__":
    unittest.main()
