"""
模型库扩展测试
"""
import unittest
import numpy as np
import pandas as pd

from core.modeling_engine import ModelLibrary, TaskType, ModelSpec
from core.parallel_modeling import ModelRegistry, ModelConfig


class TestModelLibraryNewModels(unittest.TestCase):
    """测试 ModelLibrary 新模型"""
    
    @classmethod
    def setUpClass(cls):
        ModelLibrary._init()
    
    def test_classification_hist_gb_registered(self):
        """分类 HistGradientBoosting 已注册"""
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        self.assertIn('hist_gb', models)
        self.assertEqual(models['hist_gb'].name, 'HistGradientBoosting')
        self.assertEqual(models['hist_gb'].category, 'ensemble')
    
    def test_classification_sgd_registered(self):
        """分类 SGD 已注册"""
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        self.assertIn('sgd', models)
        self.assertEqual(models['sgd'].name, 'SGD')
        self.assertTrue(models['sgd'].supports_partial_fit)
    
    def test_regression_hist_gb_registered(self):
        """回归 HistGradientBoosting 已注册"""
        models = ModelLibrary.get_models(TaskType.REGRESSION)
        self.assertIn('hist_gb', models)
    
    def test_regression_sgd_registered(self):
        """回归 SGD 已注册"""
        models = ModelLibrary.get_models(TaskType.REGRESSION)
        self.assertIn('sgd', models)
        self.assertTrue(models['sgd'].supports_partial_fit)
    
    def test_regression_ransac_registered(self):
        """回归 RANSAC 已注册"""
        models = ModelLibrary.get_models(TaskType.REGRESSION)
        self.assertIn('ransac', models)
        self.assertEqual(models['ransac'].name, 'RANSAC')
    
    def test_clustering_optics_registered(self):
        """聚类 OPTICS 已注册"""
        models = ModelLibrary.get_models(TaskType.CLUSTERING)
        self.assertIn('optics', models)
        self.assertEqual(models['optics'].name, 'OPTICS')
    
    def test_clustering_minibatch_kmeans_registered(self):
        """聚类 MiniBatchKMeans 已注册"""
        models = ModelLibrary.get_models(TaskType.CLUSTERING)
        self.assertIn('minibatch_kmeans', models)
        self.assertEqual(models['minibatch_kmeans'].name, 'MiniBatchKMeans')
    
    def test_clustering_birch_registered(self):
        """聚类 Birch 已注册"""
        models = ModelLibrary.get_models(TaskType.CLUSTERING)
        self.assertIn('birch', models)
        self.assertEqual(models['birch'].name, 'Birch')
    
    def test_new_models_can_instantiate(self):
        """新模型可以实例化"""
        # 分类
        for key in ['hist_gb', 'sgd']:
            with self.subTest(task='classification', model=key):
                model = ModelLibrary.create_model(key, TaskType.CLASSIFICATION)
                self.assertIsNotNone(model)
        # 回归
        for key in ['hist_gb', 'sgd', 'ransac']:
            with self.subTest(task='regression', model=key):
                model = ModelLibrary.create_model(key, TaskType.REGRESSION)
                self.assertIsNotNone(model)
        # 聚类
        for key in ['optics', 'minibatch_kmeans', 'birch']:
            with self.subTest(task='clustering', model=key):
                model = ModelLibrary.create_model(key, TaskType.CLUSTERING)
                self.assertIsNotNone(model)
    
    def test_model_spec_capabilities(self):
        """模型规格能力标签"""
        spec = ModelSpec(
            name='Test', key='test', model_class=None, category='linear',
            supports_gpu=True, supports_partial_fit=True,
            supports_sample_weight=True, is_probabilistic=True
        )
        self.assertTrue(spec.supports_gpu)
        self.assertTrue(spec.supports_partial_fit)
        self.assertTrue(spec.supports_sample_weight)
        self.assertTrue(spec.is_probabilistic)
    
    def test_classification_model_count(self):
        """分类模型数量 >= 15（原有14 + 新2，catboost 可能未安装）"""
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        # 排除深度学习模型
        sklearn_models = {k: v for k, v in models.items() if not k.startswith('torch_') and k not in ('image_resnet', 'text_bert')}
        self.assertGreaterEqual(len(sklearn_models), 15)
    
    def test_regression_model_count(self):
        """回归模型数量 >= 22（原有20 + 新3，catboost 可能未安装）"""
        models = ModelLibrary.get_models(TaskType.REGRESSION)
        sklearn_models = {k: v for k, v in models.items() if not k.startswith('torch_') and k not in ('image_resnet', 'text_bert')}
        self.assertGreaterEqual(len(sklearn_models), 22)
    
    def test_clustering_model_count(self):
        """聚类模型数量 >= 8（原有5 + 新3）"""
        models = ModelLibrary.get_models(TaskType.CLUSTERING)
        self.assertGreaterEqual(len(models), 8)


class TestModelLibraryFitPredict(unittest.TestCase):
    """测试新模型可以 fit/predict"""
    
    @classmethod
    def setUpClass(cls):
        ModelLibrary._init()
        np.random.seed(42)
        cls.X_clf = pd.DataFrame(np.random.randn(50, 4), columns=['a', 'b', 'c', 'd'])
        cls.y_clf = pd.Series(np.random.randint(0, 2, 50))
        cls.X_reg = pd.DataFrame(np.random.randn(50, 4))
        cls.y_reg = pd.Series(np.random.randn(50))
    
    def test_hist_gb_classification(self):
        """HistGradientBoosting 分类"""
        model = ModelLibrary.create_model('hist_gb', TaskType.CLASSIFICATION)
        model.fit(self.X_clf, self.y_clf)
        pred = model.predict(self.X_clf)
        self.assertEqual(len(pred), 50)
    
    def test_hist_gb_regression(self):
        """HistGradientBoosting 回归"""
        model = ModelLibrary.create_model('hist_gb', TaskType.REGRESSION)
        model.fit(self.X_reg, self.y_reg)
        pred = model.predict(self.X_reg)
        self.assertEqual(len(pred), 50)
    
    def test_sgd_classification(self):
        """SGD 分类"""
        model = ModelLibrary.create_model('sgd', TaskType.CLASSIFICATION)
        model.fit(self.X_clf, self.y_clf)
        pred = model.predict(self.X_clf)
        self.assertEqual(len(pred), 50)
    
    def test_sgd_regression(self):
        """SGD 回归"""
        model = ModelLibrary.create_model('sgd', TaskType.REGRESSION)
        model.fit(self.X_reg, self.y_reg)
        pred = model.predict(self.X_reg)
        self.assertEqual(len(pred), 50)
    
    def test_ransac_regression(self):
        """RANSAC 回归"""
        model = ModelLibrary.create_model('ransac', TaskType.REGRESSION)
        model.fit(self.X_reg, self.y_reg)
        pred = model.predict(self.X_reg)
        self.assertEqual(len(pred), 50)
    
    def test_optics_clustering(self):
        """OPTICS 聚类"""
        model = ModelLibrary.create_model('optics', TaskType.CLUSTERING)
        labels = model.fit_predict(self.X_reg)
        self.assertEqual(len(labels), 50)
    
    def test_minibatch_kmeans_clustering(self):
        """MiniBatchKMeans 聚类"""
        model = ModelLibrary.create_model('minibatch_kmeans', TaskType.CLUSTERING)
        labels = model.fit_predict(self.X_reg)
        self.assertEqual(len(labels), 50)
    
    def test_birch_clustering(self):
        """Birch 聚类"""
        model = ModelLibrary.create_model('birch', TaskType.CLUSTERING)
        labels = model.fit_predict(self.X_reg)
        self.assertEqual(len(labels), 50)


class TestModelRegistryNewModels(unittest.TestCase):
    """测试 ModelRegistry 新模型"""
    
    @classmethod
    def setUpClass(cls):
        ModelRegistry._init()
    
    def test_hist_gb_in_registry(self):
        """ModelRegistry 包含 hist_gb"""
        models = ModelRegistry.get_available_models('classification')
        self.assertIn('hist_gb', models)
    
    def test_sgd_in_registry(self):
        """ModelRegistry 包含 sgd"""
        models = ModelRegistry.get_available_models('classification')
        self.assertIn('sgd', models)
        self.assertTrue(models['sgd'].supports_partial_fit)
    
    def test_model_config_capabilities(self):
        """ModelConfig 能力标签"""
        config = ModelConfig(
            name='Test', model_class=None,
            supports_gpu=True, supports_partial_fit=True,
            supports_sample_weight=True, is_probabilistic=True
        )
        self.assertTrue(config.supports_partial_fit)
        self.assertTrue(config.supports_sample_weight)
        self.assertTrue(config.is_probabilistic)


if __name__ == '__main__':
    unittest.main()
