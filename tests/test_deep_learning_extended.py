"""
Extended unit tests for core/deep_learning.py

覆盖:
  - TorchMLP (分类/回归, 早停, sklearn clone)
  - TorchCNN1D (1D序列/表格数据)
  - TorchLSTM (序列reshape)
  - TorchGRU (小数据)
  - TorchAutoEncoder (fit/transform/predict)
  - TorchNAS (架构搜索)
  - TransferFeatureExtractor (迁移学习特征提取)
  - ModelLibrary 注册与 create_model
"""

import unittest
import numpy as np
import pandas as pd
import torch

from core.deep_learning import TORCH_AVAILABLE, register_deep_learning_models, _AutoArchitecture

if TORCH_AVAILABLE:
    from core.deep_learning import (
        TorchMLP, TorchCNN1D, TorchLSTM, TorchGRU, TorchAutoEncoder,
        _regression_loss,
    )
    from core.nas import TorchNAS, TransferFeatureExtractor
    from core.modeling_engine import ModelLibrary, TaskType


# =============================================================================
# TorchMLP
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRegressionLossShapeSafety(unittest.TestCase):
    def test_column_target_does_not_broadcast_against_flat_prediction(self):
        outputs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        targets = torch.tensor([[1.0], [2.0], [6.0]])

        loss = _regression_loss(torch.nn.MSELoss(), outputs, targets)

        self.assertAlmostEqual(loss.item(), 3.0)

    def test_mismatched_output_size_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "元素数不一致"):
            _regression_loss(
                torch.nn.MSELoss(),
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            )

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchMLP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # n=30 ensures train split (25) is divisible by batch_size=5,
        # avoiding batch=1 which breaks BatchNorm in training mode.
        self.X_cls = pd.DataFrame(np.random.randn(30, 5), columns=[f'f{i}' for i in range(5)])
        self.y_cls = pd.Series(np.random.randint(0, 2, 30))
        self.X_reg = pd.DataFrame(np.random.randn(30, 5))
        self.y_reg = pd.Series(np.random.randn(30))

    def test_init_defaults(self):
        model = TorchMLP()
        self.assertEqual(model.task_type, 'classification')
        self.assertIsNone(model.hidden_dims)
        self.assertEqual(model.epochs, 100)
        self.assertEqual(model.dropout, 0.3)

    def test_init_custom_params(self):
        model = TorchMLP(
            task_type='regression',
            hidden_dims=[32, 16],
            dropout=0.5,
            epochs=10,
            lr=0.01,
            batch_size=32,
            early_stopping_patience=5,
            random_state=123,
            verbose=True,
        )
        self.assertEqual(model.task_type, 'regression')
        self.assertEqual(model.hidden_dims, [32, 16])
        self.assertEqual(model.dropout, 0.5)
        self.assertEqual(model.epochs, 10)
        self.assertEqual(model.lr, 0.01)
        self.assertEqual(model.batch_size, 32)
        self.assertEqual(model.early_stopping_patience, 5)
        self.assertEqual(model.random_state, 123)
        self.assertTrue(model.verbose)

    def test_fit_classification(self):
        model = TorchMLP(
            task_type='classification',
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X_cls, self.y_cls)
        self.assertIsNotNone(model.model_)
        self.assertIsNotNone(model.scaler_)
        self.assertEqual(model.input_dim_, 5)
        self.assertEqual(model.output_dim_, 2)
        self.assertIsNotNone(model.classes_)
        self.assertIn('train_loss', model.history_)
        self.assertIn('val_loss', model.history_)

    def test_predict_classification(self):
        model = TorchMLP(
            task_type='classification',
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X_cls, self.y_cls)
        preds = model.predict(self.X_cls)
        self.assertEqual(len(preds), len(self.X_cls))
        self.assertTrue(np.all(np.isin(preds, model.classes_)))

    def test_predict_proba(self):
        model = TorchMLP(
            task_type='classification',
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X_cls, self.y_cls)
        proba = model.predict_proba(self.X_cls)
        self.assertEqual(proba.shape, (len(self.X_cls), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_regression(self):
        model = TorchMLP(
            task_type='regression',
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X_reg, self.y_reg)
        self.assertIsNotNone(model.model_)
        self.assertEqual(model.output_dim_, 1)

    def test_predict_regression_no_argmax(self):
        model = TorchMLP(
            task_type='regression',
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X_reg, self.y_reg)
        preds = model.predict(self.X_reg)
        self.assertEqual(len(preds), len(self.X_reg))
        # 回归输出应为浮点型
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))
        self.assertEqual(model.predict(self.X_reg.iloc[:1]).shape, (1,))

    def test_early_stopping(self):
        model = TorchMLP(
            task_type='classification',
            hidden_dims=[8],
            epochs=50,
            batch_size=5,
            early_stopping_patience=1,
            random_state=42,
        )
        model.fit(self.X_cls, self.y_cls)
        # 小数据+低耐心，极大概率触发早停
        self.assertLess(len(model.history_['train_loss']), 50)
        self.assertTrue(len(model.history_['train_loss']) > 0)

    def test_get_params(self):
        model = TorchMLP(task_type='regression', hidden_dims=[32], epochs=5, random_state=7)
        params = model.get_params()
        self.assertIn('task_type', params)
        self.assertIn('hidden_dims', params)
        self.assertIn('epochs', params)
        self.assertEqual(params['task_type'], 'regression')
        self.assertEqual(params['hidden_dims'], [32])
        self.assertEqual(params['epochs'], 5)

    def test_set_params(self):
        model = TorchMLP()
        model.set_params(task_type='regression', epochs=10, lr=0.005)
        self.assertEqual(model.task_type, 'regression')
        self.assertEqual(model.epochs, 10)
        self.assertEqual(model.lr, 0.005)

    def test_sklearn_clone(self):
        from sklearn.base import clone
        model = TorchMLP(
            task_type='classification',
            hidden_dims=[16],
            epochs=5,
            batch_size=5,
            random_state=42,
            verbose=True,
        )
        cloned = clone(model)
        self.assertEqual(cloned.task_type, 'classification')
        self.assertEqual(cloned.hidden_dims, [16])
        self.assertEqual(cloned.epochs, 5)
        self.assertIsNone(cloned.model_)

    def test_predict_before_fit_raises(self):
        model = TorchMLP(task_type='classification', hidden_dims=[8], epochs=2, batch_size=5)
        with self.assertRaises(ValueError):
            model.predict(self.X_cls)

    def test_predict_proba_non_classification_raises(self):
        model = TorchMLP(
            task_type='regression',
            hidden_dims=[8],
            epochs=2,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X_reg, self.y_reg)
        with self.assertRaises(ValueError):
            model.predict_proba(self.X_reg)


# =============================================================================
# TorchCNN1D
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchCNN1D(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # n=30 -> train=25, which is divisible by batch_size=5,
        # avoiding batch=1 that causes outputs.squeeze() to remove the batch dim
        # and break CrossEntropyLoss.
        self.X = pd.DataFrame(np.random.randn(30, 8))
        self.y_cls = pd.Series(np.random.randint(0, 2, 30))
        self.y_reg = pd.Series(np.random.randn(30))

    def test_fit_classification(self):
        model = TorchCNN1D(
            task_type='classification',
            hidden_channels=[16],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        self.assertIsNotNone(model.model_)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_fit_regression(self):
        model = TorchCNN1D(
            task_type='regression',
            hidden_channels=[16],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_reg)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))
        self.assertEqual(model.predict(self.X.iloc[:1]).shape, (1,))

    def test_predict_proba(self):
        model = TorchCNN1D(
            task_type='classification',
            hidden_channels=[16],
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 2))

    def test_reshaping_tabular_to_sequence(self):
        # CNN1D 将表格数据的每行视为 1D 序列 (batch, 1, features)
        model = TorchCNN1D(
            task_type='classification',
            hidden_channels=[8],
            epochs=2,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        self.assertEqual(model.input_dim_, 8)


# =============================================================================
# TorchLSTM
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchLSTM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # n=30 -> train=25, batch_size=5 avoids batch=1 squeeze issue.
        self.X = pd.DataFrame(np.random.randn(30, 6))
        self.y_cls = pd.Series(np.random.randint(0, 2, 30))
        self.y_reg = pd.Series(np.random.randn(30))

    def test_fit_classification(self):
        model = TorchLSTM(
            task_type='classification',
            hidden_dim=16,
            num_layers=1,
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        self.assertIsNotNone(model.model_)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_fit_regression(self):
        model = TorchLSTM(
            task_type='regression',
            hidden_dim=16,
            num_layers=1,
            epochs=3,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_reg)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertEqual(model.predict(self.X.iloc[:1]).shape, (1,))

    def test_sequence_reshaping(self):
        # LSTM 将 (batch, features) reshape 为 (batch, seq_len=1, features)
        model = TorchLSTM(
            task_type='classification',
            hidden_dim=16,
            num_layers=1,
            epochs=2,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        self.assertEqual(model.input_dim_, 6)

    def test_predict_proba(self):
        model = TorchLSTM(
            task_type='classification',
            hidden_dim=16,
            num_layers=1,
            epochs=2,
            batch_size=5,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 2))


# =============================================================================
# TorchGRU
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchGRU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # n=20 -> train=17, batch_size=10 gives batches 10 and 7 (no batch=1).
        self.X = pd.DataFrame(np.random.randn(20, 5))
        self.y_cls = pd.Series(np.random.randint(0, 2, 20))
        self.y_reg = pd.Series(np.random.randn(20))

    def test_fit_classification(self):
        model = TorchGRU(
            task_type='classification',
            hidden_dim=16,
            num_layers=1,
            epochs=3,
            batch_size=10,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        self.assertIsNotNone(model.model_)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_fit_regression(self):
        model = TorchGRU(
            task_type='regression',
            hidden_dim=16,
            num_layers=1,
            epochs=3,
            batch_size=10,
            random_state=42,
        )
        model.fit(self.X, self.y_reg)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertEqual(model.predict(self.X.iloc[:1]).shape, (1,))

    def test_small_data(self):
        # n=10 -> train=8, batch_size=4 gives exactly 2 batches of 4.
        X_small = pd.DataFrame(np.random.randn(10, 3))
        y_small = pd.Series(np.random.randint(0, 2, 10))
        model = TorchGRU(
            task_type='classification',
            hidden_dim=8,
            num_layers=1,
            epochs=2,
            batch_size=4,
            random_state=42,
        )
        model.fit(X_small, y_small)
        preds = model.predict(X_small)
        self.assertEqual(len(preds), 10)


# =============================================================================
# TorchAutoEncoder
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchAutoEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(40, 8))

    def test_fit(self):
        model = TorchAutoEncoder(
            encoding_dim=4,
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=16,
            random_state=42,
        )
        model.fit(self.X)
        self.assertIsNotNone(model.model_)

    def test_transform(self):
        model = TorchAutoEncoder(
            encoding_dim=4,
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=16,
            random_state=42,
        )
        model.fit(self.X)
        encoded = model.transform(self.X)
        self.assertEqual(encoded.shape, (40, 4))

    def test_fit_transform(self):
        model = TorchAutoEncoder(
            encoding_dim=4,
            hidden_dims=[16, 8],
            epochs=3,
            batch_size=16,
            random_state=42,
        )
        encoded = model.fit_transform(self.X)
        self.assertEqual(encoded.shape, (40, 4))

    def test_encoding_dimensions(self):
        model = TorchAutoEncoder(
            encoding_dim=3,
            hidden_dims=[12],
            epochs=2,
            batch_size=16,
            random_state=42,
        )
        model.fit(self.X.iloc[:20])
        encoded = model.transform(self.X.iloc[:20])
        self.assertEqual(encoded.shape[1], 3)
        # predict 返回重构误差
        mse = model.predict(self.X.iloc[:20])
        self.assertEqual(mse.shape, (20,))

    def test_predict_returns_reconstruction_error(self):
        model = TorchAutoEncoder(
            encoding_dim=4,
            hidden_dims=[16],
            epochs=2,
            batch_size=16,
            random_state=42,
        )
        model.fit(self.X)
        mse = model.predict(self.X)
        self.assertEqual(mse.shape, (40,))
        self.assertTrue(np.all(mse >= 0))


# =============================================================================
# TorchNAS
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchNAS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(40, 5))
        self.y_cls = pd.Series(np.random.randint(0, 2, 40))
        self.y_reg = pd.Series(np.random.randn(40))

    def test_init_parameter_validation(self):
        model = TorchNAS(
            task_type='classification',
            n_candidates=2,
            epochs=2,
            cv_folds=2,
            random_state=42,
        )
        self.assertEqual(model.n_candidates, 2)
        self.assertEqual(model.epochs, 2)
        self.assertEqual(model.cv_folds, 2)
        self.assertEqual(model.task_type, 'classification')

    def test_fit_small_candidates(self):
        model = TorchNAS(
            task_type='classification',
            n_candidates=2,
            epochs=2,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        self.assertIsNotNone(model.model_)
        self.assertIsNotNone(model.best_arch_)
        self.assertTrue(len(model.search_history_) > 0)

    def test_predict_after_fit(self):
        model = TorchNAS(
            task_type='classification',
            n_candidates=2,
            epochs=2,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_search_space_populated(self):
        model = TorchNAS(
            task_type='classification',
            n_candidates=2,
            epochs=2,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        for h in model.search_history_:
            self.assertIn('hidden_dims', h['architecture'])
            self.assertIn('activation', h['architecture'])
            self.assertIn('dropout', h['architecture'])
            self.assertIn('score', h)
            self.assertIn('candidate_id', h)

    def test_fit_regression(self):
        model = TorchNAS(
            task_type='regression',
            n_candidates=2,
            epochs=2,
            random_state=42,
        )
        model.fit(self.X, self.y_reg)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertEqual(model.predict(self.X.iloc[:1]).shape, (1,))

    def test_predict_proba(self):
        model = TorchNAS(
            task_type='classification',
            n_candidates=2,
            epochs=2,
            random_state=42,
        )
        model.fit(self.X, self.y_cls)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# =============================================================================
# TransferFeatureExtractor
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTransferFeatureExtractor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(30, 8))

    def test_fit(self):
        extractor = TransferFeatureExtractor(
            encoding_dim=4,
            epochs=3,
            random_state=42,
        )
        extractor.fit(self.X)
        self.assertIsNotNone(extractor.encoder_)

    def test_transform(self):
        extractor = TransferFeatureExtractor(
            encoding_dim=4,
            epochs=3,
            random_state=42,
        )
        extractor.fit(self.X)
        encoded = extractor.transform(self.X)
        self.assertEqual(encoded.shape, (30, 4))

    def test_fit_transform(self):
        extractor = TransferFeatureExtractor(
            encoding_dim=4,
            epochs=3,
            random_state=42,
        )
        encoded = extractor.fit_transform(self.X)
        self.assertEqual(encoded.shape, (30, 4))

    def test_encoding_dimension(self):
        extractor = TransferFeatureExtractor(
            encoding_dim=3,
            epochs=2,
            random_state=42,
        )
        encoded = extractor.fit_transform(self.X)
        self.assertEqual(encoded.shape[1], 3)


# =============================================================================
# Model Registration
# =============================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestModelRegistration(unittest.TestCase):
    def setUp(self):
        ModelLibrary._init()

    def test_dl_models_registered_in_library(self):
        cls_models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        reg_models = ModelLibrary.get_models(TaskType.REGRESSION)

        expected = ['torch_mlp', 'torch_cnn1d', 'torch_lstm', 'torch_gru', 'torch_nas']
        for key in expected:
            self.assertIn(key, cls_models, f"{key} not found in classification models")
            self.assertIn(key, reg_models, f"{key} not found in regression models")

    def test_create_model_torch_mlp(self):
        model = ModelLibrary.create_model('torch_mlp', TaskType.CLASSIFICATION)
        self.assertIsInstance(model, TorchMLP)
        model_reg = ModelLibrary.create_model('torch_mlp', TaskType.REGRESSION)
        self.assertIsInstance(model_reg, TorchMLP)
        self.assertEqual(model_reg.task_type, 'regression')

    def test_create_model_torch_cnn1d(self):
        model = ModelLibrary.create_model('torch_cnn1d', TaskType.CLASSIFICATION)
        self.assertIsInstance(model, TorchCNN1D)

    def test_create_model_torch_lstm(self):
        model = ModelLibrary.create_model('torch_lstm', TaskType.CLASSIFICATION)
        self.assertIsInstance(model, TorchLSTM)

    def test_create_model_torch_gru(self):
        model = ModelLibrary.create_model('torch_gru', TaskType.CLASSIFICATION)
        self.assertIsInstance(model, TorchGRU)

    def test_create_model_torch_nas(self):
        model = ModelLibrary.create_model('torch_nas', TaskType.CLASSIFICATION)
        self.assertIsInstance(model, TorchNAS)

    def test_create_model_with_override_params(self):
        model = ModelLibrary.create_model(
            'torch_mlp', TaskType.CLASSIFICATION, epochs=5, hidden_dims=[32]
        )
        self.assertEqual(model.epochs, 5)
        self.assertEqual(model.hidden_dims, [32])

    def test_dl_model_keys_expected(self):
        # DL_MODEL_KEYS 是 ModelingEngine 内部常量;
        # 此处验证 ModelLibrary 中实际已注册的关键 DL 模型
        cls_keys = set(ModelLibrary.get_models(TaskType.CLASSIFICATION).keys())
        expected_dl = {
            'torch_mlp', 'torch_cnn1d', 'torch_lstm', 'torch_gru', 'torch_nas'
        }
        self.assertTrue(
            expected_dl.issubset(cls_keys),
            f"Missing DL keys: {expected_dl - cls_keys}"
        )


if __name__ == '__main__':
    unittest.main()
