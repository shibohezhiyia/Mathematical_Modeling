"""
多模态扩展测试 — 使用 mock 避免重型下载

覆盖 ImageResNet 与 TextBERT 的初始化、构建、训练、预测、边界条件。
"""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import multimodal as mm


# =============================================================================
# Fake models that emit real tensors so torch ops work
# =============================================================================

class FakeResNetModel:
    def __init__(self, num_classes, task_type):
        self.num_classes = num_classes
        self.task_type = task_type

    def __call__(self, x):
        batch_size = x.shape[0]
        if self.task_type == "classification":
            return torch.randn(batch_size, self.num_classes, requires_grad=True)
        return torch.randn(batch_size, requires_grad=True)

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        return [MagicMock(requires_grad=True)]

    def to(self, device):
        return self


class MockTextDataset:
    def __init__(self, texts, labels=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.ones(self.max_length, dtype=torch.long),
            "attention_mask": torch.ones(self.max_length, dtype=torch.long),
        }
        if self.labels is not None:
            if self.labels.dtype in (np.float32, np.float64):
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
            else:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class FakeBERTModel:
    def __init__(self, num_classes, task_type):
        self.num_classes = num_classes
        self.task_type = task_type
        self.bert = MagicMock()

    def __call__(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        if self.task_type == "classification":
            return torch.randn(batch_size, self.num_classes, requires_grad=True)
        return torch.randn(batch_size, 1, requires_grad=True)

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        return [MagicMock(requires_grad=True)]

    def to(self, device):
        return self


def _create_mock_tokenizer():
    mock_tok = MagicMock()

    def encode(text, max_length=128, **kwargs):
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)
        return {
            "input_ids": torch.ones(batch_size, max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
        }

    mock_tok.side_effect = encode
    return mock_tok


# =============================================================================
# ImageResNet
# =============================================================================

class TestImageResNet:
    def test_init_defaults(self):
        model = mm.ImageResNet()
        assert model.task_type == "classification"
        assert model.image_col == "image_path"
        assert model.freeze_backbone is True
        assert model.epochs == 10

    def test_init_custom_params(self):
        model = mm.ImageResNet(
            task_type="regression",
            image_col="img",
            epochs=5,
            lr=1e-3,
            batch_size=8,
            freeze_backbone=False,
        )
        assert model.task_type == "regression"
        assert model.image_col == "img"
        assert model.epochs == 5
        assert model.lr == 1e-3
        assert model.batch_size == 8
        assert model.freeze_backbone is False

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    @patch("core.multimodal.models.resnet18")
    def test_build_model_returns_correct_architecture(self, mock_resnet18):
        mock_backbone = MagicMock()
        mock_backbone.fc = MagicMock()
        mock_backbone.fc.in_features = 512
        mock_backbone.to.return_value = mock_backbone
        mock_resnet18.return_value = mock_backbone

        model = mm.ImageResNet(task_type="classification")
        model.num_classes_ = 3
        built = model._build_model()
        assert built is mock_backbone
        assert isinstance(mock_backbone.fc, torch.nn.Linear)

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    @patch("core.multimodal.models.resnet18")
    def test_build_model_freeze_backbone(self, mock_resnet18):
        mock_backbone = MagicMock()
        mock_backbone.fc = MagicMock()
        mock_backbone.fc.in_features = 512
        mock_backbone.to.return_value = mock_backbone
        param1 = MagicMock()
        mock_backbone.parameters.return_value = [param1]
        mock_resnet18.return_value = mock_backbone

        model = mm.ImageResNet(freeze_backbone=True)
        model.num_classes_ = 2
        model._build_model()
        assert param1.requires_grad is False

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    @patch("core.multimodal.optim.Adam")
    def test_fit_classification(self, mock_adam):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        model = mm.ImageResNet(task_type="classification", epochs=1, batch_size=2)
        fake_model = FakeResNetModel(2, "classification")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame(
                {
                    "image_path": [
                        "fake1.jpg",
                        "fake2.jpg",
                        "fake3.jpg",
                        "fake4.jpg",
                    ]
                }
            )
            y = np.array([0, 1, 0, 1])
            model.fit(df, y)
            assert model.model_ is fake_model
            assert model.label_encoder_ is not None
            assert model.num_classes_ == 2

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    @patch("core.multimodal.optim.Adam")
    def test_fit_regression(self, mock_adam):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        model = mm.ImageResNet(task_type="regression", epochs=1, batch_size=2)
        fake_model = FakeResNetModel(1, "regression")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame(
                {"image_path": ["fake1.jpg", "fake2.jpg", "fake3.jpg"]}
            )
            y = np.array([1.5, 2.5, 3.5])
            model.fit(df, y)
            assert model.model_ is fake_model
            assert model.num_classes_ == 1

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    @patch("core.multimodal.optim.Adam")
    def test_predict_classification(self, mock_adam):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        model = mm.ImageResNet(task_type="classification", epochs=1, batch_size=2)
        fake_model = FakeResNetModel(2, "classification")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame(
                {
                    "image_path": [
                        "fake1.jpg",
                        "fake2.jpg",
                        "fake3.jpg",
                        "fake4.jpg",
                    ]
                }
            )
            y = np.array([0, 1, 0, 1])
            model.fit(df, y)
            preds = model.predict(df)
            assert len(preds) == len(df)

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    @patch("core.multimodal.optim.Adam")
    def test_predict_proba(self, mock_adam):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        model = mm.ImageResNet(task_type="classification", epochs=1, batch_size=2)
        fake_model = FakeResNetModel(2, "classification")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame(
                {
                    "image_path": [
                        "fake1.jpg",
                        "fake2.jpg",
                        "fake3.jpg",
                        "fake4.jpg",
                    ]
                }
            )
            y = np.array([0, 1, 0, 1])
            model.fit(df, y)
            proba = model.predict_proba(df)
            assert proba.shape == (4, 2)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.skipif(not mm.TORCHVISION_AVAILABLE, reason="torchvision not available")
    def test_missing_image_column(self):
        model = mm.ImageResNet(image_col="missing_col")
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少图像列"):
            model.fit(df, np.array([0, 1, 0]))


# =============================================================================
# TextBERT
# =============================================================================

class TestTextBERT:
    def test_init_defaults(self):
        model = mm.TextBERT()
        assert model.task_type == "classification"
        assert model.text_col == "text"
        assert model.freeze_backbone is True
        assert model.epochs == 3

    def test_init_custom_params(self):
        model = mm.TextBERT(
            task_type="regression",
            text_col="content",
            epochs=2,
            lr=1e-4,
            batch_size=4,
            freeze_backbone=False,
            max_length=256,
        )
        assert model.task_type == "regression"
        assert model.text_col == "content"
        assert model.epochs == 2
        assert model.lr == 1e-4
        assert model.batch_size == 4
        assert model.freeze_backbone is False
        assert model.max_length == 256

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    @patch("core.multimodal.DistilBertModel.from_pretrained")
    def test_build_model_returns_correct_architecture(self, mock_bert):
        mock_instance = MagicMock()
        mock_bert.return_value = mock_instance

        model = mm.TextBERT(task_type="classification")
        model.num_classes_ = 3
        built = model._build_model()
        assert built is not None
        assert hasattr(built, "bert")

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    @patch("core.multimodal.DistilBertModel.from_pretrained")
    def test_build_model_freeze_backbone(self, mock_bert):
        mock_instance = MagicMock()
        param1 = MagicMock()
        mock_instance.parameters.return_value = [param1]
        mock_bert.return_value = mock_instance

        model = mm.TextBERT(freeze_backbone=True)
        model.num_classes_ = 2
        model._build_model()
        assert param1.requires_grad is False

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    @patch("core.multimodal.DistilBertTokenizer.from_pretrained")
    @patch("core.multimodal.optim.Adam")
    def test_fit_classification(self, mock_adam, mock_tokenizer):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        mock_tokenizer.return_value = _create_mock_tokenizer()

        model = mm.TextBERT(task_type="classification", epochs=1, batch_size=2)
        fake_model = FakeBERTModel(2, "classification")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame({"text": ["a", "b", "c", "d"]})
            y = np.array([0, 1, 0, 1])
            model.fit(df, y)
            assert model.model_ is fake_model
            assert model.label_encoder_ is not None
            assert model.num_classes_ == 2

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    @patch("core.multimodal.DistilBertTokenizer.from_pretrained")
    @patch("core.multimodal.optim.Adam")
    def test_fit_regression(self, mock_adam, mock_tokenizer):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        mock_tokenizer.return_value = _create_mock_tokenizer()

        model = mm.TextBERT(task_type="regression", epochs=1, batch_size=2)
        fake_model = FakeBERTModel(1, "regression")

        with patch.object(mm, "_TextDataset", MockTextDataset):
            with patch.object(model, "_build_model", return_value=fake_model):
                df = pd.DataFrame({"text": ["a", "b", "c"]})
                y = np.array([1.0, 2.0, 3.0])
                model.fit(df, y)
                assert model.model_ is fake_model
                assert model.num_classes_ == 1

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    @patch("core.multimodal.DistilBertTokenizer.from_pretrained")
    @patch("core.multimodal.optim.Adam")
    def test_predict_classification(self, mock_adam, mock_tokenizer):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        mock_tokenizer.return_value = _create_mock_tokenizer()

        model = mm.TextBERT(task_type="classification", epochs=1, batch_size=2)
        fake_model = FakeBERTModel(2, "classification")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame({"text": ["a", "b", "c", "d"]})
            y = np.array([0, 1, 0, 1])
            model.fit(df, y)
            preds = model.predict(df)
            assert len(preds) == len(df)

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    @patch("core.multimodal.DistilBertTokenizer.from_pretrained")
    @patch("core.multimodal.optim.Adam")
    def test_predict_proba(self, mock_adam, mock_tokenizer):
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer
        mock_tokenizer.return_value = _create_mock_tokenizer()

        model = mm.TextBERT(task_type="classification", epochs=1, batch_size=2)
        fake_model = FakeBERTModel(2, "classification")

        with patch.object(model, "_build_model", return_value=fake_model):
            df = pd.DataFrame({"text": ["a", "b", "c", "d"]})
            y = np.array([0, 1, 0, 1])
            model.fit(df, y)
            proba = model.predict_proba(df)
            assert proba.shape == (4, 2)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    def test_missing_text_column(self):
        model = mm.TextBERT(text_col="missing_col")
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少文本列"):
            model.fit(df, np.array([0, 1, 0]))

    @pytest.mark.skipif(not mm.TRANSFORMERS_AVAILABLE, reason="transformers not available")
    def test_predict_proba_raises_on_regression(self):
        model = mm.TextBERT(task_type="regression")
        with pytest.raises(ValueError, match="only classification supports predict_proba"):
            model.predict_proba(pd.DataFrame({"text": ["a"]}))
