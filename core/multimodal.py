"""
多模态数据支持 — 图像与文本模型

为表格数据平台添加图像/文本处理能力：
  - ImageResNet: 从 DataFrame 的图像路径列加载图像，使用预训练 ResNet18 微调
  - TextBERT: 从 DataFrame 的文本列提取特征，使用 DistilBERT 微调

使用方式:
    df['image_path'] = 'path/to/image.jpg'
    model = ImageResNet(task_type='classification', image_col='image_path')
    model.fit(df, y)
"""

import os
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TORCHVISION_AVAILABLE = False
try:
    import torchvision
    from torchvision import transforms, models
    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

TRANSFORMERS_AVAILABLE = False
try:
    import transformers
    from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# 图像模型
# =============================================================================

class _ImageDataset(Dataset):
    """图像数据集"""
    def __init__(self, df: pd.DataFrame, image_col: str,
                 labels: Optional[np.ndarray] = None,
                 transform: Optional[Callable] = None) -> None:
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        from PIL import Image
        path = self.df.loc[idx, self.image_col]
        if not os.path.exists(str(path)):
            # 如果路径不存在，返回零张量
            img = torch.zeros(3, 224, 224)
        else:
            img = Image.open(str(path)).convert('RGB')
            img = self.transform(img)
        
        if self.labels is not None:
            return img, self.labels[idx]
        return img


class ImageResNet(BaseEstimator):
    """
    图像分类模型 — 基于预训练 ResNet18
    
    Args:
        task_type: 'classification' 或 'regression'
        image_col: DataFrame 中存储图像路径的列名
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        freeze_backbone: 是否冻结骨干网络（只训练分类头）
    """
    
    def __init__(self, task_type: str = 'classification',
                 image_col: str = 'image_path',
                 epochs: int = 10, lr: float = 1e-4,
                 batch_size: int = 16, freeze_backbone: bool = True,
                 random_state: int = 42) -> None:
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision 未安装")
        
        self.task_type = task_type
        self.image_col = image_col
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.freeze_backbone = freeze_backbone
        self.random_state = random_state
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_: Optional[nn.Module] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.num_classes_: Optional[int] = None
    
    def _build_model(self) -> nn.Module:
        """构建 ResNet18 + 自定义分类头"""
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if self.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, self.num_classes_)
        return backbone.to(self.device)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ImageResNet':
        torch.manual_seed(self.random_state)
        
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        if self.image_col not in df.columns:
            raise ValueError(f"DataFrame 中缺少图像列 '{self.image_col}'")
        
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y).astype(np.int64)
            self.classes_ = self.label_encoder_.classes_
            self.num_classes_ = len(self.classes_)
        else:
            y_enc = np.array(y).astype(np.float32).reshape(-1)
            self.num_classes_ = 1
        
        dataset = _ImageDataset(df, self.image_col, y_enc)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_ = self._build_model()
        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model_.parameters()), lr=self.lr)
        
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model_(images)
                if self.task_type == 'classification':
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        dataset = _ImageDataset(df, self.image_col)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model_.eval()
        preds: List[Any] = []
        with torch.no_grad():
            for images in loader:
                images = images.to(self.device)
                outputs = self.model_(images)
                if self.task_type == 'classification':
                    p = torch.argmax(outputs, dim=1).cpu().numpy()
                    preds.extend(p)
                else:
                    preds.extend(outputs.squeeze().cpu().numpy())
        
        preds_arr = np.array(preds)
        if self.task_type == 'classification' and self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(preds_arr)
        return preds_arr
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.task_type != 'classification':
            raise ValueError('only classification supports predict_proba')
        
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        dataset = _ImageDataset(df, self.image_col)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model_.eval()
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for images in loader:
                images = images.to(self.device)
                outputs = self.model_(images)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        
        return np.vstack(probs)


# =============================================================================
# 文本模型
# =============================================================================

class _TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts: List[str],
                 labels: Optional[np.ndarray] = None,
                 max_length: int = 128) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 未安装")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, max_length=self.max_length,
            padding='max_length', truncation=True,
            return_tensors='pt'
        )
        item: Dict[str, torch.Tensor] = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TextBERT(BaseEstimator):
    """
    文本分类模型 — 基于预训练 DistilBERT
    
    Args:
        task_type: 'classification' 或 'regression'
        text_col: DataFrame 中存储文本的列名
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        max_length: 最大序列长度
        freeze_backbone: 是否冻结 BERT 骨干
    """
    
    def __init__(self, task_type: str = 'classification',
                 text_col: str = 'text',
                 epochs: int = 3, lr: float = 5e-5,
                 batch_size: int = 16, max_length: int = 128,
                 freeze_backbone: bool = True, random_state: int = 42) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 未安装")
        
        self.task_type = task_type
        self.text_col = text_col
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone
        self.random_state = random_state
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_: Optional[nn.Module] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.num_classes_: Optional[int] = None
    
    def _build_model(self) -> nn.Module:
        """构建 DistilBERT + 分类头"""
        bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        if self.freeze_backbone:
            for param in bert.parameters():
                param.requires_grad = False
        
        class BertClassifier(nn.Module):
            def __init__(self, bert_model: nn.Module, num_classes: int, hidden_size: int = 768) -> None:
                super().__init__()
                self.bert = bert_model
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                return self.classifier(pooled)
        
        return BertClassifier(bert, self.num_classes_).to(self.device)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'TextBERT':
        torch.manual_seed(self.random_state)
        
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        if self.text_col not in df.columns:
            raise ValueError(f"DataFrame 中缺少文本列 '{self.text_col}'")
        
        texts = df[self.text_col].astype(str).tolist()
        
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y).astype(np.int64)
            self.classes_ = self.label_encoder_.classes_
            self.num_classes_ = len(self.classes_)
        else:
            y_enc = np.array(y).astype(np.float32).reshape(-1)
            self.num_classes_ = 1
        
        dataset = _TextDataset(texts, y_enc, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_ = self._build_model()
        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model_.parameters()), lr=self.lr)
        
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model_(input_ids, attention_mask)
                if self.task_type == 'classification':
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        texts = df[self.text_col].astype(str).tolist()
        dataset = _TextDataset(texts, max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model_.eval()
        preds: List[Any] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model_(input_ids, attention_mask)
                if self.task_type == 'classification':
                    p = torch.argmax(outputs, dim=1).cpu().numpy()
                    preds.extend(p)
                else:
                    preds.extend(outputs.squeeze().cpu().numpy())
        
        preds_arr = np.array(preds)
        if self.task_type == 'classification' and self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(preds_arr)
        return preds_arr
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.task_type != 'classification':
            raise ValueError('only classification supports predict_proba')
        
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        texts = df[self.text_col].astype(str).tolist()
        dataset = _TextDataset(texts, max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model_.eval()
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model_(input_ids, attention_mask)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        
        return np.vstack(probs)
