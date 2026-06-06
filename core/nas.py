"""
轻量级神经架构搜索 (NAS) 与迁移学习

NAS:
  - 自动搜索 MLP 架构（层数、宽度、激活、dropout）
  - 基于交叉验证评估候选架构
  - 限制搜索空间保证效率（默认最多 8 个候选）

迁移学习:
  - 预训练 AutoEncoder 特征提取
  - 支持加载外部预训练权重进行微调
"""

import random
import time
from typing import Dict, List, Optional, Any, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.helpers import log_info, log_warning


# =============================================================================
# NAS: 轻量级架构搜索
# =============================================================================

class _CandidateNet(nn.Module):
    """候选网络架构，动态构建"""
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int], activation: str,
                 dropout: float, task_type: str = 'classification') -> None:
        super().__init__()
        self.task_type = task_type
        layers: List[nn.Module] = []
        prev = input_dim
        act_map: Dict[str, type] = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
        }
        act_cls = act_map.get(activation, nn.ReLU)
        
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                act_cls(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchNAS(BaseEstimator):
    """
    PyTorch 神经架构搜索 (NAS)
    
    在 fit 中自动搜索最优 MLP 架构，搜索空间包括：
      - 层数 (1-3)
      - 每层宽度 (32, 64, 128, 256)
      - 激活函数 (ReLU, LeakyReLU, GELU)
      - Dropout (0.1, 0.3, 0.5)
      - 学习率、batch_size
    
    使用 2-fold 快速 CV 筛选候选，最佳架构用完整数据重训。
    """
    
    def __init__(self, task_type: str = 'classification',
                 n_candidates: int = 8,
                 cv_folds: int = 2,
                 epochs: int = 30,
                 random_state: int = 42,
                 verbose: bool = False) -> None:
        self.task_type = task_type
        self.n_candidates = n_candidates
        self.cv_folds = cv_folds
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_: Optional[nn.Module] = None
        self.scaler_ = StandardScaler()
        self.label_encoder_: Optional[LabelEncoder] = None
        self.best_arch_: Optional[Dict[str, Any]] = None
        self.search_history_: List[Dict[str, Any]] = []
        self.classes_: Optional[np.ndarray] = None
    
    def _sample_architectures(self, input_dim: int, output_dim: int,
                              rng: np.random.RandomState) -> List[Dict[str, Any]]:
        """采样候选架构"""
        candidates: List[Dict[str, Any]] = []
        hidden_options = [32, 64, 128, 256]
        activation_options = ['relu', 'leaky_relu', 'gelu']
        dropout_options = [0.1, 0.3, 0.5]
        
        for _ in range(self.n_candidates):
            n_layers = rng.randint(1, 4)  # 1-3 层
            hidden_dims = [int(rng.choice(hidden_options)) for _ in range(n_layers)]
            arch: Dict[str, Any] = {
                'hidden_dims': hidden_dims,
                'activation': str(rng.choice(activation_options)),
                'dropout': float(rng.choice(dropout_options)),
                'lr': float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-2)))),
                'batch_size': int(rng.choice([32, 64, 128])),
            }
            candidates.append(arch)
        return candidates
    
    def _build_model(self, input_dim: int, output_dim: int, arch: Dict[str, Any]) -> nn.Module:
        """根据架构字典构建模型"""
        return _CandidateNet(
            input_dim, output_dim,
            arch['hidden_dims'], arch['activation'],
            arch['dropout'], self.task_type
        ).to(self.device)
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader,
                     criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            outputs = model(Xb)
            if self.task_type == 'classification':
                loss = criterion(outputs, yb)
            else:
                loss = criterion(outputs.squeeze(), yb.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def _eval_epoch(self, model: nn.Module, loader: DataLoader,
                    criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                outputs = model(Xb)
                if self.task_type == 'classification':
                    loss = criterion(outputs, yb)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
                else:
                    loss = criterion(outputs.squeeze(), yb.float())
                    total_loss += loss.item()
        if self.task_type == 'classification':
            return correct / max(total, 1)
        return total_loss / max(len(loader), 1)
    
    def _quick_cv_score(self, X_np: np.ndarray, y_np: np.ndarray,
                        arch: Dict[str, Any]) -> float:
        """用少量 epoch 快速评估架构"""
        input_dim = X_np.shape[1]
        output_dim = len(np.unique(y_np)) if self.task_type == 'classification' else 1
        
        # 简单 train/val split（比完整 KFold 更快）
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_np, y_np, test_size=0.2, random_state=self.random_state,
            stratify=y_np if self.task_type == 'classification' else None
        )
        
        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=arch['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=arch['batch_size'], shuffle=False)
        
        model = self._build_model(input_dim, output_dim, arch)
        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=arch['lr'])
        
        best_val_metric = float('-inf') if self.task_type == 'classification' else float('inf')
        patience = 0
        for epoch in range(self.epochs):
            self._train_epoch(model, train_loader, criterion, optimizer)
            val_metric = self._eval_epoch(model, val_loader, criterion)
            
            if self.task_type == 'classification':
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience = 0
                else:
                    patience += 1
            else:
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    patience = 0
                else:
                    patience += 1
            
            if patience >= 5:
                break
        
        return best_val_metric
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'TorchNAS':
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
            self.classes_ = self.label_encoder_.classes_
        else:
            y_np = np.array(y).astype(np.float32).reshape(-1, 1)
        
        rng = np.random.RandomState(self.random_state)
        candidates = self._sample_architectures(
            X_np.shape[1],
            len(np.unique(y_np)) if self.task_type == 'classification' else 1,
            rng
        )
        
        log_info(f"[TorchNAS] 开始架构搜索: {len(candidates)} 个候选")
        start_time = time.time()
        
        # 评估候选架构
        scores: List[float] = []
        for i, arch in enumerate(candidates):
            t0 = time.time()
            score = self._quick_cv_score(X_np, y_np, arch)
            scores.append(score)
            self.search_history_.append({
                'candidate_id': i,
                'architecture': deepcopy(arch),
                'score': score,
                'time': time.time() - t0,
            })
            if self.verbose:
                log_info(f"[TorchNAS] 候选 {i+1}/{len(candidates)}: "
                         f"hidden={arch['hidden_dims']}, act={arch['activation']}, "
                         f"dropout={arch['dropout']:.1f}, score={score:.4f}")
        
        # 选择最佳架构
        if self.task_type == 'classification':
            best_idx = int(np.argmax(scores))
        else:
            best_idx = int(np.argmin(scores))
        
        self.best_arch_ = candidates[best_idx]
        log_info(f"[TorchNAS] 最优架构: {self.best_arch_['hidden_dims']}, "
                 f"act={self.best_arch_['activation']}, score={scores[best_idx]:.4f}, "
                 f"搜索耗时: {time.time() - start_time:.1f}s")
        
        # 用最佳架构在完整数据上训练最终模型
        self.model_ = self._build_model(
            X_np.shape[1],
            len(self.classes_) if self.task_type == 'classification' else 1,
            self.best_arch_
        )
        
        train_ds = TensorDataset(torch.tensor(X_np), torch.tensor(y_np))
        train_loader = DataLoader(train_ds, batch_size=self.best_arch_['batch_size'], shuffle=True)
        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.best_arch_['lr'])
        
        for epoch in range(self.epochs * 2):  # 最终训练更多 epoch
            self._train_epoch(self.model_, train_loader, criterion, optimizer)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.tensor(X_np).to(self.device))
            if self.task_type == 'classification':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                return self.label_encoder_.inverse_transform(preds)
            return outputs.squeeze().cpu().numpy()
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.task_type != 'classification':
            raise ValueError('only classification supports predict_proba')
        X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.tensor(X_np).to(self.device))
            return torch.softmax(outputs, dim=1).cpu().numpy()


# =============================================================================
# 迁移学习: 预训练特征提取器
# =============================================================================

class TransferFeatureExtractor(BaseEstimator):
    """
    迁移学习特征提取器
    
    使用预训练的 AutoEncoder 编码器部分作为特征提取器，
    输出低维表示供下游模型使用。
    
    用法:
        extractor = TransferFeatureExtractor(encoding_dim=16, pretrained_path='ae_weights.pt')
        X_new = extractor.fit_transform(X)
    """
    
    def __init__(self, encoding_dim: int = 16,
                 hidden_dims: List[int] = [128, 64],
                 pretrained_path: Optional[str] = None,
                 epochs: int = 50,
                 random_state: int = 42) -> None:
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.random_state = random_state
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_: Optional[nn.Module] = None
        self.scaler_ = StandardScaler()
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'TransferFeatureExtractor':
        from core.deep_learning import _AutoEncoderNet
        torch.manual_seed(self.random_state)
        
        X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
        input_dim = X_np.shape[1]
        
        # 构建或加载预训练编码器
        ae = _AutoEncoderNet(input_dim, self.encoding_dim, self.hidden_dims).to(self.device)
        
        if self.pretrained_path:
            try:
                ae.load_state_dict(torch.load(self.pretrained_path, map_location=self.device))
                log_info(f"[TransferLearning] 加载预训练权重: {self.pretrained_path}")
            except Exception as e:
                log_warning(f"[TransferLearning] 加载预训练权重失败: {e}，将重新训练")
        
        # 在目标数据上微调 AutoEncoder
        X_t = torch.tensor(X_np).to(self.device)
        optimizer = optim.Adam(ae.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        ae.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            recon, _ = ae(X_t)
            loss = criterion(recon, X_t)
            loss.backward()
            optimizer.step()
        
        # 提取编码器部分
        self.encoder_ = ae.encoder
        self.encoder_.eval()
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
        self.encoder_.eval()
        with torch.no_grad():
            encoded = self.encoder_(torch.tensor(X_np).to(self.device))
            return encoded.cpu().numpy()
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
