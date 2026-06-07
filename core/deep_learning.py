"""
深度学习引擎

提供基于 PyTorch 的深度学习模型，统一 sklearn 接口：
- MLP（多层感知机）
- TabNet（如果安装）
- 自动网络架构搜索（根据输入维度）
- GPU 加速
- 早停、学习率调度

无 PyTorch 时优雅降级。
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from utils.helpers import log_info, log_warning
from core.modeling_engine import ModelLibrary

# 尝试导入 PyTorch
TORCH_AVAILABLE = False
TABNET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    log_warning("[DeepLearningEngine] PyTorch 未安装，深度学习模型不可用")

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# 自动网络架构设计（不依赖PyTorch）
# =============================================================================

class _AutoArchitecture:
    """自动网络架构设计"""
    
    @staticmethod
    def suggest_architecture(n_features: int, n_samples: int, task_type: str) -> Dict[str, Any]:
        """
        根据数据规模自动建议网络架构
        
        Returns:
            {'hidden_dims': [...], 'dropout': 0.x, 'batch_size': ..., 'epochs': ...}
        """
        if n_samples < 1000:
            hidden_dims = [64, 32]
            epochs = 50
            batch_size = 32
        elif n_samples < 10000:
            hidden_dims = [128, 64, 32]
            epochs = 100
            batch_size = 64
        elif n_samples < 30000:
            hidden_dims = [128, 64]
            epochs = 50
            batch_size = 128
        elif n_samples < 100000:
            # 大数据轻量化：2层隐藏层 + 较少epoch即可收敛
            hidden_dims = [128, 64]
            epochs = 20
            batch_size = 256
        else:
            # 超大数据：极简网络
            hidden_dims = [64, 32]
            epochs = 15
            batch_size = 512
        
        if n_features < 10:
            hidden_dims = [h // 2 for h in hidden_dims[:2]]
        elif n_features > 500:
            hidden_dims = [min(h * 2, 256) for h in hidden_dims]
        
        dropout = 0.5 if n_samples < 1000 else 0.3
        
        return {
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': 0.001,
            'use_batchnorm': True
        }


# =============================================================================
# PyTorch 模型（条件定义）
# =============================================================================

if TORCH_AVAILABLE:
    
    class _MLPNet(nn.Module):
        """多层感知机网络"""
        
        def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                     dropout: float = 0.3, use_batchnorm: bool = True) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            
            for i, h_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, h_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            
            self.feature_extractor = nn.Sequential(*layers)
            self.output = nn.Linear(prev_dim, output_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.feature_extractor(x)
            return self.output(x)
    
    
    class TorchMLP(BaseEstimator):
        """
        PyTorch MLP，统一 sklearn 接口
        
        使用方式：
            model = TorchMLP(task_type='classification', hidden_dims=[128, 64], epochs=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        """
        
        def __init__(self,
                     task_type: str = 'classification',
                     hidden_dims: Optional[List[int]] = None,
                     dropout: float = 0.3,
                     use_batchnorm: bool = True,
                     lr: float = 0.001,
                     epochs: int = 100,
                     batch_size: int = 64,
                     early_stopping_patience: int = 10,
                     lr_scheduler_patience: int = 5,
                     val_ratio: float = 0.15,
                     random_state: int = 42,
                     device: Optional[str] = None,
                     verbose: bool = False,
                     use_amp: bool = False) -> None:
            self.task_type = task_type
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            self.use_batchnorm = use_batchnorm
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.early_stopping_patience = early_stopping_patience
            self.lr_scheduler_patience = lr_scheduler_patience
            self.val_ratio = val_ratio
            self.random_state = random_state
            self.verbose = verbose
            self.device = device  # store as string for sklearn clone compatibility
            self.use_amp = use_amp
            
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
            self.label_encoder_: Optional[LabelEncoder] = None
            self.input_dim_: Optional[int] = None
            self.output_dim_: Optional[int] = None
            self.classes_: Optional[np.ndarray] = None
            self.history_: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TorchMLP':
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            # 缓存设备解析：避免 fit/predict/_evaluate 每次重复 torch.device(...)
            _device = self._resolve_device()

            X_np = self.scaler_.fit_transform(X) if hasattr(X, 'values') else self.scaler_.fit_transform(np.array(X))
            
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_np = self.label_encoder_.fit_transform(y)
                self.classes_ = self.label_encoder_.classes_
                self.output_dim_ = len(self.classes_)
            else:
                y_np = np.array(y).astype(np.float32)
                self.output_dim_ = 1 if y_np.ndim == 1 else y_np.shape[1]
            
            self.input_dim_ = X_np.shape[1]
            
            if self.hidden_dims is None:
                arch = _AutoArchitecture.suggest_architecture(
                    self.input_dim_, len(X_np), self.task_type
                )
                self.hidden_dims = arch['hidden_dims']
                self.dropout = arch['dropout']
                self.batch_size = arch['batch_size']
                self.epochs = arch['epochs']
            
            if self.val_ratio > 0:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_np, y_np, test_size=self.val_ratio, random_state=self.random_state
                )
            else:
                X_tr, y_tr = X_np, y_np
                X_val, y_val = X_np, y_np

            train_loader = self._create_dataloader(X_tr, y_tr, shuffle=True, device=_device)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False, device=_device)

            self.model_ = _MLPNet(
                self.input_dim_, self.hidden_dims, self.output_dim_,
                self.dropout, self.use_batchnorm
            ).to(_device)

            if _device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.set_num_threads(min(4, os.cpu_count() or 1))

            # AMP: 自动混合精度，仅在 CUDA 且用户启用时生效
            use_amp_active = bool(self.use_amp and _device.type == 'cuda')
            scaler = torch.cuda.amp.GradScaler() if use_amp_active else None

            if self.task_type == 'classification':
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=self.lr_scheduler_patience, factor=0.5
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                self.model_.train()
                train_losses: List[float] = []
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(_device)
                    batch_y = batch_y.to(_device)

                    optimizer.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model_(batch_x)
                            if self.task_type == 'classification':
                                loss = criterion(outputs, batch_y.long())
                            else:
                                loss = criterion(outputs.squeeze(), batch_y.float())

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = self.model_(batch_x)
                        if self.task_type == 'classification':
                            loss = criterion(outputs, batch_y.long())
                        else:
                            loss = criterion(outputs.squeeze(), batch_y.float())

                        loss.backward()
                        optimizer.step()

                    train_losses.append(loss.item())
                
                val_loss = self._evaluate(val_loader, criterion, _device)
                self.history_['train_loss'].append(np.mean(train_losses))
                self.history_['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state_ = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    log_info(f"[TorchMLP] Epoch {epoch+1}/{self.epochs}: "
                             f"train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}")
                
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        log_info(f"[TorchMLP] 早停于 epoch {epoch+1}")
                    break
            
            if hasattr(self, 'best_state_'):
                self.model_.load_state_dict(self.best_state_)
            
            return self
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.model_ is None:
                raise ValueError("模型未训练")

            _device = self._resolve_device()
            X_np = self.scaler_.transform(X) if hasattr(X, 'values') else self.scaler_.transform(np.array(X))
            X_tensor = torch.FloatTensor(X_np).to(_device)
            
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(X_tensor)
            
            if self.task_type == 'classification':
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                if self.label_encoder_ is not None:
                    preds = self.label_encoder_.inverse_transform(preds)
                return preds
            else:
                return outputs.squeeze().cpu().numpy()
        
        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.task_type != 'classification':
                raise ValueError("只有分类任务支持 predict_proba")

            _device = self._resolve_device()
            X_np = self.scaler_.transform(X) if hasattr(X, 'values') else self.scaler_.transform(np.array(X))
            X_tensor = torch.FloatTensor(X_np).to(_device)
            
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(X_tensor)
            
            return torch.softmax(outputs, dim=1).cpu().numpy()
        
        def _resolve_device(self) -> torch.device:
            """解析并缓存 torch.device 对象。

            self.device 既可能是 None、字符串（如 'cuda:0' / 'cpu'），
            也可能是已经解析过的 torch.device 实例。每次 fit/predict/_evaluate
            重复解析会浪费数十微秒；缓存到 self._device_ 后只在首次调用时解析。
            """
            cached = getattr(self, '_device_', None)
            if cached is not None:
                return cached
            if self.device is None:
                resolved = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif isinstance(self.device, torch.device):
                resolved = self.device
            else:
                resolved = torch.device(self.device)
            self._device_ = resolved
            return resolved

        def _create_dataloader(self, X: np.ndarray, y: np.ndarray,
                               shuffle: bool = True,
                               device: Optional[torch.device] = None) -> DataLoader:
            X_tensor = torch.FloatTensor(X)
            if self.task_type == 'classification':
                y_tensor = torch.FloatTensor(y) if self.output_dim_ == 2 else torch.LongTensor(y)
            else:
                y_tensor = torch.FloatTensor(y)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            num_workers = min(4, max(0, (os.cpu_count() or 1) // 2))
            pin_memory = bool(device is not None and device.type == 'cuda')
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0
            )
        
        def _evaluate(self, dataloader: DataLoader, criterion: nn.Module,
                      device: Optional[torch.device] = None) -> float:
            self.model_.eval()
            if device is None:
                device = self._resolve_device()
            losses: List[float] = []
            amp_active = bool(self.use_amp and device.type == 'cuda')
            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    if amp_active:
                        with torch.cuda.amp.autocast():
                            outputs = self.model_(batch_x)
                            if self.task_type == 'classification':
                                loss = criterion(outputs, batch_y.long())
                            else:
                                loss = criterion(outputs.squeeze(), batch_y.float())
                    else:
                        outputs = self.model_(batch_x)
                        if self.task_type == 'classification':
                            loss = criterion(outputs, batch_y.long())
                        else:
                            loss = criterion(outputs.squeeze(), batch_y.float())

                    losses.append(loss.item())
            return np.mean(losses) if losses else float('inf')


# =============================================================================
# 扩展深度学习模型（CNN1D / LSTM / AutoEncoder）
# =============================================================================

if TORCH_AVAILABLE:
    
    class _CNN1DNet(nn.Module):
        """1D CNN网络"""
        def __init__(self, input_dim: int, output_dim: int,
                     hidden_channels: Optional[List[int]] = None,
                     kernel_size: int = 3,
                     dropout: float = 0.3,
                     task_type: str = 'classification') -> None:
            if hidden_channels is None:
                hidden_channels = [32, 64]
            super().__init__()
            self.task_type = task_type
            layers: List[nn.Module] = []
            in_ch = 1
            for out_ch in hidden_channels:
                layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout / 2),
                ])
                in_ch = out_ch
            self.conv = nn.Sequential(*layers)
            self.flatten_dim = self._get_flatten_dim(input_dim, hidden_channels, kernel_size)
            self.fc = nn.Sequential(
                nn.Linear(self.flatten_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
            )
        
        def _get_flatten_dim(self, input_dim: int, hidden_channels: List[int],
                             kernel_size: int) -> int:
            with torch.no_grad():
                x = torch.zeros(1, 1, input_dim)
                x = self.conv(x)
                return x.view(1, -1).shape[1]
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class TorchCNN1D(BaseEstimator):
        """1D CNN，统一sklearn接口"""
        def __init__(self, task_type: str = 'classification',
                     hidden_channels: Optional[List[int]] = None,
                     kernel_size: int = 3, dropout: float = 0.3, lr: float = 0.001,
                     epochs: int = 50, batch_size: int = 64,
                     early_stopping_patience: int = 10,
                     random_state: int = 42, verbose: bool = False) -> None:
            self.task_type = task_type
            self.hidden_channels = hidden_channels or [32, 64]
            self.kernel_size = kernel_size
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.early_stopping_patience = early_stopping_patience
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
            self.label_encoder_: Optional[LabelEncoder] = None
            self.input_dim_: Optional[int] = None
            self.output_dim_: Optional[int] = None
            self.classes_: Optional[np.ndarray] = None
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TorchCNN1D':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
                self.classes_ = self.label_encoder_.classes_
                self.output_dim_ = len(self.classes_)
            else:
                y_np = np.array(y).astype(np.float32).reshape(-1, 1)
                self.output_dim_ = 1
            self.input_dim_ = X_np.shape[1]
            X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.15, random_state=self.random_state)
            train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            self.model_ = _CNN1DNet(self.input_dim_, self.output_dim_, self.hidden_channels, self.kernel_size, self.dropout, self.task_type).to(self.device)
            criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(self.epochs):
                self.model_.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model_(Xb)
                    loss = criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb)
                    loss.backward()
                    optimizer.step()
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        outputs = self.model_(Xb)
                        val_loss += criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb).item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
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

    class _LSTMNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int,
                     hidden_dim: int = 64, num_layers: int = 2,
                     dropout: float = 0.3, bidirectional: bool = False,
                     task_type: str = 'classification') -> None:
            super().__init__()
            self.task_type = task_type
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                bidirectional=bidirectional)
            lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
            self.fc = nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]
            return self.fc(x)

    class TorchLSTM(BaseEstimator):
        """LSTM，统一sklearn接口"""
        def __init__(self, task_type: str = 'classification', hidden_dim: int = 64,
                     num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = False,
                     lr: float = 0.001, epochs: int = 50, batch_size: int = 64,
                     early_stopping_patience: int = 10, random_state: int = 42,
                     verbose: bool = False) -> None:
            self.task_type = task_type
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.bidirectional = bidirectional
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.early_stopping_patience = early_stopping_patience
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
            self.label_encoder_: Optional[LabelEncoder] = None
            self.input_dim_: Optional[int] = None
            self.output_dim_: Optional[int] = None
            self.classes_: Optional[np.ndarray] = None
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TorchLSTM':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
                self.classes_ = self.label_encoder_.classes_
                self.output_dim_ = len(self.classes_)
            else:
                y_np = np.array(y).astype(np.float32).reshape(-1, 1)
                self.output_dim_ = 1
            self.input_dim_ = X_np.shape[1]
            X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.15, random_state=self.random_state)
            X_tr_seq = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
            X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            self.model_ = _LSTMNet(self.input_dim_, self.output_dim_, self.hidden_dim, self.num_layers, self.dropout, self.bidirectional, self.task_type).to(self.device)
            criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(self.epochs):
                self.model_.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model_(Xb)
                    loss = criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb)
                    loss.backward()
                    optimizer.step()
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        outputs = self.model_(Xb)
                        val_loss += criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb).item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
            return self
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            X_seq = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_seq).to(self.device))
                if self.task_type == 'classification':
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    return self.label_encoder_.inverse_transform(preds)
                return outputs.squeeze().cpu().numpy()
        
        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.task_type != 'classification':
                raise ValueError('only classification supports predict_proba')
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            X_seq = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_seq).to(self.device))
                return torch.softmax(outputs, dim=1).cpu().numpy()

    class _GRUNet(nn.Module):
        """GRU网络"""
        def __init__(self, input_dim: int, output_dim: int,
                     hidden_dim: int = 64, num_layers: int = 2,
                     dropout: float = 0.3, bidirectional: bool = False,
                     task_type: str = 'classification') -> None:
            super().__init__()
            self.task_type = task_type
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
            gru_out_dim = hidden_dim * (2 if bidirectional else 1)
            self.fc = nn.Sequential(
                nn.Linear(gru_out_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gru_out, _ = self.gru(x)
            x = gru_out[:, -1, :]
            return self.fc(x)

    class TorchGRU(BaseEstimator):
        """GRU，统一sklearn接口（与TorchLSTM结构相同，使用GRU替代LSTM）"""
        def __init__(self, task_type: str = 'classification', hidden_dim: int = 64,
                     num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = False,
                     lr: float = 0.001, epochs: int = 50, batch_size: int = 64,
                     early_stopping_patience: int = 10, random_state: int = 42,
                     verbose: bool = False) -> None:
            self.task_type = task_type
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.bidirectional = bidirectional
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.early_stopping_patience = early_stopping_patience
            self.random_state = random_state
            self.verbose = verbose
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
            self.label_encoder_: Optional[LabelEncoder] = None
            self.input_dim_: Optional[int] = None
            self.output_dim_: Optional[int] = None
            self.classes_: Optional[np.ndarray] = None
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TorchGRU':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
                self.classes_ = self.label_encoder_.classes_
                self.output_dim_ = len(self.classes_)
            else:
                y_np = np.array(y).astype(np.float32).reshape(-1, 1)
                self.output_dim_ = 1
            self.input_dim_ = X_np.shape[1]
            X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.15, random_state=self.random_state)
            X_tr_seq = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
            X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            self.model_ = _GRUNet(self.input_dim_, self.output_dim_, self.hidden_dim, self.num_layers, self.dropout, self.bidirectional, self.task_type).to(self.device)
            criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(self.epochs):
                self.model_.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model_(Xb)
                    loss = criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb)
                    loss.backward()
                    optimizer.step()
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        outputs = self.model_(Xb)
                        val_loss += criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb).item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
            return self
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            X_seq = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_seq).to(self.device))
                if self.task_type == 'classification':
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    return self.label_encoder_.inverse_transform(preds)
                return outputs.squeeze().cpu().numpy()
        
        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.task_type != 'classification':
                raise ValueError('only classification supports predict_proba')
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            X_seq = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_seq).to(self.device))
                return torch.softmax(outputs, dim=1).cpu().numpy()

    class _AutoEncoderNet(nn.Module):
        def __init__(self, input_dim: int, encoding_dim: int = 16,
                     hidden_dims: Optional[List[int]] = None) -> None:
            if hidden_dims is None:
                hidden_dims = [64, 32]
            super().__init__()
            enc_layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            enc_layers.append(nn.Linear(prev, encoding_dim))
            self.encoder = nn.Sequential(*enc_layers)
            dec_layers: List[nn.Module] = []
            prev = encoding_dim
            for h in reversed(hidden_dims):
                dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            dec_layers.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*dec_layers)
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded, encoded

    class TorchAutoEncoder(BaseEstimator):
        """自编码器，用于特征降维"""
        def __init__(self, encoding_dim: int = 16,
                     hidden_dims: Optional[List[int]] = None,
                     lr: float = 0.001, epochs: int = 50, batch_size: int = 64,
                     random_state: int = 42, verbose: bool = False) -> None:
            self.encoding_dim = encoding_dim
            self.hidden_dims = hidden_dims or [64, 32]
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'TorchAutoEncoder':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            input_dim = X_np.shape[1]
            dataset = TensorDataset(torch.tensor(X_np))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model_ = _AutoEncoderNet(input_dim, self.encoding_dim, self.hidden_dims).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            for epoch in range(self.epochs):
                self.model_.train()
                for (Xb,) in loader:
                    Xb = Xb.to(self.device)
                    optimizer.zero_grad()
                    decoded, _ = self.model_(Xb)
                    loss = criterion(decoded, Xb)
                    loss.backward()
                    optimizer.step()
            return self
        
        def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            self.model_.eval()
            with torch.no_grad():
                _, encoded = self.model_(torch.tensor(X_np).to(self.device))
                return encoded.cpu().numpy()
        
        def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                          y: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
            self.fit(X)
            return self.transform(X)
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            self.model_.eval()
            with torch.no_grad():
                decoded, _ = self.model_(torch.tensor(X_np).to(self.device))
                mse = torch.mean((decoded - torch.tensor(X_np).to(self.device)) ** 2, dim=1)
                return mse.cpu().numpy()


# =============================================================================
# TabNet 包装器（条件定义）
# =============================================================================

if TORCH_AVAILABLE and TABNET_AVAILABLE:
    
    class TabNetWrapper(BaseEstimator):
        """
        TabNet 包装器（统一 sklearn 接口）
        
        TabNet 是专为表格数据设计的深度学习架构，
        具有内置的特征选择能力。
        """
        
        def __init__(self,
                     task_type: str = 'classification',
                     n_d: int = 8,
                     n_a: int = 8,
                     n_steps: int = 3,
                     gamma: float = 1.3,
                     lambda_sparse: float = 1e-4,
                     max_epochs: int = 100,
                     patience: int = 10,
                     batch_size: int = 256,
                     virtual_batch_size: int = 128,
                     device_name: str = 'auto',
                     verbose: bool = False,
                     random_state: int = 42) -> None:
            self.task_type = task_type
            self.n_d = n_d
            self.n_a = n_a
            self.n_steps = n_steps
            self.gamma = gamma
            self.lambda_sparse = lambda_sparse
            self.max_epochs = max_epochs
            self.patience = patience
            self.batch_size = batch_size
            self.virtual_batch_size = virtual_batch_size
            self.device_name = device_name
            self.verbose = verbose
            self.random_state = random_state
            
            self.model_: Optional[Any] = None
            self.label_encoder_: Optional[LabelEncoder] = None
            self.classes_: Optional[np.ndarray] = None
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TabNetWrapper':
            np.random.seed(self.random_state)
            
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_enc = self.label_encoder_.fit_transform(y)
                self.classes_ = self.label_encoder_.classes_
                
                self.model_ = TabNetClassifier(
                    n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
                    gamma=self.gamma, lambda_sparse=self.lambda_sparse,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    scheduler_params={"step_size": 50, "gamma": 0.9},
                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                    mask_type='entmax',
                    verbose=self.verbose,
                    device_name=self.device_name,
                    seed=self.random_state
                )
                self.model_.fit(
                    X.values if hasattr(X, 'values') else np.array(X),
                    y_enc,
                    eval_set=[(X.values if hasattr(X, 'values') else np.array(X), y_enc)],
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    batch_size=self.batch_size,
                    virtual_batch_size=self.virtual_batch_size
                )
            else:
                self.model_ = TabNetRegressor(
                    n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
                    gamma=self.gamma, lambda_sparse=self.lambda_sparse,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    scheduler_params={"step_size": 50, "gamma": 0.9},
                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                    mask_type='entmax',
                    verbose=self.verbose,
                    device_name=self.device_name,
                    seed=self.random_state
                )
                self.model_.fit(
                    X.values if hasattr(X, 'values') else np.array(X),
                    y.values if hasattr(y, 'values') else np.array(y).reshape(-1, 1),
                    eval_set=[(X.values if hasattr(X, 'values') else np.array(X), 
                              y.values if hasattr(y, 'values') else np.array(y).reshape(-1, 1))],
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    batch_size=self.batch_size,
                    virtual_batch_size=self.virtual_batch_size
                )
            
            return self
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.model_ is None:
                raise ValueError("模型未训练")
            
            X_np = X.values if hasattr(X, 'values') else np.array(X)
            preds = self.model_.predict(X_np)
            
            if self.task_type == 'classification':
                preds = preds.flatten().astype(int)
                if self.label_encoder_ is not None:
                    preds = self.label_encoder_.inverse_transform(preds)
            
            return preds.flatten()
        
        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.task_type != 'classification':
                raise ValueError("只有分类任务支持 predict_proba")
            
            X_np = X.values if hasattr(X, 'values') else np.array(X)
            return self.model_.predict_proba(X_np)


    # ===================================================================
    # TorchCNN1D: 1D卷积神经网络（适用于序列/表格数据）
    # ===================================================================
    class _CNN1DNet(nn.Module):
        """1D CNN网络"""
        def __init__(self, input_dim: int, output_dim: int,
                     hidden_channels: Optional[List[int]] = None,
                     kernel_size: int = 3,
                     dropout: float = 0.3,
                     task_type: str = 'classification') -> None:
            if hidden_channels is None:
                hidden_channels = [32, 64]
            super().__init__()
            self.task_type = task_type
            layers: List[nn.Module] = []
            in_ch = 1
            for out_ch in hidden_channels:
                layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout / 2),
                ])
                in_ch = out_ch
            self.conv = nn.Sequential(*layers)
            # 计算展平后的维度（假设输入长度=input_dim）
            self.flatten_dim = self._get_flatten_dim(input_dim, hidden_channels, kernel_size)
            self.fc = nn.Sequential(
                nn.Linear(self.flatten_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
            )
        
        def _get_flatten_dim(self, input_dim: int, hidden_channels: List[int],
                             kernel_size: int) -> int:
            with torch.no_grad():
                x = torch.zeros(1, 1, input_dim)
                x = self.conv(x)
                return x.view(1, -1).shape[1]
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)


    class TorchCNN1D(BaseEstimator):
        """1D CNN，统一sklearn接口"""
        def __init__(self, task_type: str = 'classification',
                     hidden_channels: Optional[List[int]] = None,
                     kernel_size: int = 3, dropout: float = 0.3, lr: float = 0.001,
                     epochs: int = 50, batch_size: int = 64,
                     early_stopping_patience: int = 10,
                     random_state: int = 42, verbose: bool = False) -> None:
            self.task_type = task_type
            self.hidden_channels = hidden_channels or [32, 64]
            self.kernel_size = kernel_size
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.early_stopping_patience = early_stopping_patience
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
            self.label_encoder_: Optional[LabelEncoder] = None
            self.input_dim_: Optional[int] = None
            self.output_dim_: Optional[int] = None
            self.classes_: Optional[np.ndarray] = None
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TorchCNN1D':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
                self.classes_ = self.label_encoder_.classes_
                self.output_dim_ = len(self.classes_)
            else:
                y_np = np.array(y).astype(np.float32).reshape(-1, 1)
                self.output_dim_ = 1
            self.input_dim_ = X_np.shape[1]
            
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np, y_np, test_size=0.15, random_state=self.random_state
            )
            train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            
            self.model_ = _CNN1DNet(
                self.input_dim_, self.output_dim_,
                self.hidden_channels, self.kernel_size, self.dropout, self.task_type
            ).to(self.device)
            
            criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                self.model_.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model_(Xb)
                    loss = criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb)
                    loss.backward()
                    optimizer.step()
                
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        outputs = self.model_(Xb)
                        val_loss += criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb).item()
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
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
                raise ValueError('只有分类任务支持predict_proba')
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_np).to(self.device))
                return torch.softmax(outputs, dim=1).cpu().numpy()


    # ===================================================================
    # TorchLSTM: 长短期记忆网络（适用于序列/时序数据）
    # ===================================================================
    class _LSTMNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int,
                     hidden_dim: int = 64, num_layers: int = 2,
                     dropout: float = 0.3, bidirectional: bool = False,
                     task_type: str = 'classification') -> None:
            super().__init__()
            self.task_type = task_type
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                bidirectional=bidirectional)
            lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
            self.fc = nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]  # 取最后一个时间步
            return self.fc(x)


    class TorchLSTM(BaseEstimator):
        """LSTM，统一sklearn接口（将表格数据视为单步序列）"""
        def __init__(self, task_type: str = 'classification', hidden_dim: int = 64,
                     num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = False,
                     lr: float = 0.001, epochs: int = 50, batch_size: int = 64,
                     early_stopping_patience: int = 10, random_state: int = 42,
                     verbose: bool = False) -> None:
            self.task_type = task_type
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.bidirectional = bidirectional
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.early_stopping_patience = early_stopping_patience
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
            self.label_encoder_: Optional[LabelEncoder] = None
            self.input_dim_: Optional[int] = None
            self.output_dim_: Optional[int] = None
            self.classes_: Optional[np.ndarray] = None
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray]) -> 'TorchLSTM':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            if self.task_type == 'classification':
                self.label_encoder_ = LabelEncoder()
                y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
                self.classes_ = self.label_encoder_.classes_
                self.output_dim_ = len(self.classes_)
            else:
                y_np = np.array(y).astype(np.float32).reshape(-1, 1)
                self.output_dim_ = 1
            self.input_dim_ = X_np.shape[1]
            
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np, y_np, test_size=0.15, random_state=self.random_state
            )
            # 将表格数据reshape为序列 (batch, seq_len=1, features)
            X_tr_seq = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
            X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            
            self.model_ = _LSTMNet(
                self.input_dim_, self.output_dim_, self.hidden_dim,
                self.num_layers, self.dropout, self.bidirectional, self.task_type
            ).to(self.device)
            
            criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                self.model_.train()
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model_(Xb)
                    loss = criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb)
                    loss.backward()
                    optimizer.step()
                
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        outputs = self.model_(Xb)
                        val_loss += criterion(outputs.squeeze(), yb.float() if self.task_type != 'classification' else yb).item()
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
            return self
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            X_seq = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_seq).to(self.device))
                if self.task_type == 'classification':
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    return self.label_encoder_.inverse_transform(preds)
                return outputs.squeeze().cpu().numpy()
        
        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.task_type != 'classification':
                raise ValueError('只有分类任务支持predict_proba')
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            X_seq = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(torch.tensor(X_seq).to(self.device))
                return torch.softmax(outputs, dim=1).cpu().numpy()


    # ===================================================================
    # TorchAutoEncoder: 自编码器（用于特征降维/异常检测）
    # ===================================================================
    class _AutoEncoderNet(nn.Module):
        def __init__(self, input_dim: int, encoding_dim: int = 16,
                     hidden_dims: Optional[List[int]] = None) -> None:
            if hidden_dims is None:
                hidden_dims = [64, 32]
            super().__init__()
            # 编码器
            enc_layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            enc_layers.append(nn.Linear(prev, encoding_dim))
            self.encoder = nn.Sequential(*enc_layers)
            # 解码器
            dec_layers: List[nn.Module] = []
            prev = encoding_dim
            for h in reversed(hidden_dims):
                dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            dec_layers.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*dec_layers)
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded, encoded


    class TorchAutoEncoder(BaseEstimator):
        """自编码器，用于特征降维（fit_transform返回编码特征）"""
        def __init__(self, encoding_dim: int = 16,
                     hidden_dims: Optional[List[int]] = None,
                     lr: float = 0.001, epochs: int = 50, batch_size: int = 64,
                     random_state: int = 42, verbose: bool = False) -> None:
            self.encoding_dim = encoding_dim
            self.hidden_dims = hidden_dims or [64, 32]
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_: Optional[nn.Module] = None
            self.scaler_ = StandardScaler()
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray],
                y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'TorchAutoEncoder':
            torch.manual_seed(self.random_state)
            X_np = self.scaler_.fit_transform(np.array(X)).astype(np.float32)
            input_dim = X_np.shape[1]
            
            dataset = TensorDataset(torch.tensor(X_np))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model_ = _AutoEncoderNet(input_dim, self.encoding_dim, self.hidden_dims).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
            
            for epoch in range(self.epochs):
                self.model_.train()
                for (Xb,) in loader:
                    Xb = Xb.to(self.device)
                    optimizer.zero_grad()
                    decoded, _ = self.model_(Xb)
                    loss = criterion(decoded, Xb)
                    loss.backward()
                    optimizer.step()
            return self
        
        def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            self.model_.eval()
            with torch.no_grad():
                _, encoded = self.model_(torch.tensor(X_np).to(self.device))
                return encoded.cpu().numpy()
        
        def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                          y: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
            self.fit(X)
            return self.transform(X)
        
        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            # 返回重构误差，可用于异常检测
            X_np = self.scaler_.transform(np.array(X)).astype(np.float32)
            self.model_.eval()
            with torch.no_grad():
                decoded, _ = self.model_(torch.tensor(X_np).to(self.device))
                mse = torch.mean((decoded - torch.tensor(X_np).to(self.device)) ** 2, dim=1)
                return mse.cpu().numpy()


# =============================================================================
# 深度学习模型库集成
# =============================================================================

def register_deep_learning_models() -> None:
    """
    将深度学习模型注册到 ModelLibrary
    
    在 modeling_engine.py 的 ModelLibrary 初始化后调用
    """
    if TORCH_AVAILABLE:
        ModelLibrary._register(
            'classification', 'torch_mlp', 'PyTorch-MLP',
            TorchMLP, 'neural',
            default_params={'task_type': 'classification', 'epochs': 100, 'verbose': False},
            hyperparam_space={
                'hidden_dims': [(64,), (128,), (64, 64), (128, 64)],
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
                'batch_size': {'type': 'int', 'low': 16, 'high': 256, 'step': 16},
                'epochs': {'type': 'int', 'low': 10, 'high': 100},
            },
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'torch_mlp', 'PyTorch-MLP',
            TorchMLP, 'neural',
            default_params={'task_type': 'regression', 'epochs': 100, 'verbose': False},
            hyperparam_space={
                'hidden_dims': [(64,), (128,), (64, 64), (128, 64)],
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
                'batch_size': {'type': 'int', 'low': 16, 'high': 256, 'step': 16},
                'epochs': {'type': 'int', 'low': 10, 'high': 100},
            },
            supports_gpu=True
        )
        log_info("[DeepLearning] PyTorch MLP 已注册")
        
        # 注册 CNN1D
        ModelLibrary._register(
            'classification', 'torch_cnn1d', 'PyTorch-CNN1D',
            TorchCNN1D, 'neural',
            default_params={'task_type': 'classification', 'epochs': 50, 'verbose': False},
            hyperparam_space={
                'hidden_channels': [[32, 64], [64, 128]],
                'kernel_size': [3, 5],
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            },
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'torch_cnn1d', 'PyTorch-CNN1D',
            TorchCNN1D, 'neural',
            default_params={'task_type': 'regression', 'epochs': 50, 'verbose': False},
            hyperparam_space={
                'hidden_channels': [[32, 64], [64, 128]],
                'kernel_size': [3, 5],
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            },
            supports_gpu=True
        )
        log_info("[DeepLearning] PyTorch CNN1D 已注册")
        
        # 注册 LSTM
        ModelLibrary._register(
            'classification', 'torch_lstm', 'PyTorch-LSTM',
            TorchLSTM, 'neural',
            default_params={'task_type': 'classification', 'epochs': 50, 'verbose': False},
            hyperparam_space={
                'hidden_dim': {'type': 'int', 'low': 16, 'high': 256, 'step': 16},
                'num_layers': {'type': 'int', 'low': 1, 'high': 3},
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            },
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'torch_lstm', 'PyTorch-LSTM',
            TorchLSTM, 'neural',
            default_params={'task_type': 'regression', 'epochs': 50, 'verbose': False},
            hyperparam_space={
                'hidden_dim': {'type': 'int', 'low': 16, 'high': 256, 'step': 16},
                'num_layers': {'type': 'int', 'low': 1, 'high': 3},
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            },
            supports_gpu=True
        )
        log_info("[DeepLearning] PyTorch LSTM 已注册")
        
        # 注册 GRU
        ModelLibrary._register(
            'classification', 'torch_gru', 'PyTorch-GRU',
            TorchGRU, 'neural',
            default_params={'task_type': 'classification', 'epochs': 50, 'verbose': False},
            hyperparam_space={
                'hidden_dim': {'type': 'int', 'low': 16, 'high': 256, 'step': 16},
                'num_layers': {'type': 'int', 'low': 1, 'high': 3},
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            },
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'torch_gru', 'PyTorch-GRU',
            TorchGRU, 'neural',
            default_params={'task_type': 'regression', 'epochs': 50, 'verbose': False},
            hyperparam_space={
                'hidden_dim': {'type': 'int', 'low': 16, 'high': 256, 'step': 16},
                'num_layers': {'type': 'int', 'low': 1, 'high': 3},
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
                'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            },
            supports_gpu=True
        )
        log_info("[DeepLearning] PyTorch GRU 已注册")
        
        # 注册 NAS
        try:
            from core.nas import TorchNAS
            ModelLibrary._register(
                'classification', 'torch_nas', 'PyTorch-NAS',
                TorchNAS, 'neural',
                default_params={'task_type': 'classification', 'n_candidates': 6, 'epochs': 20, 'verbose': False},
                supports_gpu=True
            )
            ModelLibrary._register(
                'regression', 'torch_nas', 'PyTorch-NAS',
                TorchNAS, 'neural',
                default_params={'task_type': 'regression', 'n_candidates': 6, 'epochs': 20, 'verbose': False},
                supports_gpu=True
            )
            log_info("[DeepLearning] PyTorch NAS 已注册")
        except Exception as e:
            log_warning(f"[DeepLearning] NAS 注册失败: {e}")
    
    if TORCH_AVAILABLE and TABNET_AVAILABLE:
        ModelLibrary._register(
            'classification', 'tabnet', 'TabNet',
            TabNetWrapper, 'neural',
            default_params={'task_type': 'classification', 'max_epochs': 100, 'verbose': False},
            hyperparam_space={
                'n_d': {'type': 'int', 'low': 4, 'high': 64},
                'n_a': {'type': 'int', 'low': 4, 'high': 64},
                'n_steps': {'type': 'int', 'low': 2, 'high': 8},
                'gamma': {'type': 'float', 'low': 0.8, 'high': 2.0},
            },
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'tabnet', 'TabNet',
            TabNetWrapper, 'neural',
            default_params={'task_type': 'regression', 'max_epochs': 100, 'verbose': False},
            hyperparam_space={
                'n_d': [8, 16, 32],
                'n_a': [8, 16, 32],
                'n_steps': [3, 5],
                'gamma': [1.0, 1.3, 1.5],
            },
            supports_gpu=True
        )
        log_info("[DeepLearning] TabNet 已注册")
    
    # 注册多模态模型（图像/文本）
    try:
        from core.multimodal import ImageResNet, TextBERT
        ModelLibrary._register(
            'classification', 'image_resnet', 'Image-ResNet',
            ImageResNet, 'multimodal',
            default_params={'task_type': 'classification', 'image_col': 'image_path', 'epochs': 10, 'freeze_backbone': True},
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'image_resnet', 'Image-ResNet',
            ImageResNet, 'multimodal',
            default_params={'task_type': 'regression', 'image_col': 'image_path', 'epochs': 10, 'freeze_backbone': True},
            supports_gpu=True
        )
        log_info("[DeepLearning] Image-ResNet 已注册")
        
        ModelLibrary._register(
            'classification', 'text_bert', 'Text-BERT',
            TextBERT, 'multimodal',
            default_params={'task_type': 'classification', 'text_col': 'text', 'epochs': 3, 'freeze_backbone': True},
            supports_gpu=True
        )
        ModelLibrary._register(
            'regression', 'text_bert', 'Text-BERT',
            TextBERT, 'multimodal',
            default_params={'task_type': 'regression', 'text_col': 'text', 'epochs': 3, 'freeze_backbone': True},
            supports_gpu=True
        )
        log_info("[DeepLearning] Text-BERT 已注册")
    except Exception as e:
        log_warning(f"[DeepLearning] 多模态模型注册失败: {e}")


# 自动注册
try:
    register_deep_learning_models()
except Exception as e:
    log_warning(f"[DeepLearning] 模型注册失败: {e}")
