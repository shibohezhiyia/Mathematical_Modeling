"""
强化学习超参数优化器 (DQN-based)

使用深度Q网络进行自适应超参数搜索。
将超参优化建模为 MDP：
  - 状态: 数据集元特征 + 当前超参配置(归一化) + 历史信息
  - 动作: 选择具体参数-值对 (param_name → candidate_value)
  - 奖励: CV 分数的 delta 改进 + 突破最优 bonus

改进点 (v2):
  1. 动作空间: action_dim = sum(len(candidates))，直接映射到参数-值对
  2. 状态表示: 分类参数用归一化索引，连续参数归一化，加入最近score/进度/动作
  3. 奖励函数: delta-based + 突破 bonus，鼓励持续改进
  4. 评估加速: 子集采样渐进策略 + 评估缓存 + 参数级 early stop

继承 BaseOptimizer 统一接口。
"""

import random
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from core.modeling_engine import ModelLibrary, TaskType, TaskTypeDetector
from core.optimizer_base import BaseOptimizer, OptimizationResult
from core.progress_bar import progress_range
from utils.helpers import log_info, log_warning

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# 元特征提取工具
# =============================================================================

def extract_meta_features(X: pd.DataFrame, y: Optional[pd.Series], task_type: TaskType) -> np.ndarray:
    """提取数据集元特征向量 (8维)
    
    优化：使用向量化操作替代 Python 循环，减少 dtype 检查开销。
    """
    n_samples, n_features = X.shape
    # 向量化：使用 select_dtypes 替代逐列 dtype 检查
    n_numeric = len(X.select_dtypes(include=[np.number]).columns)
    n_categorical = n_features - n_numeric
    
    features = [
        np.log1p(n_samples),
        np.log1p(n_features),
        n_numeric / max(n_features, 1),
        n_categorical / max(n_features, 1),
    ]
    
    # 向量化：使用 values 一次性计算缺失率，避免 sum().sum() 双重遍历
    missing_ratio = np.isnan(X.values).sum() / max(X.size, 1) if X.size > 0 else 0.0
    features.append(missing_ratio)
    
    # 向量化：使用 select_dtypes 筛选数值列，一次性计算零值比例
    sparsity = 0.0
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_values = X[numeric_cols].values
        sparsity = (numeric_values == 0).sum() / max(numeric_values.size, 1)
    features.append(sparsity)
    
    if y is not None and task_type == TaskType.CLASSIFICATION:
        class_counts = pd.Series(y).value_counts()
        imbalance = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1.0
        n_classes = len(class_counts)
        features.extend([np.log1p(n_classes), np.log1p(imbalance)])
    elif y is not None:
        y_std = np.std(y)
        features.extend([np.log1p(y_std), 0.0])
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# DQN 网络与回放缓冲区
# =============================================================================

class _DQN(nn.Module):
    """三层 MLP + Dropout，支持更大状态空间"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: Any) -> Any:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: List[Tuple] = []
        self.capacity = capacity
    
    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# 动作空间管理: 将 action_idx 映射到 (param_name, candidate_value)
# =============================================================================

class _ParamActionSpace:
    """
    扁平化动作空间。
    action_dim = sum(len(candidates[p]) for p in param_names)
    每个动作直接对应: 设置某个参数为某个候选值。
    """
    def __init__(self, param_candidates: Dict[str, List]) -> None:
        self.param_candidates = param_candidates
        self.param_names = list(param_candidates.keys())
        self._param_count = len(self.param_names)  # cache count for repeated use
        
        # 扁平动作列表: [(param_name, candidate_value, param_idx, cand_idx), ...]
        self.actions = []
        self.param_offsets = {}  # param_name -> start index in flat actions
        offset = 0
        for pname in self.param_names:
            self.param_offsets[pname] = offset
            for cidx, val in enumerate(param_candidates[pname]):
                self.actions.append((pname, val, cidx))
                offset += 1
        self.action_dim = len(self.actions)
    
    def decode(self, action_idx: int) -> Tuple[str, Any, int]:
        """返回 (param_name, candidate_value, candidate_index)"""
        return self.actions[action_idx]
    
    def apply(self, current_params: Dict, action_idx: int, search_space: Dict) -> Dict:
        """应用动作到当前参数配置"""
        params = deepcopy(current_params)
        pname, val, _ = self.decode(action_idx)
        spec = search_space.get(pname, {})
        if isinstance(spec, dict) and spec.get('type') == 'int':
            val = int(round(val))
        params[pname] = val
        return params
    
    def get_action_index(self, param_name: str, candidate_index: int) -> int:
        """根据参数名和候选索引获取 flat action index"""
        return self.param_offsets[param_name] + candidate_index


# =============================================================================
# 评估缓存
# =============================================================================

class _EvaluationCache:
    """缓存参数-子集比例的评估结果，避免重复计算"""
    def __init__(self) -> None:
        self._cache: Dict[str, float] = {}
    
    @staticmethod
    def _make_key(params: Dict, subset_fraction: float) -> str:
        """生成确定性哈希 key"""
        # 排序后 JSON 序列化
        serializable = {}
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif hasattr(v, 'item'):  # numpy scalar
                v = v.item()
            serializable[k] = v
        json_str = json.dumps(serializable, sort_keys=True, default=str)
        h = hashlib.md5(json_str.encode()).hexdigest()[:16]
        return f"{h}:{subset_fraction:.2f}"
    
    def get(self, params: Dict, subset_fraction: float) -> Optional[float]:
        return self._cache.get(self._make_key(params, subset_fraction), None)
    
    def set(self, params: Dict, subset_fraction: float, score: float) -> None:
        self._cache[self._make_key(params, subset_fraction)] = score


# =============================================================================
# RLOptimizer v2
# =============================================================================

class RLOptimizer(BaseOptimizer):
    """
    基于 DQN 的强化学习超参数优化器 (v2)
    
    改进:
      - 扁平动作空间: 每个动作直接修改一个参数的一个候选值
      - 丰富状态表示: 分类参数索引+连续归一化+历史信息
      - Delta reward: 鼓励持续改进而非只奖励突破最优
      - 子集采样: 前期用较少数据快速筛选，后期用全量精调
      - 评估缓存: 相同参数不重复评估
    """
    
    def __init__(self,
                 n_trials: int = 50,
                 cv_folds: int = 3,
                 random_state: int = 42,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.97,
                 buffer_capacity: int = 5000,
                 batch_size: int = 32,
                 target_update_freq: int = 10,
                 subset_schedule: Optional[List[Tuple[float, float]]] = None,
                 hidden_dim: int = 128,
                 parallel_eval: bool = False,
                 n_parallel: int = 2,
                 **kwargs) -> None:
        """
        Args:
            subset_schedule: 子集采样计划，默认 [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
                             表示前 30% trial 用 30% 数据，中间 30% 用 60% 数据，最后 40% 用全量
        """
        super().__init__(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state,
                         verbose=kwargs.get('verbose', True),
                         trial_timeout=kwargs.get('trial_timeout', 120))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_dim = hidden_dim
        
        # 子集采样计划: [(trial_ratio_start, subset_fraction), ...]
        if subset_schedule is None:
            subset_schedule = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
        self.subset_schedule = subset_schedule
        
        self._torch_available = TORCH_AVAILABLE
        self._device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self._buffer = ReplayBuffer(capacity=buffer_capacity)
        self.parallel_eval = parallel_eval
        self.n_parallel = n_parallel
        
        if kwargs:
            log_info(f"[RLOptimizer] 忽略未识别的构造参数: {list(kwargs.keys())}")
        
        if not TORCH_AVAILABLE:
            log_warning("[RLOptimizer] PyTorch 未安装，optimizer='rl' 将回退到随机搜索。"
                        "如需启用 RL，请安装 PyTorch: pip install torch")
    
    def optimize(self,
                 model_key: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: Union[str, TaskType],
                 metric: Optional[str] = None,
                 custom_search_space: Optional[Dict] = None) -> OptimizationResult:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        search_space = self._get_search_space(model_key, task_type, custom_search_space)
        models = ModelLibrary.get_models(task_type, [model_key])
        spec = models[model_key]
        
        if not search_space:
            return OptimizationResult(
                model_key=model_key,
                best_params=deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                sampler_type='rl_none'
            )
        
        start_time = time.time()
        
        if metric is None:
            metric = TaskTypeDetector.get_primary_metric(task_type)
        
        meta_features = extract_meta_features(X, y, task_type)
        param_candidates = self._build_candidates(search_space, n_candidates=8)
        
        if not self._torch_available:
            log_warning(f"[RLOptimizer] {spec.name}: PyTorch 不可用，"
                        f"回退到随机搜索 (sampler_type='rl_torch_unavailable')")
            result = self._random_search_fallback(model_key, spec, search_space, X, y, task_type, metric)
            result.optimize_time = time.time() - start_time
            result.sampler_type = 'rl_torch_unavailable'
            return result
        
        # 初始化动作空间
        action_space = _ParamActionSpace(param_candidates)
        action_dim = action_space.action_dim
        
        if action_dim == 0:
            return OptimizationResult(
                model_key=model_key,
                best_params=deepcopy(spec.default_params),
                best_score=0.0,
                n_trials=0,
                sampler_type='rl_no_actions'
            )
        
        # 状态维度 = 元特征 + 参数归一化值 + 历史信息(8维)
        #   last_score, progress, last_action, epsilon, score_trend, score_std,
        #   exploration_ratio, param_coverage_avg
        state_dim = len(meta_features) + len(param_candidates) + 8
        
        policy_net = _DQN(state_dim, action_dim, hidden_dim=self.hidden_dim).to(self._device)
        target_net = _DQN(state_dim, action_dim, hidden_dim=self.hidden_dim).to(self._device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        optimizer = optim.Adam(policy_net.parameters(), lr=self.lr)
        
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
        
        cache = _EvaluationCache()
        best_score = float('-inf')
        best_params = deepcopy(spec.default_params)
        history = []
        
        # 初始化随机配置
        current_params = self._random_params(search_space)
        last_score = 0.0
        last_action_idx = 0
        
        # 计算初始状态的 trial 进度和 epsilon
        trial_progress = 0.0
        epsilon = self.epsilon_start
        state = self._build_state(meta_features, current_params, param_candidates,
                                   last_score, trial_progress, last_action_idx, epsilon)
        
        log_info(f"[RLOptimizer] 启动 RL 优化: {spec.name}, trials={self.n_trials}, "
                 f"action_dim={action_dim}, state_dim={state_dim}, device={self._device}, metric={metric}")
        log_info(f"[RLOptimizer] 初始配置: {current_params}")
        if param_candidates:
            cand_summary = {k: len(v) for k, v in param_candidates.items()}
            log_info(f"[RLOptimizer] 候选空间大小: {cand_summary}")
        
        # 维护移动平均分数用于 reward baseline
        score_window = []
        window_size = max(5, self.n_trials // 10)
        
        # 追踪配置多样性: 已尝试的 (param_name, value) 对
        param_value_tried: Dict[str, set] = {p: set() for p in param_candidates}
        unique_configs: set = set()
        last_param_changed: Optional[str] = None
        same_param_streak: int = 0
        
        for trial in progress_range(self.n_trials, desc=f"RL优化 {model_key}", disable=not self.verbose):
            trial_progress = trial / max(self.n_trials - 1, 1)
            
            # 动态 epsilon: 前期多探索，后期利用，带小幅随机扰动
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-5.0 * trial_progress)
            
            # epsilon-greedy 动作选择
            if random.random() < epsilon:
                action_idx = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
                    q_values = policy_net(state_t)
                    action_idx = q_values.argmax(dim=1).item()
            
            # 应用动作得到新配置
            next_params = action_space.apply(current_params, action_idx, search_space)
            
            # 确定子集比例
            subset_fraction = self._get_subset_fraction(trial_progress)
            
            # 评估（带缓存和子集采样）
            try:
                score = self._evaluate_with_cache(
                    model_key, next_params, X, y, task_type, metric,
                    cache, subset_fraction
                )
            except Exception as e:
                log_warning(f"[RLOptimizer] 评估失败 trial={trial+1}: {e}")
                score = best_score - 1.0
            
            # --- 统计信息更新 ---
            score_window.append(score)
            if len(score_window) > window_size:
                score_window.pop(0)
            
            # 计算 score 趋势和标准差
            score_trend = 0.0
            score_std = 0.0
            if len(score_window) >= 3:
                x = np.arange(len(score_window))
                y_scores = np.array(score_window)
                score_trend = float(np.polyfit(x, y_scores, 1)[0])
                score_std = float(np.std(y_scores))
            
            # 追踪参数覆盖率和配置多样性
            config_key = tuple(sorted((k, str(v)) for k, v in next_params.items()))
            is_new_config = config_key not in unique_configs
            if is_new_config:
                unique_configs.add(config_key)
            
            action_param_name, action_val, _ = action_space.decode(action_idx)
            is_new_param_value = action_val not in param_value_tried.get(action_param_name, set())
            if is_new_param_value:
                param_value_tried[action_param_name].add(action_val)
            
            exploration_ratio = len(unique_configs) / max(trial + 1, 1)
            param_coverage_avg = np.mean([
                len(tried) / max(len(param_candidates.get(p, [])), 1)
                for p, tried in param_value_tried.items()
            ]) if param_value_tried else 0.0
            
            # 检测连续修改同一参数
            if last_param_changed == action_param_name:
                same_param_streak += 1
            else:
                same_param_streak = 1
            last_param_changed = action_param_name
            
            # --- Reward 计算 (delta-based + 多样性奖励) ---
            delta = score - last_score
            
            if score > best_score:
                reward = delta * 3.0 + 1.0
                best_score = score
                best_params = deepcopy(next_params)
            elif delta > 0:
                reward = delta * 2.0 + 0.3
            elif delta > -0.01:
                reward = -0.05
            else:
                reward = delta * 1.5 - 0.3
            
            # 多样性奖励: 新配置 +0.2, 新参数-值对 +0.1
            if is_new_config and trial > 0:
                reward += 0.2
            if is_new_param_value and trial > 0:
                reward += 0.1
            
            # 稳定性惩罚: 连续3次改同一参数 -0.1
            if same_param_streak >= 3:
                reward -= 0.1
            
            # 重复配置惩罚
            if not is_new_config and trial > 0:
                reward -= 0.05
            
            # 构建下一状态（带丰富历史信息）
            next_state = self._build_state(
                meta_features, next_params, param_candidates,
                score, trial_progress, action_idx, epsilon,
                score_trend=score_trend,
                score_std=score_std,
                exploration_ratio=exploration_ratio,
                param_coverage_avg=param_coverage_avg
            )
            done = (trial == self.n_trials - 1)
            
            # 存入回放缓冲区
            self._buffer.push(state, action_idx, reward, next_state, done)
            
            # 更新当前状态
            state = next_state
            current_params = next_params
            last_score = score
            last_action_idx = action_idx
            
            history.append({
                'trial': trial + 1,
                'params': deepcopy(next_params),
                'score': score,
                'best_so_far': best_score,
                'reward': reward,
                'epsilon': epsilon,
                'subset_fraction': subset_fraction,
                'action': action_param_name,
                'is_new_config': is_new_config,
                'exploration_ratio': exploration_ratio,
                'param_coverage': param_coverage_avg,
            })
            
            # 调试日志
            is_milestone = (trial + 1) % max(1, self.n_trials // 5) == 0
            if score > best_score or is_milestone or trial < 3:
                log_info(f"[RLOptimizer] Trial {trial+1}/{self.n_trials}: "
                         f"score={score:.4f}, reward={reward:.3f}, epsilon={epsilon:.3f}, "
                         f"action={action_param_name}={action_val}, subset={subset_fraction:.0%}, "
                         f"best={best_score:.4f}, diversity={exploration_ratio:.2f}, coverage={param_coverage_avg:.2f}")
            
            # DQN 训练
            if len(self._buffer) >= self.batch_size:
                self._train_step(policy_net, target_net, optimizer)
            
            # 目标网络更新
            if trial % self.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # --- 并行最终验证: 对 top-K 唯一配置用全量数据重新评估 ---
        if self.parallel_eval and TORCH_AVAILABLE:
            best_score, best_params = self._parallel_final_eval(
                model_key, history, X, y, task_type, metric, best_score, best_params
            )
        
        optimize_time = time.time() - start_time
        cache_hits = len(cache._cache)
        log_info(f"[RLOptimizer] {spec.name} 优化完成: best_score={best_score:.4f}, "
                 f"time={optimize_time:.1f}s, unique_evals={cache_hits}, unique_configs={len(unique_configs)}")
        log_info(f"[RLOptimizer] 最优参数: {best_params}")
        
        return OptimizationResult(
            model_key=model_key,
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=self.n_trials,
            optimize_time=optimize_time,
            sampler_type='rl_dqn_v2'
        )
    
    # -------------------------------------------------------------------------
    # 评估辅助
    # -------------------------------------------------------------------------
    
    def _evaluate_model_with_params(self, model_key: str, params: Dict, X: pd.DataFrame, y: pd.Series, task_type: TaskType, metric: Optional[str]) -> float:
        """用给定参数创建模型并评估"""
        model = ModelLibrary.create_model(model_key, task_type, **params)
        return self._evaluate_model(model, X, y, task_type, metric)
    
    def _evaluate_with_cache(self, model_key: str, params: Dict, X: pd.DataFrame, y: pd.Series, task_type: TaskType, metric: Optional[str],
                             cache: _EvaluationCache, subset_fraction: float) -> float:
        """带缓存和子集采样的评估"""
        # 检查缓存
        cached = cache.get(params, subset_fraction)
        if cached is not None:
            return cached
        
        # 子集采样
        if subset_fraction < 1.0 and len(X) > 100:
            n_sub = max(int(len(X) * subset_fraction), min(50, len(X) // 2))
            rng = np.random.RandomState(self.random_state + len(cache._cache))
            idx = rng.choice(len(X), n_sub, replace=False)
            if isinstance(X, pd.DataFrame):
                X_eval = X.iloc[idx]
            else:
                X_eval = X[idx]
            if isinstance(y, pd.Series):
                y_eval = y.iloc[idx]
            else:
                y_eval = y[idx]
        else:
            X_eval, y_eval = X, y
        
        score = self._evaluate_model_with_params(model_key, params, X_eval, y_eval, task_type, metric)
        cache.set(params, subset_fraction, score)
        return score
    
    def _get_subset_fraction(self, trial_progress: float) -> float:
        """根据 trial 进度返回子集采样比例"""
        for threshold, fraction in self.subset_schedule:
            if trial_progress >= threshold:
                current = fraction
            else:
                break
        return current
    
    # -------------------------------------------------------------------------
    # 动作空间与状态构建
    # -------------------------------------------------------------------------
    
    def _build_candidates(self, search_space: Dict, n_candidates: int = 8) -> Dict[str, List]:
        """从搜索空间构建候选值列表（支持 SearchSpace 对象）"""
        # 如果已经是 SearchSpace，直接调用其 build_candidates
        if hasattr(search_space, 'build_candidates'):
            return search_space.build_candidates(n=n_candidates)
        
        # 向后兼容: dict 格式
        candidates = {}
        for name, spec in search_space.items():
            if isinstance(spec, list):
                candidates[name] = spec
                continue
            if not isinstance(spec, dict):
                candidates[name] = [spec]
                continue
            ptype = spec.get('type', 'float')
            if ptype == 'int':
                low, high = spec['low'], spec['high']
                vals = list(range(low, high + 1, max(1, (high - low) // n_candidates)))
                if len(vals) > n_candidates:
                    vals = vals[:n_candidates]
                candidates[name] = vals
            elif ptype == 'float':
                low, high = spec['low'], spec['high']
                scale = spec.get('scale', 'linear')
                if scale == 'log':
                    import math
                    if low <= 0:
                        low = 1e-10
                    log_vals = np.linspace(math.log(low), math.log(high), n_candidates)
                    vals = [float(math.exp(v)) for v in log_vals]
                    seen = set()
                    unique = []
                    for v in vals:
                        key = round(v, 10)
                        if key not in seen:
                            seen.add(key)
                            unique.append(v)
                    candidates[name] = unique
                else:
                    candidates[name] = list(np.linspace(low, high, n_candidates))
            elif ptype == 'categorical':
                candidates[name] = spec['choices']
            else:
                candidates[name] = [spec.get('default', 0)]
        return candidates
    
    def _random_params(self, search_space: Dict) -> Dict:
        """从搜索空间随机采样一组参数（支持 SearchSpace 对象）"""
        if hasattr(search_space, 'sample'):
            return search_space.sample(random_state=self.random_state)
        
        # 向后兼容: dict 格式
        params = {}
        for name, spec in search_space.items():
            if isinstance(spec, list):
                params[name] = random.choice(spec)
                continue
            if not isinstance(spec, dict):
                params[name] = spec
                continue
            ptype = spec.get('type', 'float')
            if ptype == 'int':
                params[name] = random.randint(spec['low'], spec['high'])
            elif ptype == 'float':
                params[name] = random.uniform(spec['low'], spec['high'])
            elif ptype == 'categorical':
                params[name] = random.choice(spec['choices'])
        return params
    
    def _build_state(self, meta_features: np.ndarray,
                     current_params: Dict,
                     param_candidates: Dict[str, List],
                     last_score: float,
                     trial_progress: float,
                     last_action_idx: int,
                     epsilon: float,
                     score_trend: float = 0.0,
                     score_std: float = 0.0,
                     exploration_ratio: float = 0.0,
                     param_coverage_avg: float = 0.0) -> np.ndarray:
        """
        构建 DQN 输入状态 (v3 丰富状态编码)。
        
        包含:
          1. 数据集元特征 (8维)
          2. 参数归一化值 (n_params维)
          3. 历史信息 (8维):
             - last_score_norm: 最近 trial 得分（tanh 压缩）
             - trial_progress: 当前进度 [0, 1]
             - last_action_norm: 上次动作索引 / action_dim
             - epsilon: 当前探索率 [0, 1]
             - score_trend: 最近窗口 score 趋势（线性回归斜率，tanh 压缩）
             - score_std: 最近窗口 score 标准差（sigmoid 压缩）
             - exploration_ratio: 已尝试 unique 配置比例 [0, 1]
             - param_coverage_avg: 参数-值对覆盖率 [0, 1]
        """
        param_values = []
        for pname, cand_list in param_candidates.items():
            val = current_params.get(pname)
            if val is None:
                param_values.append(0.0)
                continue
            
            # 处理 numpy 标量类型（np.float64, np.int64, np.str_ 等）
            if isinstance(val, (np.integer, np.floating)):
                val = float(val) if isinstance(val, np.floating) else int(val)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if len(cand_list) > 0:
                    cmin, cmax = min(cand_list), max(cand_list)
                    if cmax > cmin:
                        norm = (float(val) - cmin) / (cmax - cmin)
                    else:
                        norm = 0.5
                else:
                    norm = 0.5
                param_values.append(norm)
            else:
                try:
                    # numpy 字符串与 Python 字符串混合比较
                    str_val = str(val)
                    idx = next((i for i, c in enumerate(cand_list) if str(c) == str_val), -1)
                    if idx >= 0:
                        norm = idx / max(len(cand_list) - 1, 1)
                    else:
                        norm = 0.5
                except Exception:
                    norm = 0.5
                param_values.append(norm)
        
        total_actions = sum(len(c) for c in param_candidates.values())
        
        # 安全构造 extra 数组，避免 numpy 标量类型问题
        extra = np.zeros(8, dtype=np.float32)
        extra[0] = float(np.tanh(float(last_score)))
        extra[1] = float(trial_progress)
        extra[2] = float(last_action_idx) / max(float(total_actions), 1.0)
        extra[3] = float(epsilon)
        extra[4] = float(np.tanh(float(score_trend)))
        extra[5] = float((1.0 / (1.0 + np.exp(-float(score_std))) - 0.5) * 2.0)
        extra[6] = float(exploration_ratio)
        extra[7] = float(param_coverage_avg)
        
        state = np.concatenate([
            meta_features,
            np.array(param_values, dtype=np.float32),
            extra
        ])
        
        return state
    
    # -------------------------------------------------------------------------
    # DQN 训练
    # -------------------------------------------------------------------------
    
    def _train_step(self, policy_net: Any, target_net: Any, optimizer: Any) -> None:
        states, actions, rewards, next_states, dones = self._buffer.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self._device)
        actions_t = torch.LongTensor(actions).to(self._device)
        rewards_t = torch.FloatTensor(rewards).to(self._device)
        next_states_t = torch.FloatTensor(next_states).to(self._device)
        dones_t = torch.FloatTensor(dones).to(self._device)
        
        current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()
    
    # -------------------------------------------------------------------------
    # 回退策略
    # -------------------------------------------------------------------------
    
    def _parallel_final_eval(self, model_key: str, history: List[Dict],
                              X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                              metric: Optional[str],
                              current_best_score: float,
                              current_best_params: Dict) -> Tuple[float, Dict]:
        """
        并行最终验证: 从历史中选择 top-K 唯一配置，用全量数据并行重新评估。
        
        由于子集采样可能引入噪声，训练结束后对历史最佳配置的若干变体
        进行全量并行验证，取最优结果。
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            log_warning("[RLOptimizer] joblib 未安装，跳过并行最终验证")
            return current_best_score, current_best_params
        
        # 提取历史中的唯一配置及其最佳 score
        config_scores: Dict[tuple, float] = {}
        config_params: Dict[tuple, Dict] = {}
        for h in history:
            cfg_key = tuple(sorted((k, str(v)) for k, v in h['params'].items()))
            if cfg_key not in config_scores or h['score'] > config_scores[cfg_key]:
                config_scores[cfg_key] = h['score']
                config_params[cfg_key] = h['params']
        
        # 按 score 排序，取 top-K — 使用 heapq.nlargest，O(n log k) 优于 sorted 的 O(n log n)
        top_k = min(self.n_parallel * 2, len(config_scores))
        import heapq
        top_configs = heapq.nlargest(top_k, config_scores.items(), key=lambda x: x[1])
        
        if len(top_configs) <= 1:
            return current_best_score, current_best_params
        
        log_info(f"[RLOptimizer] 并行最终验证: {len(top_configs)} 个配置，n_jobs={self.n_parallel}")
        
        def _eval_cfg(params: Dict) -> Tuple[float, Dict]:
            try:
                score = self._evaluate_model_with_params(model_key, params, X, y, task_type, metric)
                return score, params
            except Exception as e:
                return float('-inf'), params
        
        results = Parallel(n_jobs=self.n_parallel, backend='threading')(
            delayed(_eval_cfg)(config_params[cfg_key]) for cfg_key, _ in top_configs
        )
        
        best_score = current_best_score
        best_params = current_best_params
        for score, params in results:
            if score > best_score:
                best_score = score
                best_params = deepcopy(params)
        
        log_info(f"[RLOptimizer] 并行验证完成: best_score={best_score:.4f}")
        return best_score, best_params
    
    def _random_search_fallback(self, model_key: str, spec: Any, search_space: Dict, X: pd.DataFrame, y: pd.Series, task_type: TaskType, metric: Optional[str]) -> OptimizationResult:
        best_score = float('-inf')
        best_params = deepcopy(spec.default_params)
        history = []
        
        for trial in progress_range(self.n_trials, desc=f"RL随机搜索 {model_key}", disable=not self.verbose):
            params = self._random_params(search_space)
            try:
                score = self._evaluate_model_with_params(model_key, params, X, y, task_type, metric)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_params = deepcopy(params)
            history.append({
                'trial': trial + 1,
                'params': params,
                'score': score,
                'best_so_far': best_score,
                'reward': 0.0,
                'epsilon': 1.0,
                'subset_fraction': 1.0,
                'action': 'random',
            })
        
        return OptimizationResult(
            model_key=model_key,
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            n_trials=self.n_trials,
            sampler_type='rl_random_fallback'
        )
