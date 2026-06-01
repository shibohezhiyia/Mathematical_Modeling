"""
强化学习优化器

基于 DQN (Deep Q-Network) 的超参数优化器，
作为传统贝叶斯优化(Optuna)的替代方案。

可用于：
- 模型超参数优化
- 特征选择
- 自动化机器学习流程决策
"""

import random
import warnings
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class RLConfig:
    """RL优化器配置"""
    learning_rate: float = 0.001
    gamma: float = 0.99           # 折扣因子
    epsilon: float = 1.0          # 探索率
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    max_episodes: int = 50
    max_steps_per_episode: int = 20


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


class SimpleDQN:
    """简化版 DQN（使用numpy实现，无外部依赖）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 简单三层网络权重
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b3 = np.zeros(action_dim)
        # 目标网络（副本）
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state, use_target=False):
        if use_target:
            W1, b1, W2, b2, W3, b3 = self.target_W1, self.target_b1, self.target_W2, self.target_b2, self.target_W3, self.target_b3
        else:
            W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        h1 = self._relu(state @ W1 + b1)
        h2 = self._relu(h1 @ W2 + b2)
        return h2 @ W3 + b3
    
    def predict(self, state):
        """预测Q值"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.forward(state)
    
    def update(self, states, actions, rewards, next_states, dones, gamma=0.99, lr=0.001):
        """批量更新"""
        batch_size = states.shape[0]
        # 当前Q值
        q_values = self.forward(states)
        # 目标Q值 (Double DQN style)
        next_q_main = self.forward(next_states)
        next_actions = np.argmax(next_q_main, axis=1)
        next_q_target = self.forward(next_states, use_target=True)
        next_q = next_q_target[np.arange(batch_size), next_actions]
        targets = q_values.copy()
        for i in range(batch_size):
            targets[i, actions[i]] = rewards[i] + gamma * next_q[i] * (1 - dones[i])
        
        # 简化的梯度下降（数值梯度）
        loss_grad = (q_values - targets) / batch_size
        # 反向传播更新（简化版）
        h1 = self._relu(states @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        
        # 输出层梯度
        dW3 = h2.T @ loss_grad
        db3 = np.sum(loss_grad, axis=0)
        dh2 = loss_grad @ self.W3.T
        dh2[h2 <= 0] = 0
        
        dW2 = h1.T @ dh2
        db2 = np.sum(dh2, axis=0)
        dh1 = dh2 @ self.W2.T
        dh1[h1 <= 0] = 0
        
        dW1 = states.T @ dh1
        db1 = np.sum(dh1, axis=0)
        
        # 更新权重
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def sync_target(self):
        """同步目标网络"""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()


class RLOptimizer:
    """
    基于强化学习的超参数优化器
    
    将超参数优化视为马尔可夫决策过程：
    - State: 当前超参数配置 + 历史性能
    - Action: 调整某个超参数（增大/减小/保持）
    - Reward: 模型性能提升
    """
    
    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.dqn = None
        self.buffer = ReplayBuffer(self.config.memory_size)
        self.episode_rewards = []
    
    def optimize(self,
                 param_space: Dict[str, List[Any]],
                 evaluate_fn: Callable[[Dict[str, Any]], float],
                 max_trials: int = 50) -> Tuple[Dict[str, Any], float]:
        """
        使用RL优化超参数
        
        Args:
            param_space: 超参数搜索空间，如 {'lr': [0.001, 0.01, 0.1], 'batch_size': [32, 64, 128]}
            evaluate_fn: 评估函数，接收参数字典，返回分数（越高越好）
            max_trials: 最大尝试次数
        
        Returns:
            (best_params, best_score)
        """
        param_names = list(param_space.keys())
        n_params = len(param_names)
        # 每个参数有3个动作：减小、保持、增大
        action_dim = n_params * 3
        # 状态：当前各参数的索引 + 上一轮的分数
        state_dim = n_params + 1
        
        self.dqn = SimpleDQN(state_dim, action_dim)
        
        # 初始化参数（随机选择）
        current_indices = [random.randint(0, len(param_space[p]) - 1) for p in param_names]
        best_params = None
        best_score = float('-inf')
        
        cfg = self.config
        step_counter = 0
        
        for episode in range(cfg.max_episodes):
            if step_counter >= max_trials:
                break
            
            state = np.array(current_indices + [0.0], dtype=np.float32)
            episode_reward = 0
            
            for step in range(cfg.max_steps_per_episode):
                if step_counter >= max_trials:
                    break
                
                # Epsilon-greedy 选择动作
                if random.random() < cfg.epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    q_values = self.dqn.predict(state)
                    action = int(np.argmax(q_values))
                
                # 执行动作：调整对应参数
                param_idx = action // 3
                action_type = action % 3  # 0=减小, 1=保持, 2=增大
                new_indices = current_indices.copy()
                if action_type == 0 and new_indices[param_idx] > 0:
                    new_indices[param_idx] -= 1
                elif action_type == 2 and new_indices[param_idx] < len(param_space[param_names[param_idx]]) - 1:
                    new_indices[param_idx] += 1
                
                # 构建参数并评估
                params = {name: param_space[name][idx] for name, idx in zip(param_names, new_indices)}
                score = evaluate_fn(params)
                
                # 计算奖励（相对于之前的提升）
                reward = score - state[-1] if state[-1] != 0 else score
                reward = max(-1, min(1, reward))  # 裁剪到 [-1, 1]
                
                next_state = np.array(new_indices + [score], dtype=np.float32)
                done = (step == cfg.max_steps_per_episode - 1)
                
                self.buffer.push(state, action, reward, next_state, done)
                
                # 训练DQN
                if len(self.buffer) >= cfg.batch_size:
                    batch = self.buffer.sample(cfg.batch_size)
                    self.dqn.update(*batch, gamma=cfg.gamma, lr=cfg.learning_rate)
                
                # 同步目标网络
                if step_counter % cfg.target_update_freq == 0:
                    self.dqn.sync_target()
                
                state = next_state
                current_indices = new_indices
                episode_reward += reward
                step_counter += 1
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            self.episode_rewards.append(episode_reward)
            cfg.epsilon = max(cfg.epsilon_min, cfg.epsilon * cfg.epsilon_decay)
        
        return best_params or params, best_score
    
    def get_stats(self) -> Dict[str, Any]:
        """返回优化统计信息"""
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
            'epsilon': self.config.epsilon,
        }
