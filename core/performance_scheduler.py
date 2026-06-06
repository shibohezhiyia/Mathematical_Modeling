"""
自动性能调度器

根据数据规模、硬件资源、任务类型自动选择最优处理策略，
实现"无感切换"——用户无需关心底层实现，系统自动调至最佳性能模式。

策略分级：
- STANDARD (标准模式): 完整分析，无采样，全部特征参与
- FAST (快速模式): 分层采样，减少候选列，简化统计检验
- ULTRA (极速模式): 极大采样，极简分析，GPU优先，多进程最大化
"""

import os
import psutil
import platform
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import pandas as pd
import numpy as np

from utils.helpers import log_info, log_warning, timer


class StrategyLevel(Enum):
    """性能策略级别"""
    STANDARD = "standard"    # 标准模式：完整精确
    FAST = "fast"            # 快速模式：采样加速
    ULTRA = "ultra"          # 极速模式：极简+GPU
    
    def __repr__(self) -> str:
        return f"StrategyLevel.{self.name}"


@dataclass
class DataScaleMetrics:
    """数据规模指标"""
    n_rows: int = 0
    n_cols: int = 0
    memory_mb: float = 0.0
    n_numeric: int = 0
    n_categorical: int = 0
    n_text: int = 0
    n_datetime: int = 0
    total_cells: int = 0
    sparsity: float = 0.0  # 缺失率
    
    @property
    def size_tier(self) -> str:
        """数据量级分级"""
        if self.n_rows < 10_000:
            return "small"
        elif self.n_rows < 100_000:
            return "medium"
        elif self.n_rows < 1_000_000:
            return "large"
        elif self.n_rows < 10_000_000:
            return "xlarge"
        else:
            return "huge"
    
    @property
    def complexity_score(self) -> float:
        """
        复杂度评分 (0-100)
        综合考虑行数、列数、内存、文本列比例
        """
        # 对数缩放，避免大数据线性爆炸，同时保证中等数据有合理分数
        import math
        row_score = min(math.log10(max(self.n_rows, 10)) * 12, 50)  # 1万→48分, 100万→60→封顶50
        col_score = min(self.n_cols / 50, 20)        # 100列→20分封顶
        mem_score = min(self.memory_mb / 200, 15)    # 3GB→15分封顶
        text_score = min(self.n_text * 5, 15)         # 文本列贡献最多15分
        return row_score + col_score + mem_score + text_score


@dataclass
class HardwareProfile:
    """硬件资源画像"""
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    has_gpu: bool = False
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    os_name: str = ""
    
    @property
    def compute_score(self) -> float:
        """计算能力评分 (0-100)"""
        cpu_score = min(self.cpu_count * 5, 30)
        mem_score = min(self.memory_total_gb / 4, 30)
        gpu_score = 40 if self.has_gpu else 0
        return cpu_score + mem_score + gpu_score


@dataclass
class ExecutionPlan:
    """执行计划"""
    strategy: StrategyLevel = StrategyLevel.STANDARD
    reason: str = ""
    
    # 数据模块参数
    sample_size: Optional[int] = None
    type_detect_sample: Optional[int] = None
    
    # 缺失分析参数
    missing_sample_size: Optional[int] = None
    missing_max_candidates: int = 50
    missing_structural_threshold: float = 0.90
    use_mi_correlation: bool = True
    
    # 建模参数
    n_jobs: int = 1
    use_gpu: bool = False
    cv_folds: int = 5
    max_models: int = 10
    hyperparameter_trials: int = 50
    early_stopping_rounds: int = 50
    
    # 内存管理
    chunk_size: Optional[int] = None
    gc_frequency: int = 1  # 每处理多少列GC一次


class HardwareDetector:
    """硬件探测器"""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def detect() -> HardwareProfile:
        """探测硬件资源（结果缓存）"""
        profile = HardwareProfile()
        
        # CPU
        profile.cpu_count = os.cpu_count() or 1
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                profile.cpu_freq_mhz = cpu_freq.current
        except Exception:
            pass
        
        # Memory
        mem = psutil.virtual_memory()
        profile.memory_total_gb = mem.total / (1024 ** 3)
        profile.memory_available_gb = mem.available / (1024 ** 3)
        
        # GPU
        profile.has_gpu, profile.gpu_count, profile.gpu_names, profile.gpu_memory_gb = \
            HardwareDetector._detect_gpu()
        
        profile.os_name = platform.system()
        
        return profile
    
    @staticmethod
    def _detect_gpu() -> Tuple[bool, int, List[str], List[float]]:
        """检测GPU可用性（快速版本）"""
        gpu_names: List[str] = []
        gpu_mems: List[float] = []
        
        # 方法1: pynvml (最快最轻量)
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_names.append(name)
                gpu_mems.append(mem.total / (1024 ** 3))
            pynvml.nvmlShutdown()
            return True, count, gpu_names, gpu_mems
        except ImportError:
            log_warning("[HardwareDetector] pynvml 未安装，跳过NVML检测")
        except Exception as e:
            log_warning(f"[HardwareDetector] pynvml 检测失败: {e}")
        
        # 方法2: PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                for i in range(count):
                    gpu_names.append(torch.cuda.get_device_name(i))
                    props = torch.cuda.get_device_properties(i)
                    gpu_mems.append(props.total_memory / (1024 ** 3))
                return True, count, gpu_names, gpu_mems
            else:
                log_warning("[HardwareDetector] PyTorch已安装但CUDA不可用（可能是CPU版本）")
        except ImportError:
            log_warning("[HardwareDetector] PyTorch 未安装，跳过CUDA检测")
        
        # 方法3: RAPIDS/cuPy
        try:
            import cupy as cp
            gpu_names.append(f"CUDA Device")
            gpu_mems.append(cp.cuda.Device(0).mem_info[1] / (1024 ** 3))
            return True, 1, gpu_names, gpu_mems
        except ImportError:
            log_warning("[HardwareDetector] cuPy 未安装，跳过RAPIDS检测")
        except Exception as e:
            log_warning(f"[HardwareDetector] cuPy 检测失败: {e}")
        
        # 方法4: nvidia-smi 命令行兜底（Windows/Linux通用）
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5, check=True
            )
            for line in result.stdout.strip().split('\n'):
                if ',' in line:
                    name, mem_str = line.split(',', 1)
                    gpu_names.append(name.strip())
                    # 解析显存字符串，如 "16384 MiB" 或 "16 GB"
                    mem_val = 0.0
                    for token in mem_str.strip().split():
                        try:
                            mem_val = float(token)
                            break
                        except ValueError:
                            continue
                    if 'MiB' in mem_str or 'MB' in mem_str:
                        mem_val = mem_val / 1024
                    gpu_mems.append(mem_val)
            if gpu_names:
                log_info(f"[HardwareDetector] 通过 nvidia-smi 检测到 {len(gpu_names)} 个GPU")
                return True, len(gpu_names), gpu_names, gpu_mems
        except FileNotFoundError:
            log_warning("[HardwareDetector] nvidia-smi 未找到，请确认NVIDIA驱动已安装且PATH配置正确")
        except subprocess.TimeoutExpired:
            log_warning("[HardwareDetector] nvidia-smi 执行超时")
        except Exception as e:
            log_warning(f"[HardwareDetector] nvidia-smi 检测失败: {e}")
        
        return False, 0, [], []


class DataScaleEvaluator:
    """数据规模评估器"""
    
    @staticmethod
    def evaluate(df: pd.DataFrame) -> DataScaleMetrics:
        """评估数据规模（向量化实现避免逐列 Python 循环）"""
        metrics = DataScaleMetrics()
        metrics.n_rows = len(df)
        metrics.n_cols = len(df.columns)
        metrics.memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        metrics.total_cells = metrics.n_rows * metrics.n_cols
        
        # 向量化统计各类型列数（一次性推断所有列类型）
        dtypes = df.dtypes
        is_numeric = dtypes.apply(pd.api.types.is_numeric_dtype)
        is_datetime = dtypes.apply(pd.api.types.is_datetime64_any_dtype)
        is_object = (dtypes == object)
        
        metrics.n_numeric = int(is_numeric.sum())
        metrics.n_datetime = int(is_datetime.sum())
        
        # 对 object 列批量判断文本 vs 类别（避免逐列 str.len().mean()）
        obj_cols = df.columns[is_object]
        if len(obj_cols) > 0:
            n_unique = df[obj_cols].nunique(dropna=True)
            # 只对可能为文本的列计算字符串长度（启发式：高唯一值或名称含 text/comment 等）
            text_like = n_unique > 50
            # 检查列名是否包含文本特征关键词
            text_keywords = ['text', 'comment', 'desc', 'content', 'message', 'body']
            for col in obj_cols:
                if any(kw in col.lower() for kw in text_keywords):
                    text_like[col] = True
            metrics.n_text = int(text_like.sum())
            metrics.n_categorical = len(obj_cols) - metrics.n_text
        else:
            metrics.n_text = 0
            metrics.n_categorical = int((~is_numeric & ~is_datetime).sum())
        
        # 缺失率（向量化计算，避免 sum().sum() 的双重遍历）
        total_nulls = df.isnull().sum().sum()
        metrics.sparsity = total_nulls / metrics.total_cells if metrics.total_cells > 0 else 0
        
        return metrics


class PerformanceScheduler:
    """
    自动性能调度器
    
    核心职责：
    1. 评估数据规模
    2. 探测硬件资源
    3. 综合决策最优策略
    4. 生成详细执行计划
    
    决策矩阵：
    - 小数据 + 强硬件 → STANDARD（精确分析）
    - 中数据 + 一般硬件 → FAST（采样加速）
    - 大数据 + 任意硬件 → ULTRA（极简+GPU）
    - 大数据 + GPU → ULTRA with GPU full power
    """
    
    # 策略切换阈值
    FAST_THRESHOLD = 30.0      # 复杂度评分超过此值切FAST
    ULTRA_THRESHOLD = 60.0     # 复杂度评分超过此值切ULTRA
    
    def __init__(self, user_preference: Optional[StrategyLevel] = None) -> None:
        """
        Args:
            user_preference: 用户偏好策略（None则自动决策）
        """
        self.user_preference = user_preference
        self._hardware: Optional[HardwareProfile] = None
        self._data_metrics: Optional[DataScaleMetrics] = None
    
    @timer
    def schedule(self, df: pd.DataFrame) -> ExecutionPlan:
        """
        主调度入口：根据数据和硬件生成执行计划
        
        Returns:
            ExecutionPlan: 完整执行计划
        """
        # 评估数据
        data_metrics = DataScaleEvaluator.evaluate(df)
        self._data_metrics = data_metrics
        
        # 探测硬件
        hardware = HardwareDetector.detect()
        self._hardware = hardware
        
        log_info(f"[Scheduler] 数据规模: {data_metrics.n_rows:,}行 × {data_metrics.n_cols}列, "
                 f"内存: {data_metrics.memory_mb:.1f}MB, 复杂度: {data_metrics.complexity_score:.1f}")
        log_info(f"[Scheduler] 硬件: {hardware.cpu_count}核CPU, "
                 f"{hardware.memory_total_gb:.1f}GB内存, "
                 f"GPU: {'有' if hardware.has_gpu else '无'}")
        
        # 决策策略
        if self.user_preference:
            strategy = self.user_preference
            reason = f"用户指定: {strategy.value}"
        else:
            strategy, reason = self._decide_strategy(data_metrics, hardware)
        
        log_info(f"[Scheduler] 选择策略: {strategy.value} ({reason})")
        
        # 生成执行计划
        plan = self._build_plan(strategy, data_metrics, hardware)
        plan.reason = reason
        
        return plan
    
    def _decide_strategy(self, data: DataScaleMetrics, hw: HardwareProfile) -> Tuple[StrategyLevel, str]:
        """
        策略决策核心逻辑
        
        综合考虑数据复杂度和硬件能力
        """
        complexity = data.complexity_score
        compute = hw.compute_score
        
        # 决策规则（复杂度优先，硬件调节）
        if complexity >= self.ULTRA_THRESHOLD:
            # 大数据必须极速模式
            if hw.has_gpu and compute >= 50:
                return StrategyLevel.ULTRA, f"超大数据(复杂度{complexity:.0f})+强硬件(GPU)"
            else:
                return StrategyLevel.ULTRA, f"超大数据(复杂度{complexity:.0f})，硬件受限"
        
        elif complexity >= self.FAST_THRESHOLD:
            # 中等数据
            if compute >= 60:
                return StrategyLevel.FAST, f"中等数据(复杂度{complexity:.0f})+强硬件，可快速处理"
            else:
                return StrategyLevel.FAST, f"中等数据(复杂度{complexity:.0f})+弱硬件，必须加速"
        
        else:
            # 小数据
            if compute >= 40:
                return StrategyLevel.STANDARD, f"小数据(复杂度{complexity:.0f})+充足资源，精确分析"
            else:
                return StrategyLevel.FAST, f"小数据但硬件较弱，使用快速模式"
    
    def _build_plan(self, strategy: StrategyLevel, data: DataScaleMetrics, hw: HardwareProfile) -> ExecutionPlan:
        """根据策略构建详细执行计划"""
        plan = ExecutionPlan(strategy=strategy)
        
        if strategy == StrategyLevel.STANDARD:
            plan = self._build_standard_plan(data, hw)
        elif strategy == StrategyLevel.FAST:
            plan = self._build_fast_plan(data, hw)
        elif strategy == StrategyLevel.ULTRA:
            plan = self._build_ultra_plan(data, hw)
        
        return plan
    
    def _build_standard_plan(self, data: DataScaleMetrics, hw: HardwareProfile) -> ExecutionPlan:
        """标准模式：完整分析"""
        plan = ExecutionPlan(strategy=StrategyLevel.STANDARD)
        
        # 数据模块：不采样
        plan.sample_size = None
        plan.type_detect_sample = None
        
        # 缺失分析：完整
        plan.missing_sample_size = None
        plan.missing_max_candidates = 100
        plan.missing_structural_threshold = 0.90
        plan.use_mi_correlation = True
        
        # 建模：标准配置
        plan.n_jobs = min(hw.cpu_count - 1, 8) if hw.cpu_count > 1 else 1
        plan.use_gpu = hw.has_gpu
        plan.cv_folds = 5
        plan.max_models = 10
        plan.hyperparameter_trials = 100
        plan.early_stopping_rounds = 100
        
        return plan
    
    def _build_fast_plan(self, data: DataScaleMetrics, hw: HardwareProfile) -> ExecutionPlan:
        """快速模式：采样+简化"""
        plan = ExecutionPlan(strategy=StrategyLevel.FAST)
        
        # 根据数据量确定采样大小
        if data.n_rows > 500_000:
            plan.sample_size = 50_000
            plan.type_detect_sample = 30_000
        elif data.n_rows > 100_000:
            plan.sample_size = 30_000
            plan.type_detect_sample = 20_000
        else:
            plan.sample_size = 10_000
            plan.type_detect_sample = 10_000
        
        # 缺失分析：减少候选
        plan.missing_sample_size = plan.sample_size
        plan.missing_max_candidates = 30
        plan.missing_structural_threshold = 0.85  # 降低阈值提高召回
        plan.use_mi_correlation = True
        
        # 建模：加速配置
        plan.n_jobs = min(hw.cpu_count, 16) if hw.cpu_count > 1 else 1
        plan.use_gpu = hw.has_gpu
        plan.cv_folds = 3  # 减少交叉验证折数
        plan.max_models = 6
        plan.hyperparameter_trials = 30
        plan.early_stopping_rounds = 50
        
        # 内存管理
        plan.gc_frequency = 5
        
        return plan
    
    def _build_ultra_plan(self, data: DataScaleMetrics, hw: HardwareProfile) -> ExecutionPlan:
        """极速模式：极简+最大化并行+GPU优先"""
        plan = ExecutionPlan(strategy=StrategyLevel.ULTRA)
        
        # 数据模块：大量采样
        if data.n_rows > 5_000_000:
            plan.sample_size = 100_000
            plan.type_detect_sample = 50_000
        elif data.n_rows > 1_000_000:
            plan.sample_size = 50_000
            plan.type_detect_sample = 30_000
        else:
            plan.sample_size = 20_000
            plan.type_detect_sample = 10_000
        
        # 缺失分析：极简
        plan.missing_sample_size = plan.sample_size
        plan.missing_max_candidates = 15  # 只分析最相关的列
        plan.missing_structural_threshold = 0.80  # 进一步降低阈值
        plan.use_mi_correlation = False  # 跳过耗时的互信息计算
        
        # 建模：最大化并行，GPU强制
        plan.n_jobs = min(hw.cpu_count, 32) if hw.cpu_count > 1 else 1
        plan.use_gpu = hw.has_gpu  # 有GPU就用
        plan.cv_folds = 3
        plan.max_models = 4  # 只跑最强模型
        plan.hyperparameter_trials = 15
        plan.early_stopping_rounds = 30
        
        # 分块处理
        plan.chunk_size = 100_000
        plan.gc_frequency = 3
        
        return plan
    
    def get_recommendation_text(self) -> str:
        """获取策略推荐说明文本"""
        if self._data_metrics is None or self._hardware is None:
            return "尚未执行调度，请先调用 schedule()"
        
        d = self._data_metrics
        h = self._hardware
        
        lines: List[str] = [
            "=" * 60,
            "性能调度决策报告",
            "=" * 60,
            f"",
            f"【数据规模】",
            f"  行数: {d.n_rows:,}",
            f"  列数: {d.n_cols}",
            f"  内存: {d.memory_mb:.1f} MB",
            f"  量级: {d.size_tier}",
            f"  复杂度评分: {d.complexity_score:.1f}/100",
            f"",
            f"【硬件资源】",
            f"  CPU: {h.cpu_count} 核 @ {h.cpu_freq_mhz:.0f} MHz",
            f"  内存: {h.memory_total_gb:.1f} GB (可用 {h.memory_available_gb:.1f} GB)",
            f"  GPU: {'✅ ' + ', '.join(h.gpu_names) if h.has_gpu else '❌ 无'}",
            f"  计算能力评分: {h.compute_score:.1f}/100",
            f"",
            f"【推荐策略】",
        ]
        return "\n".join(lines)


# =============================================================================
# 便捷函数
# =============================================================================

def auto_schedule(df: pd.DataFrame, preference: Optional[str] = None) -> ExecutionPlan:
    """
    一键调度：自动评估并返回执行计划
    
    Args:
        df: 数据框
        preference: 用户偏好 ('standard', 'fast', 'ultra', None=自动)
        
    Returns:
        ExecutionPlan
    """
    pref = None
    if preference:
        pref = StrategyLevel(preference.lower())
    
    scheduler = PerformanceScheduler(user_preference=pref)
    return scheduler.schedule(df)
