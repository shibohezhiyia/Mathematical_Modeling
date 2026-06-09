"""
硬件加速层

提供 GPU 加速和多进程并行的统一抽象，
自动检测硬件并选择最优执行路径。

支持：
- GPU: cuML/cuDF (RAPIDS), XGBoost/LightGBM/CatBoost GPU, PyTorch/TensorFlow
- CPU多进程: Joblib, multiprocessing, ProcessPoolExecutor
- 自动回退: GPU不可用时无缝回退到CPU
"""

import os
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps

import numpy as np
import pandas as pd
import psutil

from utils.helpers import log_info, log_warning


# =============================================================================
# GPU 检测与封装
# =============================================================================

class GPUManager:
    """GPU 管理器：统一封装各类GPU库"""
    
    def __init__(self) -> None:
        self.available = False
        self.backend = None  # 'cuml', 'cupy', 'torch', 'tf'
        self.device_count = 0
        self._detect()
    
    def _detect(self) -> None:
        """检测可用的GPU后端（快速检测，避免导入重型库）"""
        # 通过环境变量可跳过GPU检测加速启动
        if os.environ.get('SKIP_GPU_DETECT', '').lower() in ('1', 'true', 'yes'):
            log_info("[GPU] 跳过检测 (SKIP_GPU_DETECT=1)")
            return
        
        # 1. PyTorch (常用且检测快)
        try:
            import torch
            if torch.cuda.is_available():
                self.available = True
                self.backend = 'torch'
                self.device_count = torch.cuda.device_count()
                log_info(f"[GPU] 检测到 PyTorch CUDA，{self.device_count} 个GPU")
                return
        except ImportError:
            pass
        
        # 2. RAPIDS/cuML
        try:
            import cuml
            self.available = True
            self.backend = 'cuml'
            self.device_count = self._get_cuml_device_count()
            log_info(f"[GPU] 检测到 RAPIDS/cuML，{self.device_count} 个GPU")
            return
        except ImportError:
            pass
        
        # 3. TensorFlow (可能启动慢，放最后)
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.available = True
                self.backend = 'tf'
                self.device_count = len(gpus)
                log_info(f"[GPU] 检测到 TensorFlow GPU，{self.device_count} 个")
                return
        except ImportError:
            pass
        
        log_info("[GPU] 未检测到可用GPU，将使用CPU")
    
    def _get_cuml_device_count(self) -> int:
        try:
            import cuml
            return cuml.cuda.device_count()
        except Exception:
            # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
            return 1
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, Any]:
        """
        获取 GPU 显存信息
        
        Returns:
            {
                'total_mb': 总显存(MB),
                'used_mb': 已用显存(MB),
                'free_mb': 可用显存(MB),
                'utilization': 使用率(0-1)
            }
        """
        if not self.available:
            return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization': 0.0}
        
        # 优先使用 pynvml（最准确）
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem.total / (1024 ** 2)
            used_mb = mem.used / (1024 ** 2)
            free_mb = mem.free / (1024 ** 2)
            return {
                'total_mb': round(total_mb, 1),
                'used_mb': round(used_mb, 1),
                'free_mb': round(free_mb, 1),
                'utilization': round(used_mb / total_mb, 4) if total_mb > 0 else 0.0
            }
        except Exception:
            pass
        
        # 回退到 torch.cuda
        if self.backend == 'torch':
            try:
                import torch
                props = torch.cuda.get_device_properties(device_id)
                total_mb = props.total_memory / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
                allocated_mb = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                return {
                    'total_mb': round(total_mb, 1),
                    'used_mb': round(allocated_mb, 1),
                    'free_mb': round(total_mb - reserved_mb, 1),
                    'utilization': round(allocated_mb / total_mb, 4) if total_mb > 0 else 0.0
                }
            except Exception:
                pass
        
        return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization': 0.0}
    
    def check_memory(self, device_id: int = 0, min_free_mb: float = 500,
                     warn_threshold: float = 0.95) -> bool:
        """
        检查 GPU 显存是否充足
        
        Args:
            device_id: GPU 设备 ID
            min_free_mb: 最小可用显存阈值(MB)
            warn_threshold: 使用率警告阈值
            
        Returns:
            True 如果显存充足
        """
        info = self.get_memory_info(device_id)
        if info['total_mb'] == 0:
            return False
        
        if info['free_mb'] < min_free_mb:
            log_warning(
                f"[GPU] 显存不足: 可用 {info['free_mb']:.0f}MB < 阈值 {min_free_mb:.0f}MB"
            )
            return False
        
        if info['utilization'] > warn_threshold:
            log_warning(
                f"[GPU] 显存使用率过高: {info['utilization']*100:.1f}%"
            )
        
        return True
    
    def to_gpu(self, data: Union[pd.DataFrame, np.ndarray, 'cudf.DataFrame']) -> Any:
        """将数据转移到GPU"""
        if not self.available:
            return data
        
        if self.backend == 'cuml':
            try:
                import cudf
                import cupy as cp
                if isinstance(data, pd.DataFrame):
                    return cudf.DataFrame.from_pandas(data)
                elif isinstance(data, np.ndarray):
                    return cp.array(data)
            except Exception as e:
                log_warning(f"[GPU] 数据传输失败: {e}，回退到CPU")
                return data
        
        return data
    
    def to_cpu(self, data: Any) -> Any:
        """将数据转移回CPU"""
        if data is None:
            return None
        
        try:
            # cuDF/cuPy → pandas/numpy
            if hasattr(data, 'to_pandas'):
                return data.to_pandas()
            if hasattr(data, 'get'):
                return data.get()
        except Exception:
            # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
            pass
        
        return data


# 全局GPU管理器单例
_gpu_manager = None

def get_gpu_manager() -> GPUManager:
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


# =============================================================================
# 多进程引擎
# =============================================================================

class ParallelEngine:
    """
    并行计算引擎 (v2)
    
    自动选择最优并行策略：
    - 计算密集型 → 多进程 (ProcessPool)
    - IO密集型 / 模型训练 → 多线程 (ThreadPool) 或 Joblib
    - 分布式集群 → Dask (可选)
    """
    
    def __init__(self, n_jobs: int = -1, backend: str = 'auto') -> None:
        """
        Args:
            n_jobs: 并行数，-1=全部核心
            backend: 'auto', 'process', 'thread', 'joblib', 'dask'
        """
        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        self.backend = backend
        self._executor = None
        self._dask_client = None
    
    def map(self, func: Callable, iterable: List[Any], 
            chunksize: int = 1) -> List[Any]:
        """
        并行映射
        
        Args:
            func: 处理函数
            iterable: 输入序列
            chunksize: 每块大小
            
        Returns:
            结果列表
        """
        if len(iterable) == 1 or self.n_jobs == 1:
            return [func(item) for item in iterable]
        
        backend = self._choose_backend(func)
        
        if backend == 'joblib':
            return self._map_joblib(func, iterable)
        elif backend == 'thread':
            return self._map_thread(func, iterable)
        elif backend == 'dask':
            return self._map_dask(func, iterable)
        else:
            return self._map_process(func, iterable, chunksize)
    
    def _choose_backend(self, func: Callable) -> str:
        """自动选择后端"""
        if self.backend != 'auto':
            return self.backend
        
        # 如果是模型训练（有fit方法），用线程或joblib避免序列化开销
        if hasattr(func, '__name__') and any(kw in func.__name__ for kw in ['fit', 'train', 'model']):
            return 'thread'
        
        # 默认多进程
        return 'process'
    
    def _map_joblib(self, func: Callable, iterable: List[Any]) -> List[Any]:
        try:
            from joblib import Parallel, delayed
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(item) for item in iterable
            )
        except ImportError:
            return self._map_process(func, iterable)
    
    def _map_thread(self, func: Callable, iterable: List[Any]) -> List[Any]:
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(executor.map(func, iterable))
    
    def _map_process(self, func: Callable, iterable: List[Any], 
                     chunksize: int = 1) -> List[Any]:
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(executor.map(func, iterable, chunksize=chunksize))
    
    def _map_dask(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """使用 Dask 分布式进行并行计算"""
        try:
            from dask.distributed import Client, LocalCluster
            if self._dask_client is None:
                cluster = LocalCluster(n_workers=self.n_jobs, threads_per_worker=1)
                self._dask_client = Client(cluster)
                log_info(f"[ParallelEngine] Dask 本地集群启动: {self.n_jobs} workers")
            futures = [self._dask_client.submit(func, item) for item in iterable]
            return [f.result() for f in futures]
        except ImportError:
            log_warning("[ParallelEngine] Dask 未安装，回退到进程池")
            return self._map_process(func, iterable)
    
    def close(self) -> None:
        """关闭资源（如 Dask client）"""
        if self._dask_client is not None:
            try:
                self._dask_client.close()
                self._dask_client = None
            except Exception as e:
                log_warning(f"[ParallelEngine] 关闭 Dask client 失败: {e}")
    
    def __del__(self) -> None:
        self.close()
    
    def starmap(self, func: Callable, args_list: List[tuple]) -> List[Any]:
        """支持多参数并行"""
        wrapper = lambda args: func(*args)
        return self.map(wrapper, args_list)


# =============================================================================
# 模型GPU包装器
# =============================================================================

def auto_gpu_model(model_class: type, use_gpu: bool = True, **kwargs) -> Any:
    """
    自动为模型启用GPU支持
    
    支持的模型：
    - XGBClassifier/XGBRegressor: tree_method='gpu_hist'
    - LGBMClassifier/LGBMRegressor: device='gpu'
    - CatBoost: task_type='GPU'
    - sklearn: 尝试cuML等效模型
    
    Args:
        model_class: 模型类
        use_gpu: 是否尝试GPU
        **kwargs: 模型参数
        
    Returns:
        配置好的模型实例
    """
    if not use_gpu:
        return model_class(**kwargs)
    
    gpu = get_gpu_manager()
    if not gpu.available:
        return model_class(**kwargs)
    
    class_name = model_class.__name__
    module_name = model_class.__module__
    
    # XGBoost
    if 'xgboost' in module_name:
        kwargs['tree_method'] = 'gpu_hist'
        kwargs['predictor'] = 'gpu_predictor'
        log_info(f"[GPU] XGBoost 启用 GPU 加速")
        return model_class(**kwargs)
    
    # LightGBM
    if 'lightgbm' in module_name:
        kwargs['device'] = 'gpu'
        kwargs['gpu_platform_id'] = 0
        kwargs['gpu_device_id'] = 0
        log_info(f"[GPU] LightGBM 启用 GPU 加速")
        return model_class(**kwargs)
    
    # CatBoost
    if 'catboost' in module_name:
        kwargs['task_type'] = 'GPU'
        kwargs['devices'] = '0'
        log_info(f"[GPU] CatBoost 启用 GPU 加速")
        return model_class(**kwargs)
    
    # sklearn → 尝试 cuML
    if gpu.backend == 'cuml' and 'sklearn' in module_name:
        cuml_model = _sklearn_to_cuml(model_class)
        if cuml_model:
            log_info(f"[GPU] sklearn {class_name} → cuML 等效模型")
            return cuml_model(**kwargs)
    
    return model_class(**kwargs)


def _sklearn_to_cuml(sklearn_class: type) -> Optional[type]:
    """sklearn模型到cuML模型的映射"""
    try:
        import cuml
        mapping = {
            'LogisticRegression': cuml.linear_model.LogisticRegression,
            'Ridge': cuml.linear_model.Ridge,
            'Lasso': cuml.linear_model.Lasso,
            'ElasticNet': cuml.linear_model.ElasticNet,
            'LinearRegression': cuml.linear_model.LinearRegression,
            'RandomForestClassifier': cuml.ensemble.RandomForestClassifier,
            'RandomForestRegressor': cuml.ensemble.RandomForestRegressor,
            'KNeighborsClassifier': cuml.neighbors.KNeighborsClassifier,
            'KNeighborsRegressor': cuml.neighbors.KNeighborsRegressor,
            'PCA': cuml.decomposition.PCA,
            'UMAP': cuml.manifold.UMAP,
            'KMeans': cuml.cluster.KMeans,
            'DBSCAN': cuml.cluster.DBSCAN,
        }
        return mapping.get(sklearn_class.__name__)
    except ImportError:
        return None


# =============================================================================
# 数据预处理GPU加速
# =============================================================================

class GPUDataTransformer:
    """GPU加速的数据预处理"""
    
    def __init__(self) -> None:
        self.gpu = get_gpu_manager()
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """尝试在GPU上执行预处理"""
        if not self.gpu.available or self.gpu.backend != 'cuml':
            return df
        
        try:
            import cudf
            import cupy as cp
            
            # 转GPU
            gdf = cudf.DataFrame.from_pandas(df)
            
            # 数值列标准化（GPU）
            numeric_cols = gdf.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                means = gdf[numeric_cols].mean()
                stds = gdf[numeric_cols].std()
                gdf[numeric_cols] = (gdf[numeric_cols] - means) / (stds + 1e-8)
            
            # 转回CPU
            return gdf.to_pandas()
        except Exception as e:
            log_warning(f"[GPU] 预处理加速失败: {e}")
            return df


# =============================================================================
# 装饰器
# =============================================================================

def gpu_fallback(func: Callable) -> Callable:
    """
    GPU回退装饰器
    
    函数执行时自动尝试GPU，失败则回退CPU
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        gpu = get_gpu_manager()
        if not gpu.available:
            return func(*args, **kwargs)
        
        # 尝试GPU路径
        try:
            gpu_kwargs = kwargs.copy()
            gpu_kwargs['_use_gpu'] = True
            return func(*args, **gpu_kwargs)
        except Exception as e:
            log_warning(f"[GPU] {func.__name__} GPU执行失败: {e}，回退CPU")
            return func(*args, **kwargs)
    
    return wrapper


def parallelize(n_jobs: int = -1, backend: str = 'auto') -> Callable[[Callable], Callable]:
    """
    并行化装饰器
    
    将函数转换为对列表输入的并行处理
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[Any], *args: Any, **kwargs: Any) -> Any:
            engine = ParallelEngine(n_jobs=n_jobs, backend=backend)
            task_fn = lambda item: func(item, *args, **kwargs)
            return engine.map(task_fn, items)
        return wrapper
    return decorator


# =============================================================================
# 内存优化
# =============================================================================

def optimize_memory(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    自动优化DataFrame内存占用
    
    对数值列降精度，类别列转category
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    
    # 缓存 len(df) 避免每列都重新调（O(1) 但 Python 调用有常量开销）
    n_total = len(df)
    
    for col in df.columns:
        # 缓存 series：原代码 df[col] 在 min/max/astype 之间多次出现，
        # 每次都走 IndexingEngine + 可能的拷贝。
        series = df[col]
        col_type = series.dtype
        
        if pd.api.types.is_integer_dtype(col_type):
            # 一次 agg 拿 min + max，省一次 O(n) 扫描
            cmin_cmax = series.agg(['min', 'max'])
            c_min, c_max = cmin_cmax['min'], cmin_cmax['max']
            if c_min >= 0:
                if c_max < 255:
                    df[col] = series.astype(np.uint8)
                elif c_max < 65535:
                    df[col] = series.astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = series.astype(np.uint32)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = series.astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = series.astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = series.astype(np.int32)
        
        elif pd.api.types.is_float_dtype(col_type):
            # dtype check：已经是 float32 时跳过 astype，省一次 O(n) 复制
            if col_type != np.float32:
                df[col] = series.astype(np.float32)
        
        elif col_type == object:
            n_unique = series.nunique()
            if n_unique / n_total < 0.5:
                df[col] = series.astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    
    if verbose:
        reduction = (start_mem - end_mem) / start_mem * 100 if start_mem > 0 else 0
        log_info(f"[Memory] {start_mem:.1f}MB → {end_mem:.1f}MB (减少 {reduction:.1f}%)")
    
    return df


# =============================================================================
# 便捷函数
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """获取系统信息摘要（含 GPU 显存）"""
    gpu = get_gpu_manager()
    hw = psutil.virtual_memory()
    
    info = {
        'cpu_count': os.cpu_count(),
        'memory_gb': hw.total / (1024 ** 3),
        'memory_available_gb': hw.available / (1024 ** 3),
        'gpu_available': gpu.available,
        'gpu_backend': gpu.backend,
        'gpu_count': gpu.device_count,
    }
    
    # 添加 GPU 显存信息
    if gpu.available and gpu.device_count > 0:
        try:
            mem = gpu.get_memory_info(device_id=0)
            info['gpu_memory_total_mb'] = mem['total_mb']
            info['gpu_memory_used_mb'] = mem['used_mb']
            info['gpu_memory_free_mb'] = mem['free_mb']
            info['gpu_memory_utilization'] = mem['utilization']
        except Exception:
            pass
    
    return info
