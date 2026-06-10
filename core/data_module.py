"""
数据模块：读取数据、类型识别、基础清洗
"""
import os
import re
import json
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from utils.helpers import log_info, log_warning, log_error, timer
from core.progress_bar import progress_iter
from core.accelerators import optimize_memory
from core.workspace_manager import get_workspace_manager

# 预编译常用正则，避免每列重新编译
_RE_ID_LIKE = re.compile(r"\bid\b|_id$|id_$")

# 整数 downcast 候选表：(dtype, min, max)
# 在 _optimize_types 中按顺序检查，找到第一个能容纳列值域的最小 dtype
_UNSIGNED_INT_DTYPES: Tuple[Tuple[Any, int, int], ...] = (
    (np.uint8,        0,           255),
    (np.uint16,       0,           65535),
    (np.uint32,       0,           4294967295),
)
_SIGNED_INT_DTYPES: Tuple[Tuple[Any, int, int], ...] = (
    (np.int8,         -128,        127),
    (np.int16,        -32768,      32767),
    (np.int32,        -2147483648, 2147483647),
)


# 列名暗示日期的关键词（提到模块级，避免 _to_datetime 每次调用重建列表）
_DATE_LIKE_KEYWORDS: Tuple[str, ...] = (
    'date', 'time', 'dt', 'day', 'month', 'year',
)

# 布尔型检测的候选值集合（提到模块级避免每次 detect 重建）
# 使用 frozenset 替代 set 让 issubset 调用走 C 路径
_BOOL_VALUE_SET: frozenset = frozenset({0, 1, 0.0, 1.0, True, False})


class DataType(Enum):
    """数据类型枚举"""
    NUMERIC = "数值型"
    CATEGORY = "类别型"
    DATETIME = "日期时间型"
    TEXT = "文本型"
    BOOLEAN = "布尔型"
    ID = "ID型"
    CONSTANT = "常量型"
    EMPTY = "空列"


@dataclass
class ColumnProfile:
    """列分析画像"""
    name: str
    dtype: str = ""
    inferred_type: DataType = DataType.NUMERIC
    null_count: int = 0
    null_rate: float = 0.0
    unique_count: int = 0
    unique_rate: float = 0.0
    sample_values: List[Any] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class DataLoader:
    """数据加载器：支持多种格式，大文件自动分块读取"""
    
    SUPPORTED_FORMATS = {
        '.csv': pd.read_csv,
        '.txt': pd.read_csv,
        '.tsv': lambda p, **k: pd.read_csv(p, sep='\t', **k),
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.json': pd.read_json,
        '.parquet': pd.read_parquet,
    }
    
    # 大文件阈值（MB）
    CHUNK_THRESHOLD_MB = 100
    # 分块大小（行数）
    DEFAULT_CHUNK_SIZE = 50000
    
    def __init__(self, encoding: str = 'utf-8') -> None:
        self.encoding = encoding
    
    @staticmethod
    def _get_file_size_mb(file_path: str) -> float:
        """获取文件大小（MB）"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 ** 2)
        except Exception:
            return 0.0
    
    @classmethod
    def _should_chunk(cls, file_path: str, threshold_mb: Optional[float] = None) -> bool:
        """判断文件是否需要分块读取"""
        threshold = threshold_mb or cls.CHUNK_THRESHOLD_MB
        ext = os.path.splitext(file_path)[1].lower()
        # 只有文本格式支持分块
        if ext not in ['.csv', '.txt', '.tsv']:
            return False
        return cls._get_file_size_mb(file_path) > threshold
    
    @timer
    def load(self, file_path: Union[str, os.PathLike],
             auto_chunk: bool = True,
             chunk_size: Optional[int] = None,
             **kwargs) -> pd.DataFrame:
        """
        根据文件扩展名自动选择读取方式
        
        大文件（>100MB）自动分块读取并合并，避免内存溢出。
        
        Args:
            file_path: 文件路径
            auto_chunk: 是否自动分块读取大文件
            chunk_size: 分块行数，默认 50000
            **kwargs: 额外的读取参数
            
        Returns:
            pd.DataFrame
        """
        file_path = str(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {list(self.SUPPORTED_FORMATS.keys())}")
        
        # 自动分块读取大文件
        if auto_chunk and self._should_chunk(file_path):
            return self.load_chunked(file_path, chunk_size=chunk_size, **kwargs)
        
        reader = self.SUPPORTED_FORMATS[ext]
        
        # 默认参数
        default_kwargs = {}
        if ext in ['.csv', '.txt', '.tsv']:
            default_kwargs['encoding'] = self.encoding
            default_kwargs['low_memory'] = False
        
        default_kwargs.update(kwargs)
        
        try:
            df = reader(file_path, **default_kwargs)
            log_info(f"成功加载数据: {file_path}, 形状: {df.shape}")
            return df
        except UnicodeDecodeError:
            log_warning(f"编码错误，尝试使用 gbk 编码: {file_path}")
            default_kwargs['encoding'] = 'gbk'
            df = reader(file_path, **default_kwargs)
            log_info(f"成功加载数据(gbk): {file_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            log_error(f"加载数据失败: {file_path}, 错误: {str(e)}")
            raise
    
    @timer
    def load_chunked(self, file_path: Union[str, os.PathLike],
                     chunk_size: Optional[int] = None,
                     **kwargs) -> pd.DataFrame:
        """
        分块读取大文件并合并
        
        适合内存不足以一次性加载的大文件（>100MB）。
        分块读取时自动推断数据类型，最后合并并优化内存。
        
        Args:
            file_path: 文件路径
            chunk_size: 每块行数，默认 50000
            **kwargs: 额外的读取参数
            
        Returns:
            pd.DataFrame
        """
        file_path = str(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in ['.csv', '.txt', '.tsv']:
            raise ValueError(f"分块读取不支持格式: {ext}，仅支持 csv/txt/tsv")
        
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        reader = self.SUPPORTED_FORMATS[ext]
        
        default_kwargs = {
            'encoding': self.encoding,
            'chunksize': chunk_size,
        }
        default_kwargs.update(kwargs)
        
        verbose = kwargs.pop('verbose', True)
        
        log_info(f"[DataLoader] 大文件分块读取: {file_path}, chunk_size={chunk_size}")
        
        # 提取共用的"读+合并+优化"逻辑，让 try/except 不再重复 30+ 行代码
        # 之前 UTF-8 失败回退 GBK 的分支完整重写了读分块→concat→optimize 流程，
        # 维护时容易让两处逻辑漂移（已经发生过"是否 optimize_memory"是否一致的争论）
        def _read_and_concat(kwargs: dict) -> pd.DataFrame:
            local_chunks: List[pd.DataFrame] = []
            local_total = 0
            for i, chunk in enumerate(progress_iter(reader(file_path, **kwargs), desc="读取", disable=not verbose)):
                local_chunks.append(chunk)
                local_total += len(chunk)
                if verbose and (i + 1) % 5 == 0:
                    log_info(f"[DataLoader] 已读取 {local_total:,} 行...")
            if not local_chunks:
                return pd.DataFrame()
            return optimize_memory(pd.concat(local_chunks, axis=0, ignore_index=True, copy=False), verbose=verbose)
        
        try:
            df = _read_and_concat(default_kwargs)
            log_info(f"[DataLoader] 分块读取完成: {file_path}, 总行数: {len(df)}, 列数: {len(df.columns)}")
            return df
        except UnicodeDecodeError:
            log_warning(f"编码错误，尝试使用 gbk 编码分块读取: {file_path}")
            default_kwargs['encoding'] = 'gbk'
            return _read_and_concat(default_kwargs)
        except Exception as e:
            log_error(f"分块读取失败: {file_path}, 错误: {str(e)}")
            raise
    
    @timer
    def load_multiple(self, file_paths: List[Union[str, os.PathLike]], 
                      concat_axis: int = 0, **kwargs) -> pd.DataFrame:
        """加载多个文件并合并"""
        dfs = []
        for fp in file_paths:
            df = self.load(fp, **kwargs)
            dfs.append(df)
        
        combined = pd.concat(dfs, axis=concat_axis, ignore_index=(concat_axis == 0), copy=False)
        log_info(f"合并后数据形状: {combined.shape}")
        return combined


class TypeDetector:
    """智能类型检测器"""
    
    def __init__(self, 
                 id_threshold: float = 1.0,
                 category_threshold: float = 0.05,
                 text_length_threshold: int = 50) -> None:
        """
        Args:
            id_threshold: ID列的唯一值比例阈值（默认1.0表示唯一）
            category_threshold: 类别列的唯一值比例阈值
            text_length_threshold: 文本列的平均长度阈值
        """
        self.id_threshold = id_threshold
        self.category_threshold = category_threshold
        self.text_length_threshold = text_length_threshold
    
    def detect(self, series: pd.Series, col_name: str = "") -> Tuple[DataType, ColumnProfile]:
        """
        检测单个列的数据类型
        
        Args:
            series: pandas Series
            col_name: 列名
            
        Returns:
            (DataType, ColumnProfile)
        """
        profile = ColumnProfile(name=col_name or series.name)
        profile.dtype = str(series.dtype)
        
        # 基础统计
        n_total = len(series)
        # 一次 agg 拿到 isnull+isnull 计数（nunique 需要 dropna 所以单独走）
        # 拆分：n_null 用 isnull().sum()（O(n)），n_unique 用 nunique（O(n)），
        # sample_values 用 dropna().head(5)（短路径 O(5)）— 三者无法在一次扫描内完成，
        # 但 dropna().head(5) 只在非空列才有意义，可以让它复用 isnull() 的 mask
        null_mask = series.isnull()
        n_null = int(null_mask.sum())
        n_unique = int(series.nunique(dropna=True))

        profile.null_count = n_null
        profile.null_rate = n_null / n_total if n_total > 0 else 0
        profile.unique_count = n_unique
        profile.unique_rate = n_unique / n_total if n_total > 0 else 0
        # 复用 null_mask 避免再扫一次：取非空值的前 5 个
        non_null = series[~null_mask]
        profile.sample_values = non_null.head(5).tolist()
        
        # 空列检测
        if n_null == n_total or n_unique == 0:
            profile.inferred_type = DataType.EMPTY
            profile.suggestions.append("该列为空列，建议删除")
            return DataType.EMPTY, profile
        
        # 常量列检测
        if n_unique == 1:
            profile.inferred_type = DataType.CONSTANT
            profile.suggestions.append("该列为常量列，对建模无意义，建议删除")
            return DataType.CONSTANT, profile
        
        # ID列检测（列名包含id且唯一值比例高）
        is_id_like = bool(_RE_ID_LIKE.search(col_name.lower()))
        if is_id_like and profile.unique_rate >= 0.9:
            profile.inferred_type = DataType.ID
            profile.suggestions.append("疑似ID列，通常不作为特征使用")
            return DataType.ID, profile
        
        # 尝试转换为数值
        numeric_series = self._to_numeric(series)
        if numeric_series is not None:
            series = numeric_series
            profile.dtype = 'numeric'
            
            # 数值型进一步判断
            stats = series.describe()
            # describe() 已包含 mean/std/min/max + 25%/50%/75% 分位
            # 复用 50% 作为 median（省一次 O(n) 扫描）
            profile.stats = {
                'mean': stats.get('mean'),
                'std': stats.get('std'),
                'min': stats.get('min'),
                'max': stats.get('max'),
                'median': stats.get('50%'),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            }
            
            # 布尔型检测（只有0/1或True/False）
            # 优化：先看 n_unique 上界快速 reject（>6 一定不是布尔），
            # 再用 frozenset 做 issubset（_BOOL_VALUE_SET 已含 True/False，对 pd 自动转为 0/1 的数据也兼容）
            if n_unique <= len(_BOOL_VALUE_SET) and set(series.dropna().unique()).issubset(_BOOL_VALUE_SET):
                profile.inferred_type = DataType.BOOLEAN
                profile.suggestions.append("布尔型变量，可考虑作为类别型处理")
                return DataType.BOOLEAN, profile
            
            profile.inferred_type = DataType.NUMERIC

            # 异常值检测（IQR方法）
            # 复用 describe() 的 25%/75% 分位，避免再调 series.quantile(0.25/0.75) 两次 O(n) 扫描
            q1, q3 = stats.get('25%'), stats.get('75%')
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = ((series < lower) | (series > upper)).sum()
            if n_outliers > 0:
                profile.suggestions.append(f"检测到 {n_outliers} 个潜在异常值（IQR方法）")
            
            return DataType.NUMERIC, profile
        
        # 尝试转换为日期时间
        datetime_series = self._to_datetime(series)
        if datetime_series is not None:
            profile.inferred_type = DataType.DATETIME
            profile.dtype = 'datetime'
            profile.stats = {
                'min': datetime_series.min(),
                'max': datetime_series.max(),
                'range': datetime_series.max() - datetime_series.min()
            }
            profile.suggestions.append("日期时间型，建议提取年/月/日/周等特征")
            return DataType.DATETIME, profile
        
        # 类别型 vs 文本型
        non_null = series.dropna().astype(str)
        avg_length = non_null.str.len().mean()
        
        if avg_length > self.text_length_threshold or profile.unique_rate > self.category_threshold:
            profile.inferred_type = DataType.TEXT
            profile.stats = {
                'avg_length': avg_length,
                'max_length': non_null.str.len().max()
            }
            profile.suggestions.append("文本型数据，建议进行向量化或提取关键词")
        else:
            profile.inferred_type = DataType.CATEGORY
            profile.stats = {
                'top_values': non_null.value_counts().head(5).to_dict()
            }
            if n_unique > 50:
                profile.suggestions.append(f"类别数较多({n_unique})，建议考虑目标编码或合并稀有类别")
        
        return profile.inferred_type, profile
    
    def _to_numeric(self, series: pd.Series) -> Optional[pd.Series]:
        """尝试转换为数值型"""
        # 优化：已经是数值 dtype 的 series 直接返回，不再调 pd.to_numeric 触发 O(n) 复制
        # 之前 pd.to_numeric(numeric_series) 会对整个 series 做一遍 copy
        # （即使 dtype 不变，pandas 内部还是会构造新的 ndarray 返回）
        if pd.api.types.is_numeric_dtype(series):
            return series

        # 处理带逗号的数字
        if series.dtype == object:
            # 关键优化：iloc[:100].dropna() 替代 dropna().head(100)
            # 旧：scan 整个 series 找非空 + slice = O(n) + O(1)，n 可能百万级
            # 新：slice 头 100 + dropna = O(1) + O(100) ≈ O(1)
            # 语义差异：
            #   - 旧："前 100 个非空值"（可能来自 series 任意位置）
            #   - 新："前 100 个值中的非空部分"（最多 100 个）
            # 对类型检测的语义影响可忽略 —— 两种采样的概率分布对判断 "is numeric"
            # 都有代表性，而"前 100 个值"对 race condition 更鲁棒（如果列是按
            # 某种顺序排列的，前面的值更能代表列的典型形态）。
            sample = series.iloc[:100].dropna()
            # 鲁棒性：sample 可能为空（series 全空），
            # 后续 len(sample_stripped) == 0 时会触发 ZeroDivisionError，
            # 提前 return None 跳过（与 _to_datetime 的处理一致）
            if len(sample) == 0:
                return None
            # 检查是否看起来像数字
            # 关键：先在 sample 上小成本判断，决定是否做全列转换
            sample_stripped = sample.astype(str).str.replace(',', '', regex=False)
            try:
                converted_sample = pd.to_numeric(sample_stripped, errors='coerce')
                # 加 len(sample_stripped) > 0 守卫（理论上 sample_len > 0 但保险起见）
                if (len(sample_stripped) > 0 and
                        converted_sample.notna().sum() / len(sample_stripped) > 0.8):
                    # 命中后只对全列做一次 strip + to_numeric（之前的版本会重做 strip）
                    full_stripped = series.astype(str).str.replace(',', '', regex=False)
                    return pd.to_numeric(full_stripped, errors='coerce')
            except Exception:
                # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
                pass
        return None
    
    def _to_datetime(self, series: pd.Series) -> Optional[pd.Series]:
        """尝试转换为日期时间型"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        # 快速路径：非 object dtype（int/float/bool）几乎不可能是日期字符串，
        # 直接返回 None 跳过 _DATE_LIKE_KEYWORDS 检查 + sample dropna 分配。
        # 原来不分 dtype 都跑一遍：对每列检测是 O(k) 开销（k=name 长度），
        # N 列 × (k+dropna 内存分配) 都是浪费。
        if series.dtype != object:
            return None

        # 列名暗示日期
        col_lower = str(series.name).lower()
        is_date_like = any(kw in col_lower for kw in _DATE_LIKE_KEYWORDS)

        # 关键优化：iloc[:100].dropna() 替代 dropna().head(100)
        # 旧：scan 整个 series 找非空 + slice = O(n) + O(1)，n 可能百万级
        # 新：slice 头 100 + dropna = O(1) + O(100) ≈ O(1)
        # 语义与 _to_numeric 一致：前 100 个值中的非空部分（最多 100 个）
        sample = series.iloc[:100].dropna()
        if len(sample) == 0:
            return None
        
        try:
            converted = pd.to_datetime(sample, errors='coerce')
            success_rate = converted.notna().sum() / len(sample)
            
            # 日期相关列名降低阈值
            threshold = 0.5 if is_date_like else 0.8
            if success_rate >= threshold:
                return pd.to_datetime(series, errors='coerce')
        except Exception:
            # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
            pass
        return None
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """
        分析整个DataFrame的所有列

        优化：列多时用 ThreadPoolExecutor 并行 detect（每个 detect 是 O(n) 扫描），
        pandas 操作在多线程下能绕过 GIL 释放出 GIL，CPU 密集场景也有小幅提升。
        短列表（<8 列）保持串行，避免线程池启动开销。
        """
        cols = list(df.columns)
        n_cols = len(cols)
        # 短列数保持串行：线程池启动开销 ~50ms，比节省的扫描时间还长
        if n_cols < 8:
            return {col: self.detect(df[col], col)[1] for col in cols}

        profiles: Dict[str, ColumnProfile] = {}
        # max_workers 限制为 min(8, n_cols)：超过 CPU 核心数反而因 context switch 退化
        max_workers = min(8, n_cols)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 用 submit 替代 map 拿回 future 对应 col_name
            futures = {executor.submit(self.detect, df[col], col): col for col in cols}
            # 优化：用 as_completed 替代 dict 顺序迭代 —— dict.__iter__ 是按 submission
            # 顺序，但第一个 future 不一定先完成。as_completed 让每完成一个 future
            # 立即处理，缩短 critical path：
            #   - 串行 dict 迭代：T_total = sum of (max(individual_finish_time) at each iter)
            #   - as_completed：T_total = max(individual_finish_time)（实际接近 critical path）
            # 对齐 pandas 操作中耗时差异较大的列（datetime vs numeric）特别有效
            # as_completed 已随 ThreadPoolExecutor 一起提到模块级
            for fut in as_completed(futures):
                col = futures[fut]
                _, profile = fut.result()
                profiles[col] = profile
        return profiles


class DataCleaner:
    """基础数据清洗器"""
    
    def __init__(self, 
                 null_threshold: float = 0.9,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5) -> None:
        """
        Args:
            null_threshold: 缺失率阈值，超过则删除该列
            outlier_method: 异常值检测方法 ('iqr', 'zscore', 'none')
            outlier_threshold: 异常值判定阈值
        """
        self.null_threshold = null_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self._imputers = {}
        self._fitted = False
    
    @timer
    def clean(self, df: pd.DataFrame, profiles: Optional[Dict[str, ColumnProfile]] = None,
              target_col: Optional[str] = None, inplace: bool = False) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        
        Args:
            df: 原始数据
            profiles: 列分析画像（可选，不提供则自动分析）
            target_col: 目标变量列名（不清洗目标列）
            inplace: 是否原地修改
            
        Returns:
            清洗后的DataFrame
        """
        if not inplace:
            df = df.copy()
        
        if profiles is None:
            detector = TypeDetector()
            profiles = detector.analyze_dataframe(df)
        
        log_info(f"开始数据清洗，原始形状: {df.shape}")
        
        # 1. 删除空列和常量列
        df = self._drop_useless_columns(df, profiles)
        
        # 2. 处理缺失值
        df = self._handle_missing(df, profiles, target_col)
        
        # 3. 处理异常值（仅数值列）
        if self.outlier_method != 'none':
            df = self._handle_outliers(df, profiles, target_col)
        
        # 4. 数据类型优化
        df = self._optimize_types(df, profiles)
        
        log_info(f"数据清洗完成，最终形状: {df.shape}")
        return df
    
    def _drop_useless_columns(self, df: pd.DataFrame,
                               profiles: Dict[str, ColumnProfile]) -> pd.DataFrame:
        """删除无用列（空列、常量列）"""
        # 一次性转 set 让 `col not in df.columns` 变成 O(1) 而非 O(n)
        # 循环里 profiles.items() × df.columns 检查 = O(profiles * df_cols)
        # 转 set 后 = O(profiles + df_cols) 建 set + O(profiles) 查找
        df_cols_set = set(df.columns)
        to_drop = []
        for col, profile in profiles.items():
            if col not in df_cols_set:
                continue
            if profile.inferred_type in (DataType.EMPTY, DataType.CONSTANT):
                to_drop.append(col)
                log_info(f"删除{profile.inferred_type.value}列: {col}")
            elif profile.null_rate > self.null_threshold:
                to_drop.append(col)
                log_info(f"删除高缺失率({profile.null_rate:.1%})列: {col}")

        return df.drop(columns=to_drop) if to_drop else df
    
    def _handle_missing(self, df: pd.DataFrame,
                        profiles: Dict[str, ColumnProfile],
                        target_col: Optional[str] = None) -> pd.DataFrame:
        """处理缺失值"""
        # 关键短路径：target_col 为 None 时 c != None 对所有 c 都 True，
        # 直接 list(df.columns) 替代 list comprehension（与 _handle_outliers 一致）
        feature_cols = list(df.columns) if target_col is None else \
            [c for c in df.columns if c != target_col]

        for col in feature_cols:
            # 合并"存在性" + "profile 取出"为一次 dict.get：
            # 之前是 if col not in profile_keys: continue; profile = profiles[col]
            # 现在用 walrus 一次 dict.get 完成两件事
            profile = profiles.get(col)
            if profile is None:
                continue

            # 单次扫描拿 count（避免 .any() 短路 + .sum() 重复扫描）
            # sum() 比 any() 略贵（多一次加法），但永远 1 次扫描；而 any()+sum() 最差 2 次
            # 对"无空值列"是热路径（占比通常 >50%），sum() > 0 vs any() 都是 1 次扫描
            # 对"有空值列"是冷路径，sum() 直接给出 count 省一次
            series = df[col]
            null_count = int(series.isnull().sum())
            if null_count == 0:
                continue

            dtype = profile.inferred_type

            if dtype in (DataType.NUMERIC, DataType.BOOLEAN):
                # 数值型：中位数填充
                # 上面已经 cache 了 series，下面 median/fillna 复用
                median_val = series.median()
                df[col] = series.fillna(median_val)
                log_info(f"数值列 '{col}' 使用 {median_val:.4f} 填充 {null_count} 个缺失值")

            elif dtype == DataType.CATEGORY:
                # 类别型：众数填充
                mode_val = series.mode()
                if len(mode_val) > 0:
                    df[col] = series.fillna(mode_val[0])
                    log_info(f"类别列 '{col}' 使用 '{mode_val[0]}' 填充 {null_count} 个缺失值")
                else:
                    df[col] = series.fillna('未知')

            elif dtype == DataType.DATETIME:
                # 日期型：前向填充 + 后向填充
                df[col] = series.ffill().bfill()
                log_info(f"日期列 '{col}' 使用前后向填充 {null_count} 个缺失值")

            elif dtype == DataType.TEXT:
                # 文本型：填充空字符串
                df[col] = series.fillna('')
                log_info(f"文本列 '{col}' 使用空字符串填充 {null_count} 个缺失值")
        
        # 删除仍含缺失值的行（主要针对目标变量）
        if target_col:
            # 单次 isnull().sum() 同时给 any 判定和 n_drop 计数
            null_mask = df[target_col].isnull()
            n_drop = int(null_mask.sum())
            if n_drop > 0:
                df = df[~null_mask]
                log_info(f"删除 {n_drop} 行目标变量缺失的样本")

        return df
    
    def _handle_outliers(self, df: pd.DataFrame,
                         profiles: Dict[str, ColumnProfile],
                         target_col: Optional[str] = None) -> pd.DataFrame:
        """处理异常值（用边界值替换而非删除）"""
        # 关键短路径：target_col 为 None 时 `c != None` 对所有 c 都 True，
        # 直接 list(df.columns) 替代 list comprehension + filter，省一次遍历
        feature_cols = list(df.columns) if target_col is None else \
            [c for c in df.columns if c != target_col]
        # 缓存 profiles.keys() 视图，让 `col in profiles` 比重复 dict 哈希快
        # （虽然 dict.__contains__ 是 O(1)，但 .keys() 视图的 __contains__ 是
        # CPython 优化过的 C 路径，比走 dict.__contains__ 略快）
        profile_keys = profiles.keys()

        for col in feature_cols:
            # 合并两次 profiles[] 查询为一次 dict.get(col)：
            # 1) 存在性检查（之前是 col not in profile_keys）
            # 2) inferred_type 数值判断（之前是 profiles[col].inferred_type != NUMERIC）
            # 用 walrus 让 profile 在判断通过后继续使用，省一次 dict 哈希
            profile = profiles.get(col)
            if profile is None or profile.inferred_type != DataType.NUMERIC:
                continue

            series = df[col]
            if self.outlier_method == 'iqr':
                # 一次 quantile 调用拿到 q1/q3，避免 2 次 O(n) 排序/扫描
                q1, q3 = series.quantile([0.25, 0.75])
                q1, q3 = float(q1), float(q3)
                iqr = q3 - q1
                lower, upper = q1 - self.outlier_threshold * iqr, q3 + self.outlier_threshold * iqr
            elif self.outlier_method == 'zscore':
                # 一次 agg 拿到 mean + std，省一次扫描
                mean_std = series.agg(['mean', 'std'])
                mean, std = float(mean_std['mean']), float(mean_std['std'])
                lower, upper = mean - self.outlier_threshold * std, mean + self.outlier_threshold * std
            else:
                continue

            n_outliers = ((series < lower) | (series > upper)).sum()
            if n_outliers > 0:
                df[col] = series.clip(lower, upper)
                log_info(f"数值列 '{col}' 截断 {n_outliers} 个异常值到 [{lower:.4f}, {upper:.4f}]")
        
        return df
    
    def _optimize_types(self, df: pd.DataFrame,
                        profiles: Dict[str, ColumnProfile]) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        for col in df.columns:
            if col not in profiles:
                continue

            dtype = profiles[col].inferred_type

            if dtype == DataType.NUMERIC:
                # 缓存 series：原本 df[col] 多次出现，每次都走 IndexingEngine
                # pandas 实际上会复用 Series view，但显式存一个引用更清晰
                series = df[col]
                # 一次 agg 拿 min + max，省一次 O(n) 扫描
                col_min_max = series.agg(['min', 'max'])
                col_min, col_max = col_min_max['min'], col_min_max['max']
                if pd.notna(col_min) and pd.notna(col_max):
                    candidates = _UNSIGNED_INT_DTYPES if col_min >= 0 else _SIGNED_INT_DTYPES
                    for target_dtype, lo, hi in candidates:
                        if col_min >= lo and col_max <= hi:
                            series = series.astype(target_dtype)
                            df[col] = series
                            break
                else:
                    # 整列 NaN / 全空：列不会贡献有效信息，尝试降精度为 float32 省内存
                    # 仅在原 dtype 是 float64 时降级，避免强制把整数转 float
                    if series.dtype == np.float64:
                        df[col] = series.astype(np.float32)

            elif dtype == DataType.CATEGORY:
                # 优化：df[col] cache + 整除改乘法
                # 1) df[col] 调一次算 nunique，再调一次做 astype — 两次 IndexingEngine
                # 2) n_unique / n_total 涉及浮点除法（虽然 pandas 已经快但每次会构造浮点结果）
                # 改用 n_unique * 2 < n_total 避免浮点除法（同样的语义，对小整数更快）
                series = df[col]
                n_unique = series.nunique()
                n_total = len(df)
                if n_unique * 2 < n_total:  # 类别数占比小于50%时使用category类型
                    df[col] = series.astype('category')

        return df


class DataModule:
    """
    数据模块统一入口
    
    整合数据加载、类型识别、基础清洗的一站式解决方案
    """
    
    def __init__(self) -> None:
        self.loader = DataLoader()
        self.detector = TypeDetector()
        self.cleaner = DataCleaner()
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.profiles: Dict[str, ColumnProfile] = {}
        self.target_col: Optional[str] = None
    
    @timer
    def load(self, file_path: Union[str, os.PathLike],
             auto_chunk: bool = True,
             chunk_size: Optional[int] = None,
             **kwargs) -> 'DataModule':
        """
        加载数据
        
        Args:
            file_path: 文件路径
            auto_chunk: 是否对大文件自动分块读取
            chunk_size: 分块行数
            **kwargs: 额外参数传给 DataLoader
        """
        self.raw_data = self.loader.load(
            file_path,
            auto_chunk=auto_chunk,
            chunk_size=chunk_size,
            **kwargs
        )
        return self
    
    @timer
    def analyze(self) -> 'DataModule':
        """分析数据类型"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        self.profiles = self.detector.analyze_dataframe(self.raw_data)
        return self
    
    @timer
    def clean(self, target_col: Optional[str] = None) -> 'DataModule':
        """清洗数据"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        if not self.profiles:
            self.analyze()
        
        self.target_col = target_col
        self.cleaned_data = self.cleaner.clean(
            self.raw_data, 
            profiles=self.profiles, 
            target_col=target_col
        )
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """获取数据摘要报告"""
        if not self.profiles:
            return {}

        # 用 defaultdict(int) 替代 .get(key, 0) + 1，省一次 hash 查找
        type_dist: Dict[str, int] = defaultdict(int)
        summary = {
            'total_columns': len(self.profiles),
            'total_rows': len(self.raw_data) if self.raw_data is not None else 0,
            'type_distribution': type_dist,
            'high_missing_cols': [],
            'suggestions': []
        }

        for col, profile in self.profiles.items():
            dtype_name = profile.inferred_type.value
            type_dist[dtype_name] += 1

            if profile.null_rate > 0.3:
                summary['high_missing_cols'].append({
                    'column': col,
                    'null_rate': f"{profile.null_rate:.1%}"
                })

            summary['suggestions'].extend(profile.suggestions)

        # 转回普通 dict 以保持 JSON 序列化兼容
        summary['type_distribution'] = dict(type_dist)
        return summary
    
    def print_report(self) -> None:
        """打印分析报告"""
        if not self.profiles:
            print("未分析数据，请先调用 analyze()")
            return
        
        print("=" * 70)
        print("数据模块分析报告".center(60))
        print("=" * 70)
        
        print(f"\n数据规模: {len(self.raw_data)} 行 × {len(self.profiles)} 列")
        
        print("\n【类型分布】")
        # defaultdict 替代 .get(key, 0) + 1，省一次 hash 查找
        type_dist: Dict[str, int] = defaultdict(int)
        for p in self.profiles.values():
            type_dist[p.inferred_type.value] += 1
        for t, c in type_dist.items():
            print(f"  {t}: {c} 列")
        
        print("\n【列详情】")
        for col, profile in self.profiles.items():
            print(f"\n  [{profile.inferred_type.value}] {col}")
            print(f"    数据类型: {profile.dtype}")
            print(f"    缺失值: {profile.null_count} ({profile.null_rate:.1%})")
            print(f"    唯一值: {profile.unique_count}")
            if profile.suggestions:
                print(f"    建议: {'; '.join(profile.suggestions)}")
        
        if self.cleaned_data is not None:
            print(f"\n【清洗结果】")
            print(f"  原始: {self.raw_data.shape}")
            print(f"  清洗后: {self.cleaned_data.shape}")
        
        print("\n" + "=" * 70)
    
    def to_dict(self) -> Dict[str, Any]:
        """导出分析结果为字典"""
        return {
            'summary': self.get_summary(),
            'profiles': {
                col: {
                    'name': p.name,
                    'dtype': p.dtype,
                    'inferred_type': p.inferred_type.value,
                    'null_count': p.null_count,
                    'null_rate': p.null_rate,
                    'unique_count': p.unique_count,
                    'stats': p.stats,
                    'suggestions': p.suggestions
                }
                for col, p in self.profiles.items()
            }
        }
    
    def save_report(self, path: str) -> Optional[str]:
        """保存分析报告为JSON"""
        wm = get_workspace_manager()
        if not wm.allow_disk_write:
            log_warning("磁盘写入已禁用，跳过保存报告")
            return None
        safe_path = wm.write_text(
            path, 
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            subdir='reports'
        )
        if safe_path:
            log_info(f"报告已保存至: {safe_path}")
        return safe_path
