"""
缺失值智能分析引擎

功能：
1. 缺失分类机制：真缺失 / 结构性缺失（业务缺失）/ 目标缺失
2. 结构性缺失检测：基于条件概率、互信息、卡方检验
3. 分类型处理策略
4. 自动流程集成
5. 性能优化：分层采样 + 惰性执行 + 缓存
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import time

import pandas as pd
from core.progress_bar import progress_iter
from core.workspace_manager import get_workspace_manager
from sklearn.feature_selection import mutual_info_classif

from utils.helpers import log_info, log_warning, timer


# =============================================================================
# 枚举与数据类
# =============================================================================

class MissingPattern(Enum):
    """缺失模式分类"""
    NONE = "无缺失"              # 无缺失值
    TRUE_MISSING = "真缺失"      # 随机缺失（MCAR/MAR）
    STRUCTURAL = "结构性缺失"     # 业务/条件导致的结构性缺失
    TARGET_MISSING = "目标缺失"   # 需要预测的目标列缺失
    MIXED = "混合缺失"           # 同时存在多种缺失模式


class MissingStrategy(Enum):
    """缺失处理策略"""
    # 数值型
    MEDIAN = "中位数填充"
    MEAN = "均值填充"
    KNN = "K近邻填充"
    ITERATIVE = "迭代插补（MICE）"
    FLAG_MEDIAN = "标记+中位数填充"
    
    # 类别型
    MODE = "众数填充"
    CONSTANT = "常数填充（未知）"
    NEW_CATEGORY = "新增类别"
    
    # 时间型
    FFILL = "前向填充"
    BFILL = "后向填充"
    INTERPOLATE = "插值填充"
    
    # 结构性缺失
    CONDITIONAL_MODE = "条件众数填充"
    CONDITIONAL_MEDIAN = "条件中位数填充"
    GROUP_IMPUTE = "分组填充"
    DERIVED_FEATURE = "衍生特征（是否缺失）"
    
    # 目标缺失
    PREDICT = "待预测（保留NaN）"
    
    # 通用
    DROP_ROW = "删除行"
    DROP_COL = "删除列"
    NONE = "不处理"


@dataclass
class StructuralRule:
    """结构性缺失规则"""
    condition_col: str           # 条件列
    condition_value: Any         # 条件值
    confidence: float            # 置信度（条件概率）
    support: int                 # 支持度（样本数）
    
    def __repr__(self) -> str:
        return f"当 [{self.condition_col} = {self.condition_value}] 时缺失，置信度={self.confidence:.2%}, 支持度={self.support}"


@dataclass
class ColumnMissingProfile:
    """列缺失画像"""
    col_name: str
    total_rows: int = 0
    missing_count: int = 0
    missing_rate: float = 0.0
    
    # 缺失分类
    pattern: MissingPattern = MissingPattern.NONE
    pattern_confidence: float = 1.0
    
    # 结构性缺失规则
    structural_rules: List[StructuralRule] = field(default_factory=list)
    structural_primary_col: Optional[str] = None  # 最主要的条件列
    
    # 关联分析
    correlated_cols: List[Tuple[str, float]] = field(default_factory=list)  # (列名, 关联强度)
    
    # 策略
    recommended_strategy: MissingStrategy = MissingStrategy.NONE
    strategy_reason: str = ""
    
    # 执行元数据
    execution_time_ms: float = 0.0
    sample_size: int = 0  # 实际分析的样本数（分层采样时）


@dataclass  
class MissingReport:
    """缺失分析完整报告"""
    total_rows: int = 0
    total_cols: int = 0
    target_col: Optional[str] = None
    train_rows: int = 0
    test_rows: int = 0
    column_profiles: Dict[str, ColumnMissingProfile] = field(default_factory=dict)
    execution_summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 性能优化：惰性执行 + 缓存
# =============================================================================

class CacheManager:
    """内存缓存管理器"""
    
    def __init__(self, max_size: int = 100) -> None:
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, func_name: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            # LRU简单实现：移除最早的
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value
    
    def cached(self, func: Callable) -> Callable:
        """缓存装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = self._make_key(func.__name__, *args, **kwargs)
            cached = self.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        return wrapper
    
    def stats(self) -> Dict[str, int]:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0,
            'size': len(self._cache)
        }


class LazyExecutor:
    """惰性执行器：延迟计算直到结果被访问"""
    
    def __init__(self, compute_fn: Callable, *args, **kwargs) -> None:
        self._compute_fn = compute_fn
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._computed = False
        self._compute_time = 0.0
    
    @property
    def result(self) -> Any:
        if not self._computed:
            start = time.time()
            self._result = self._compute_fn(*self._args, **self._kwargs)
            self._compute_time = time.time() - start
            self._computed = True
        return self._result
    
    def invalidate(self) -> None:
        self._computed = False
        self._result = None


def lazy_property(fn: Callable) -> Callable:
    """惰性属性装饰器"""
    attr_name = f"_lazy_{fn.__name__}"
    
    @property
    @wraps(fn)
    def wrapper(self) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return wrapper


# =============================================================================
# 缺失分类器
# =============================================================================

class MissingPatternClassifier:
    """
    缺失模式分类器
    
    核心逻辑：
    1. 目标缺失：列被标记为目标列，且存在缺失 → TARGET_MISSING
    2. 结构性缺失：缺失与某条件列高度相关（条件概率 > threshold）→ STRUCTURAL
    3. 真缺失：上述都不是，且存在随机缺失 → TRUE_MISSING
    4. 混合缺失：同时满足多种模式 → MIXED
    """
    
    def __init__(self,
                 structural_threshold: float = 0.90,
                 structural_min_support: int = 10,
                 mi_threshold: float = 0.15,
                 max_correlated_cols: int = 5) -> None:
        """
        Args:
            structural_threshold: 结构性缺失条件概率阈值（默认90%）
            structural_min_support: 结构性缺失最小支持度
            mi_threshold: 互信息阈值，用于判断关联强度
            max_correlated_cols: 最大关联列数
        """
        self.structural_threshold = structural_threshold
        self.structural_min_support = structural_min_support
        self.mi_threshold = mi_threshold
        self.max_correlated_cols = max_correlated_cols
        self._cache = CacheManager(max_size=200)
    
    @timer
    def classify(self,
                 df: pd.DataFrame,
                 col: str,
                 target_col: Optional[str] = None,
                 candidate_cols: Optional[List[str]] = None,
                 sample_size: Optional[int] = None) -> ColumnMissingProfile:
        """
        对指定列进行缺失模式分类
        
        Args:
            df: 数据框
            col: 待分析列
            target_col: 目标列名（如果该列是目标列）
            candidate_cols: 候选条件列（默认排除id/文本/常量列）
            sample_size: 分层采样大小（大数据集优化）
            
        Returns:
            ColumnMissingProfile
        """
        start_time = time.time()
        
        # 采样（如果需要）
        analysis_df, is_sampled = self._sample_if_needed(df, sample_size)
        
        profile = ColumnMissingProfile(
            col_name=col,
            total_rows=len(df),
            sample_size=len(analysis_df)
        )
        
        series = analysis_df[col]
        n_missing = series.isnull().sum()
        profile.missing_count = int(n_missing)
        profile.missing_rate = n_missing / len(analysis_df)
        
        if n_missing == 0:
            profile.pattern = MissingPattern.NONE
            profile.recommended_strategy = MissingStrategy.NONE
            profile.execution_time_ms = (time.time() - start_time) * 1000
            return profile
        
        # 1. 检查是否为目标缺失
        is_target = (col == target_col) if target_col else False
        if is_target:
            # 检查是否测试集缺失
            missing_pattern = self._check_target_missing(analysis_df, col)
            if missing_pattern:
                profile.pattern = MissingPattern.TARGET_MISSING
                profile.recommended_strategy = MissingStrategy.PREDICT
                profile.strategy_reason = "目标列缺失，待预测"
                profile.execution_time_ms = (time.time() - start_time) * 1000
                return profile
        
        # 2. 检测结构性缺失
        candidate_cols = candidate_cols or self._select_candidate_cols(analysis_df, col)
        structural_rules = self._detect_structural_missing(analysis_df, col, candidate_cols)
        
        if structural_rules:
            profile.structural_rules = structural_rules
            profile.structural_primary_col = structural_rules[0].condition_col
            profile.pattern = MissingPattern.STRUCTURAL
            profile.pattern_confidence = structural_rules[0].confidence
            
            # 选择结构性缺失策略
            strategy, reason = self._select_structural_strategy(
                analysis_df, col, structural_rules[0]
            )
            profile.recommended_strategy = strategy
            profile.strategy_reason = reason
            
            profile.execution_time_ms = (time.time() - start_time) * 1000
            return profile
        
        # 3. 计算与其他列的关联（用于判断真缺失 vs 混合）
        correlated = self._calculate_correlations(analysis_df, col, candidate_cols)
        profile.correlated_cols = correlated[:self.max_correlated_cols]
        
        # 4. 判定为真缺失
        profile.pattern = MissingPattern.TRUE_MISSING
        strategy, reason = self._select_true_missing_strategy(analysis_df, col, profile)
        profile.recommended_strategy = strategy
        profile.strategy_reason = reason
        
        profile.execution_time_ms = (time.time() - start_time) * 1000
        return profile
    
    def _sample_if_needed(self, df: pd.DataFrame, 
                          sample_size: Optional[int]) -> Tuple[pd.DataFrame, bool]:
        """分层采样（保留缺失比例）"""
        if sample_size is None or len(df) <= sample_size:
            return df, False
        
        # 简单随机采样（保留缺失结构）
        return df.sample(n=sample_size, random_state=42), True
    
    def _select_candidate_cols(self, df: pd.DataFrame, 
                                exclude_col: str) -> List[str]:
        """选择候选条件列（排除高基数列和文本列）"""
        candidates = []
        for col in df.columns:
            if col == exclude_col:
                continue
            
            n_unique = df[col].nunique(dropna=True)
            n_total = len(df)
            
            # 排除高基数类别（>50）和全唯一列（疑似ID）
            if n_unique > min(50, n_total * 0.5):
                continue
            
            # 排除几乎全空列
            null_rate = df[col].isnull().sum() / n_total
            if null_rate > 0.9:
                continue
            
            candidates.append(col)
        
        return candidates
    
    def _detect_structural_missing(self,
                                    df: pd.DataFrame,
                                    target_col: str,
                                    candidate_cols: List[str]) -> List[StructuralRule]:
        """
        检测结构性缺失规则

        核心逻辑：对于候选列的每个取值，计算 P(目标列缺失 | 候选列=取值)
        如果某取值下条件概率 > threshold，则判定为结构性缺失规则

        优化：原实现 O(V·n) 内循环（每个 val 做 3 次全列扫描），
        现用单次 groupby + agg 一次性拿到所有 val 的 sum/count，O(n) 总开销。
        """
        rules = []
        missing_mask = df[target_col].isnull()
        n_missing = missing_mask.sum()

        if n_missing < self.structural_min_support:
            return rules

        for cond_col in candidate_cols:
            series = df[cond_col]

            # 只考虑类别型（包括布尔型）
            if series.dtype.kind in 'fiucb' and series.nunique() > 10:
                continue  # 数值型且唯一值多，跳过

            # 单次 groupby：每 val 的非空样本数 + 缺失样本数
            # 关键：先 dropna(cond_col) 排除条件列为空的情况，避免分母偏差
            mask_combined = missing_mask & series.notna()
            valid = pd.DataFrame({'miss': mask_combined})
            # 保留原索引以便 groupby(series)
            valid['cond'] = series.values
            valid = valid[valid['cond'].notna()]
            if valid.empty:
                continue
            grouped = valid.groupby('cond', sort=False)['miss'].agg(['sum', 'count'])
            # 向量化筛选：min_support 与 threshold 一次过滤
            pass_mask = grouped['count'] >= self.structural_min_support
            prob_series = grouped['sum'] / grouped['count']
            valid_rules_mask = pass_mask & (prob_series >= self.structural_threshold)
            for val, row in grouped[valid_rules_mask].iterrows():
                rules.append(StructuralRule(
                    condition_col=cond_col,
                    condition_value=val,
                    confidence=row['sum'] / row['count'],
                    support=int(row['count'])
                ))

        # 按置信度和支持度排序
        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        return rules
    
    def _check_target_missing(self, df: pd.DataFrame, col: str) -> bool:
        """检查是否为目标缺失（有训练值有测试值空的分割模式）"""
        # 简单判断：如果该列存在非空值，说明是训练+测试混合
        # 更复杂的可以基于index或其他标记判断
        return df[col].notna().sum() > 0 and df[col].isna().sum() > 0
    
    def _calculate_correlations(self,
                                 df: pd.DataFrame,
                                 col: str,
                                 candidate_cols: List[str]) -> List[Tuple[str, float]]:
        """计算缺失与其他列的关联强度（基于互信息）"""
        correlations = []
        missing_mask = df[col].isnull().astype(int)
        
        for other_col in candidate_cols:
            if other_col == col:
                continue
            
            try:
                other_series = df[other_col]
                
                # 对数值型做离散化
                if other_series.dtype.kind in 'fi':
                    other_series = pd.qcut(other_series, q=10, duplicates='drop', labels=False)
                
                # 计算互信息
                mi = mutual_info_classif(
                    other_series.fillna(-999).values.reshape(-1, 1),
                    missing_mask.values,
                    random_state=42
                )[0]
                
                if mi > 0.001:  # 过滤噪声
                    correlations.append((other_col, float(mi)))
            except Exception:
                continue
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        return correlations
    
    def _select_structural_strategy(self,
                                     df: pd.DataFrame,
                                     col: str,
                                     rule: StructuralRule) -> Tuple[MissingStrategy, str]:
        """为结构性缺失选择策略"""
        series = df[col]
        
        if pd.api.types.is_numeric_dtype(series):
            strategy = MissingStrategy.CONDITIONAL_MEDIAN
            reason = f"数值型结构性缺失：当[{rule.condition_col}={rule.condition_value}]时用该组中位数填充"
        elif pd.api.types.is_datetime64_any_dtype(series):
            strategy = MissingStrategy.GROUP_IMPUTE
            reason = f"时间型结构性缺失：按[{rule.condition_col}]分组填充"
        else:
            strategy = MissingStrategy.CONDITIONAL_MODE
            reason = f"类别型结构性缺失：当[{rule.condition_col}={rule.condition_value}]时用该组众数填充"
        
        return strategy, reason
    
    def _select_true_missing_strategy(self,
                                       df: pd.DataFrame,
                                       col: str,
                                       profile: ColumnMissingProfile) -> Tuple[MissingStrategy, str]:
        """为真缺失选择策略"""
        series = df[col]
        missing_rate = profile.missing_rate
        
        # 高缺失率：考虑删除或标记
        if missing_rate > 0.8:
            if pd.api.types.is_numeric_dtype(series):
                return MissingStrategy.FLAG_MEDIAN, "缺失率>80%，标记缺失后中位数填充"
            else:
                return MissingStrategy.NEW_CATEGORY, "缺失率>80%，新增'缺失'类别"
        
        # 数值型
        if pd.api.types.is_numeric_dtype(series):
            # 偏态严重用中位数，否则用均值
            skewness = series.dropna().skew()
            if abs(skewness) > 2:
                return MissingStrategy.MEDIAN, f"数值型右偏(skew={skewness:.2f})，中位数填充更稳健"
            elif missing_rate < 0.05:
                return MissingStrategy.MEAN, "缺失率<5%，均值填充对分布影响小"
            else:
                return MissingStrategy.MEDIAN, "数值型真缺失，中位数填充"
        
        # 时间型
        if pd.api.types.is_datetime64_any_dtype(series):
            return MissingStrategy.INTERPOLATE, "时间序列，插值填充"
        
        # 类别型
        n_unique = series.nunique()
        if n_unique <= 2:
            return MissingStrategy.MODE, "二值类别，众数填充"
        else:
            return MissingStrategy.NEW_CATEGORY, f"类别型({n_unique}类)，新增'缺失'类别避免引入偏差"


# =============================================================================
# 缺失值处理器
# =============================================================================

class MissingValueHandler:
    """缺失值处理器：执行具体的填充策略"""

    # 类级常量：handler_map 提到此处，避免每次 handle 调用重建 13 项 dict
    _HANDLER_MAP = {
        MissingStrategy.MEAN: '_fill_mean',
        MissingStrategy.MEDIAN: '_fill_median',
        MissingStrategy.MODE: '_fill_mode',
        MissingStrategy.CONSTANT: '_fill_constant',
        MissingStrategy.NEW_CATEGORY: '_fill_new_category',
        MissingStrategy.FLAG_MEDIAN: '_fill_flag_median',
        MissingStrategy.FFILL: '_fill_ffill',
        MissingStrategy.BFILL: '_fill_bfill',
        MissingStrategy.INTERPOLATE: '_fill_interpolate',
        MissingStrategy.CONDITIONAL_MEDIAN: '_fill_conditional_median',
        MissingStrategy.CONDITIONAL_MODE: '_fill_conditional_mode',
        MissingStrategy.GROUP_IMPUTE: '_fill_group_impute',
        MissingStrategy.DERIVED_FEATURE: '_fill_derived',
        MissingStrategy.DROP_ROW: '_drop_rows',
        MissingStrategy.DROP_COL: '_drop_col',
    }

    def __init__(self, cache: Optional[CacheManager] = None) -> None:
        self.cache = cache or CacheManager()
        self._imputer_cache: Dict[str, Any] = {}
    
    def handle(self,
               df: pd.DataFrame,
               col: str,
               strategy: MissingStrategy,
               rule: Optional[StructuralRule] = None,
               **kwargs) -> pd.DataFrame:
        """
        执行缺失处理
        
        Args:
            df: 数据框
            col: 目标列
            strategy: 处理策略
            rule: 结构性缺失规则（如果适用）
            **kwargs: 额外参数
            
        Returns:
            处理后的数据框
        """
        if strategy == MissingStrategy.NONE:
            return df
        
        if strategy == MissingStrategy.PREDICT:
            log_info(f"[{col}] 目标缺失，保留NaN待预测")
            return df
        
        handler_map = {
            k: getattr(self, v) for k, v in self._HANDLER_MAP.items()
        }
        
        handler = handler_map.get(strategy)
        if handler is None:
            log_warning(f"[{col}] 未知策略 {strategy}，跳过处理")
            return df
        
        return handler(df, col, rule=rule, **kwargs)
    
    def _fill_mean(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        log_info(f"[{col}] 均值填充: {mean_val:.4f}")
        return df
    
    def _fill_median(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        log_info(f"[{col}] 中位数填充: {median_val:.4f}")
        return df
    
    def _fill_mode(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
            log_info(f"[{col}] 众数填充: {mode_val[0]}")
        return df
    
    def _fill_constant(self, df: pd.DataFrame, col: str, 
                       constant: str = "未知", **kwargs) -> pd.DataFrame:
        df[col] = df[col].fillna(constant)
        log_info(f"[{col}] 常数填充: '{constant}'")
        return df
    
    def _fill_new_category(self, df: pd.DataFrame, col: str, 
                           label: str = "__MISSING__", **kwargs) -> pd.DataFrame:
        df[col] = df[col].astype(str).replace('nan', label).replace('None', label)
        df.loc[df[col].isnull(), col] = label
        # 转回category类型
        df[col] = df[col].astype('category')
        log_info(f"[{col}] 新增缺失类别: '{label}'")
        return df
    
    def _fill_flag_median(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        flag_col = f"{col}_is_missing"
        df[flag_col] = df[col].isnull().astype(int)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        log_info(f"[{col}] 标记缺失+中位数填充({median_val:.4f})，生成标记列: {flag_col}")
        return df
    
    def _fill_ffill(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        df[col] = df[col].ffill()
        log_info(f"[{col}] 前向填充")
        return df
    
    def _fill_bfill(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        df[col] = df[col].bfill()
        log_info(f"[{col}] 后向填充")
        return df
    
    def _fill_interpolate(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        df[col] = df[col].interpolate(method='linear')
        # 首尾可能仍有缺失，用ffill/bfill补充
        df[col] = df[col].ffill().bfill()
        log_info(f"[{col}] 线性插值填充")
        return df
    
    def _fill_conditional_median(self, df: pd.DataFrame, col: str,
                                  rule: Optional[StructuralRule] = None, **kwargs) -> pd.DataFrame:
        if rule is None:
            return self._fill_median(df, col)
        
        condition = df[rule.condition_col] == rule.condition_value
        median_val = df.loc[condition, col].median()
        
        # 对该条件下缺失的行填充
        fill_mask = condition & df[col].isnull()
        df.loc[fill_mask, col] = median_val
        
        # 其他缺失用全局中位数
        other_mask = df[col].isnull()
        if other_mask.any():
            global_median = df[col].median()
            df.loc[other_mask, col] = global_median
        
        log_info(f"[{col}] 条件中位数填充: [{rule.condition_col}={rule.condition_value}] -> {median_val:.4f}")
        return df
    
    def _fill_conditional_mode(self, df: pd.DataFrame, col: str,
                                rule: Optional[StructuralRule] = None, **kwargs) -> pd.DataFrame:
        if rule is None:
            return self._fill_mode(df, col)
        
        condition = df[rule.condition_col] == rule.condition_value
        mode_val = df.loc[condition, col].mode()
        fill_value = mode_val[0] if len(mode_val) > 0 else "未知"
        
        fill_mask = condition & df[col].isnull()
        df.loc[fill_mask, col] = fill_value
        
        other_mask = df[col].isnull()
        if other_mask.any():
            global_mode = df[col].mode()
            df.loc[other_mask, col] = global_mode[0] if len(global_mode) > 0 else "未知"
        
        log_info(f"[{col}] 条件众数填充: [{rule.condition_col}={rule.condition_value}] -> {fill_value}")
        return df
    
    def _fill_group_impute(self, df: pd.DataFrame, col: str,
                            group_col: Optional[str] = None,
                            rule: Optional[StructuralRule] = None, **kwargs) -> pd.DataFrame:
        if rule:
            group_col = rule.condition_col
        
        if group_col and group_col in df.columns:
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median() if pd.api.types.is_numeric_dtype(x) else x.mode()[0] if len(x.mode()) > 0 else x)
            )
            log_info(f"[{col}] 按 [{group_col}] 分组填充")
        else:
            return self._fill_median(df, col)
        return df
    
    def _fill_derived(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        flag_col = f"{col}_is_missing"
        df[flag_col] = df[col].isnull().astype(int)
        log_info(f"[{col}] 生成衍生特征: {flag_col}")
        return df
    
    def _drop_rows(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        n_before = len(df)
        df = df.dropna(subset=[col])
        log_info(f"[{col}] 删除缺失行: {n_before} -> {len(df)}")
        return df
    
    def _drop_col(self, df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
        df = df.drop(columns=[col])
        log_info(f"[{col}] 删除该列")
        return df


# =============================================================================
# 快速分类器（性能分级模式）
# =============================================================================

class FastMissingClassifier:
    """
    快速缺失分类器（性能分级模式）
    
    适用于大数据集，采用：
    - 分层采样
    - 减少候选列
    - 简化统计检验
    """
    
    def __init__(self, sample_size: int = 10000, max_candidates: int = 20) -> None:
        self.sample_size = sample_size
        self.max_candidates = max_candidates
        self.base_classifier = MissingPatternClassifier()
    
    def classify_all(self,
                     df: pd.DataFrame,
                     target_col: Optional[str] = None) -> Dict[str, ColumnMissingProfile]:
        """快速分类所有列"""
        profiles = {}
        
        # 优先分析高缺失率列
        cols_by_missing = sorted(
            df.columns,
            key=lambda c: df[c].isnull().sum(),
            reverse=True
        )
        
        for col in progress_iter(cols_by_missing, desc="快速缺失分类", total=len(cols_by_missing), disable=False):
            profile = self.base_classifier.classify(
                df, col,
                target_col=target_col,
                sample_size=self.sample_size
            )
            profiles[col] = profile
        
        return profiles


# =============================================================================
# 结果导出
# =============================================================================

def export_missing_report(report: MissingReport, path: str) -> Optional[str]:
    """导出缺失分析报告为JSON"""
    wm = get_workspace_manager()
    if not wm.allow_disk_write:
        log_warning("磁盘写入已禁用，跳过导出报告")
        return None
    
    data = {
        'total_rows': report.total_rows,
        'total_cols': report.total_cols,
        'target_col': report.target_col,
        'train_rows': report.train_rows,
        'test_rows': report.test_rows,
        'columns': {}
    }
    
    for col, profile in report.column_profiles.items():
        data['columns'][col] = {
            'missing_count': profile.missing_count,
            'missing_rate': f"{profile.missing_rate:.2%}",
            'pattern': profile.pattern.value,
            'pattern_confidence': profile.pattern_confidence,
            'structural_rules': [
                {
                    'condition_col': r.condition_col,
                    'condition_value': str(r.condition_value),
                    'confidence': f"{r.confidence:.2%}",
                    'support': r.support
                }
                for r in profile.structural_rules
            ],
            'recommended_strategy': profile.recommended_strategy.value,
            'strategy_reason': profile.strategy_reason,
            'execution_time_ms': profile.execution_time_ms
        }
    
    content = json.dumps(data, ensure_ascii=False, indent=2)
    safe_path = wm.write_text(path, content, subdir='reports')
    if safe_path:
        log_info(f"缺失分析报告已保存: {safe_path}")
    return safe_path
