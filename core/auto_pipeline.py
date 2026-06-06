"""
自动缺失处理流程

自动执行以下步骤：
1. 识别目标列
2. 分离 train/test（基于目标列是否缺失）
3. 分类列类型（数值/类别/时间/文本/ID等）
4. 判断缺失率
5. 检测结构性缺失
6. 选择填充策略
7. 执行填充
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd

from core.data_module import TypeDetector, ColumnProfile
from core.missing_engine import (
    MissingPatternClassifier, MissingValueHandler,
    MissingPattern, MissingStrategy,
    ColumnMissingProfile, MissingReport, FastMissingClassifier,
)
from core.progress_bar import progress_iter
from utils.helpers import log_info, log_warning, timer


@dataclass
class PipelineConfig:
    """流程配置"""
    # 目标列
    target_col: Optional[str] = None
    auto_detect_target: bool = True
    target_candidates: List[str] = field(default_factory=lambda: [
        'target', 'label', 'y', 'Target', 'Label', 'TARGET', 'LABEL'
    ])
    
    # 性能模式
    fast_mode: bool = False
    sample_size: Optional[int] = None
    
    # 缺失阈值
    drop_col_threshold: float = 0.95   # 缺失率超过此值删除列
    drop_row_threshold: float = 0.50   # 缺失率超过此值考虑删除行（针对目标列）
    
    # 结构性缺失
    structural_threshold: float = 0.90
    structural_min_support: int = 10
    
    # 策略覆盖
    strategy_overrides: Dict[str, MissingStrategy] = field(default_factory=dict)
    
    # ID列排除
    id_pattern_hints: List[str] = field(default_factory=lambda: [
        'id', 'ID', 'Id', 'index', '编号', '序号'
    ])
    
    # 磁盘写入开关
    allow_disk_write: bool = True


class AutoMissingPipeline:
    """
    自动缺失处理流程
    
    使用方式：
        pipeline = AutoMissingPipeline(config)
        train_df, test_df, report = pipeline.run(df)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = True) -> None:
        self.config = config or PipelineConfig()
        self.verbose = verbose
        self.classifier = MissingPatternClassifier(
            structural_threshold=self.config.structural_threshold,
            structural_min_support=self.config.structural_min_support
        )
        self.fast_classifier = FastMissingClassifier(
            sample_size=self.config.sample_size or (10000 if self.config.fast_mode else None)
        )
        self.handler = MissingValueHandler()
        self.detector = TypeDetector()
        
        # 初始化工作空间
        from core.workspace_manager import set_workspace_config
        set_workspace_config(allow_disk_write=self.config.allow_disk_write)
        
        # 状态
        self.raw_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.column_profiles: Dict[str, ColumnMissingProfile] = {}
        self.type_profiles: Dict[str, ColumnProfile] = {}
        self.report: Optional[MissingReport] = None
    
    @timer
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MissingReport]:
        """
        执行完整自动流程
        
        Args:
            df: 原始数据（训练集+测试集合并）
            
        Returns:
            (train_df, test_df, report)
            - test_df 为 None 表示无法分离或没有测试集
        """
        self.raw_df = df.copy()
        log_info(f"=" * 60)
        log_info("自动缺失处理流程启动")
        log_info(f"输入数据: {df.shape}")
        
        # Step 1: 识别目标列
        target_col = self._identify_target(df)
        log_info(f"[Step 1] 目标列识别: {target_col or '未识别'}")
        
        # Step 2: 分离 train/test
        train_df, test_df = self._split_train_test(df, target_col)
        log_info(f"[Step 2] 数据分割: 训练集={len(train_df) if train_df is not None else 0}, 测试集={len(test_df) if test_df is not None else 0}")
        
        # Step 3: 分类列类型
        self.type_profiles = self.detector.analyze_dataframe(df)
        log_info(f"[Step 3] 列类型分类完成: {len(self.type_profiles)} 列")
        
        # Step 4-6: 缺失分析 + 策略选择
        if self.config.fast_mode:
            self.column_profiles = self.fast_classifier.classify_all(df, target_col)
        else:
            self.column_profiles = self._classify_all_missing(df, target_col)
        
        log_info(f"[Step 4-6] 缺失分析完成")
        
        # Step 7: 执行处理
        processed_df = self._execute_processing(df)
        
        # 重新分离（基于处理后的数据）
        if target_col:
            train_mask = processed_df[target_col].notna()
            final_train = processed_df[train_mask].copy()
            final_test = processed_df[~train_mask].copy() if (~train_mask).any() else None
        else:
            final_train = processed_df
            final_test = None
        
        self.train_df = final_train
        self.test_df = final_test
        
        # 生成报告
        self.report = self._generate_report(df, target_col)
        
        log_info(f"[完成] 训练集: {final_train.shape}, 测试集: {final_test.shape if final_test is not None else None}")
        log_info(f"=" * 60)
        
        return final_train, final_test, self.report
    
    def _identify_target(self, df: pd.DataFrame) -> Optional[str]:
        """
        智能识别目标列
        
        策略优先级：
        1. 配置中显式指定
        2. 列名匹配常见目标列名（target/label/y等）
        3. 列在训练集有值但测试集全空
        4. 列名在数据末尾（常见放置位置）
        """
        # 1. 显式指定
        if self.config.target_col and self.config.target_col in df.columns:
            return self.config.target_col
        
        # 2. 列名匹配
        for candidate in self.config.target_candidates:
            if candidate in df.columns:
                return candidate
        
        # 3. 查找训练有值测试空的列（训练集+测试集合并的情况）
        # 简单启发：如果某列缺失率在30%-70%之间，且列名在末尾，可能是目标
        best_candidate = None
        best_score = -1
        
        for col in df.columns:
            # 排除ID列
            if any(hint in col for hint in self.config.id_pattern_hints):
                continue
            
            missing_rate = df[col].isnull().sum() / len(df)
            
            # 缺失率在合理范围（可能是部分有标签的数据）
            if 0.1 <= missing_rate <= 0.9:
                score = 0
                
                # 数值型且唯一值少：更可能是分类目标
                if pd.api.types.is_numeric_dtype(df[col]):
                    n_unique = df[col].nunique()
                    if n_unique <= 20:
                        score += 10
                
                # 列名在末尾加分
                col_idx = list(df.columns).index(col)
                if col_idx >= len(df.columns) - 3:
                    score += 5
                
                # 列名短加分（y, label等通常较短）
                if len(col) <= 6:
                    score += 3
                
                if score > best_score:
                    best_score = score
                    best_candidate = col
        
        if best_candidate and best_score >= 8:
            log_info(f"自动识别目标列: {best_candidate} (得分={best_score})")
            self.config.target_col = best_candidate
            return best_candidate
        
        if self.config.auto_detect_target:
            log_warning("未能自动识别目标列，将整份数据作为训练集处理")
        
        return None
    
    def _split_train_test(self, df: pd.DataFrame, 
                          target_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        分离训练集和测试集
        
        基于目标列：非空为训练，空为测试
        """
        if target_col is None or target_col not in df.columns:
            return df, None
        
        train_mask = df[target_col].notna()
        train_df = df[train_mask].copy()
        test_df = df[~train_mask].copy() if (~train_mask).any() else None
        
        return train_df, test_df
    
    def _classify_all_missing(self, df: pd.DataFrame,
                              target_col: Optional[str]) -> Dict[str, ColumnMissingProfile]:
        """对所有列进行缺失分类"""
        profiles: Dict[str, ColumnMissingProfile] = {}
        
        for col in progress_iter(df.columns, desc="缺失分类", disable=not self.verbose):
            profile = self.classifier.classify(
                df, col,
                target_col=target_col,
                sample_size=self.config.sample_size
            )
            profiles[col] = profile
        
        return profiles
    
    def _execute_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行缺失处理"""
        df = df.copy()
        
        # 按策略优先级排序：先处理结构性（避免影响全局统计量），再处理真缺失
        processing_order: List[Tuple[int, str, MissingStrategy, ColumnMissingProfile]] = []
        
        for col, profile in self.column_profiles.items():
            if col not in df.columns:
                continue
            
            # 应用策略覆盖
            strategy = self.config.strategy_overrides.get(col, profile.recommended_strategy)
            
            # 高缺失率列直接删除
            if profile.missing_rate > self.config.drop_col_threshold:
                strategy = MissingStrategy.DROP_COL
                log_warning(f"[{col}] 缺失率{profile.missing_rate:.1%}超过阈值，删除该列")
            
            priority = 0
            if profile.pattern == MissingPattern.STRUCTURAL:
                priority = 1
            elif profile.pattern == MissingPattern.TRUE_MISSING:
                priority = 2
            elif profile.pattern == MissingPattern.TARGET_MISSING:
                priority = 3  # 最后处理，保留NaN
            
            processing_order.append((priority, col, strategy, profile))
        
        processing_order.sort(key=lambda x: x[0])
        
        for priority, col, strategy, profile in processing_order:
            if strategy == MissingStrategy.DROP_COL:
                df = self.handler.handle(df, col, strategy)
                continue
            
            # 获取结构性规则（如果适用）
            rule = profile.structural_rules[0] if profile.structural_rules else None
            
            df = self.handler.handle(df, col, strategy, rule=rule)
        
        return df
    
    def _generate_report(self, raw_df: pd.DataFrame, 
                         target_col: Optional[str]) -> MissingReport:
        """生成完整报告"""
        train_mask = raw_df[target_col].notna() if target_col else pd.Series([True] * len(raw_df))
        
        report = MissingReport(
            total_rows=len(raw_df),
            total_cols=len(raw_df.columns),
            target_col=target_col,
            train_rows=train_mask.sum(),
            test_rows=(~train_mask).sum() if target_col else 0,
            column_profiles=self.column_profiles
        )
        
        # 统计摘要
        pattern_counts: Dict[str, int] = {}
        strategy_counts: Dict[str, int] = {}
        total_missing_handled = 0
        
        for profile in self.column_profiles.values():
            p_name = profile.pattern.value
            pattern_counts[p_name] = pattern_counts.get(p_name, 0) + 1
            
            s_name = profile.recommended_strategy.value
            strategy_counts[s_name] = strategy_counts.get(s_name, 0) + 1
            
            if profile.pattern != MissingPattern.NONE:
                total_missing_handled += profile.missing_count
        
        report.execution_summary = {
            'pattern_distribution': pattern_counts,
            'strategy_distribution': strategy_counts,
            'total_missing_handled': total_missing_handled,
            'structural_rules_found': sum(
                len(p.structural_rules) for p in self.column_profiles.values()
            )
        }
        
        return report
    
    def print_report(self) -> None:
        """打印可视化报告"""
        if self.report is None:
            print("尚未运行流程，请先调用 run()")
            return
        
        r = self.report
        print("\n" + "=" * 70)
        print("缺失值智能处理报告".center(60))
        print("=" * 70)
        
        print(f"\n📊 数据概览")
        print(f"   总行数: {r.total_rows}")
        print(f"   总列数: {r.total_cols}")
        print(f"   目标列: {r.target_col or '未识别'}")
        print(f"   训练集: {r.train_rows} 行")
        print(f"   测试集: {r.test_rows} 行")
        
        print(f"\n📋 缺失模式分布")
        for pattern, count in r.execution_summary.get('pattern_distribution', {}).items():
            print(f"   {pattern}: {count} 列")
        
        print(f"\n🔧 处理策略分布")
        for strategy, count in r.execution_summary.get('strategy_distribution', {}).items():
            print(f"   {strategy}: {count} 列")
        
        print(f"\n🔍 列详情")
        for col, profile in r.column_profiles.items():
            if profile.pattern == MissingPattern.NONE:
                continue
            
            print(f"\n   [{profile.pattern.value}] {col}")
            print(f"      缺失: {profile.missing_count} ({profile.missing_rate:.1%})")
            print(f"      策略: {profile.recommended_strategy.value}")
            if profile.structural_rules:
                print(f"      结构性规则: {profile.structural_rules[0]}")
            print(f"      说明: {profile.strategy_reason}")
        
        print("\n" + "=" * 70)
    
    def get_train_test(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """获取处理后的训练集和测试集"""
        return self.train_df, self.test_df
