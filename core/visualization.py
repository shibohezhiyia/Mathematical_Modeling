"""
Mathematical Modeling - 可视化模块

提供数据探索、模型结果、自动评估决策的全方位可视化能力。

设计原则：
1. 自动中文化：检测并使用系统可用中文字体
2. 安全保存：所有文件操作通过 workspace_manager，遵循 allow_disk_write
3. 优雅降级：matplotlib 不可用时静默跳过
4. 统一风格：一致的颜色主题和布局

使用方式：
    from core.visualization import DataVisualizer, ModelVisualizer, EvaluationVisualizer
    
    dv = DataVisualizer()
    dv.plot_distribution(df, 'age', save_path='age_dist.png')
    dv.plot_correlation_heatmap(df, save_path='corr.png')
    
    mv = ModelVisualizer()
    mv.plot_feature_importance(result, save_path='fi.png')
    mv.plot_roc_curve(cv_results, save_path='roc.png')
    
    ev = EvaluationVisualizer()
    ev.plot_radar_comparison(decision_report, save_path='radar.png')
"""

import os
from math import pi
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

from core.workspace_manager import get_workspace_manager
from utils.helpers import log_info, log_warning
from sklearn.metrics import confusion_matrix, r2_score, roc_curve, auc

# =============================================================================
# Matplotlib 初始化与中文支持
# =============================================================================

_MPL_AVAILABLE = False
_SNS_AVAILABLE = False

# 颜色主题（统一配色）
_COLOR_THEME = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'neutral': '#6C757D',
    'background': '#F8F9FA',
    'palette': ['#2E86AB', '#A23B72', '#28A745', '#F18F01', '#C73E1D',
                '#6F2DBD', '#3B1F2B', '#95C623', '#EE4266', '#540D6E']
}


def _init_matplotlib() -> bool:
    """延迟初始化 matplotlib，配置中文字体"""
    global _MPL_AVAILABLE, _SNS_AVAILABLE
    if _MPL_AVAILABLE:
        return True
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无头模式，避免弹窗
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        
        # 查找中文字体（优先级：Microsoft YaHei > SimHei > 其他）
        cn_fonts = []
        for f in font_manager.fontManager.ttflist:
            name = f.name.lower()
            if 'yahei' in name or 'microsoft yahei' in name:
                cn_fonts.append((0, f.name))  # 最高优先级
            elif 'simhei' in name:
                cn_fonts.append((1, f.name))
            elif 'heiti' in name:
                cn_fonts.append((2, f.name))
            elif 'song' in name and 'nsimsun' not in name:
                cn_fonts.append((3, f.name))
        
        cn_fonts.sort(key=lambda x: x[0])
        if cn_fonts:
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [cn_fonts[0][1], 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 默认样式
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = _COLOR_THEME['neutral']
        plt.rcParams['axes.labelcolor'] = _COLOR_THEME['neutral']
        plt.rcParams['text.color'] = '#333333'
        plt.rcParams['xtick.color'] = '#555555'
        plt.rcParams['ytick.color'] = '#555555'
        plt.rcParams['figure.dpi'] = 120
        
        _MPL_AVAILABLE = True
        
        # 初始化 seaborn
        try:
            import seaborn as sns
            sns.set_palette(_COLOR_THEME['palette'])
            _SNS_AVAILABLE = True
        except ImportError:
            pass
        
        return True
    except ImportError:
        log_warning("[Visualization] matplotlib 未安装，可视化功能不可用")
        return False


def _save_or_show(fig: Any, save_path: Optional[str] = None,
                  default_name: str = 'figure.png') -> Optional[str]:
    """保存或显示图表"""
    if save_path:
        wm = get_workspace_manager()
        if not wm.check_permission("写入"):
            return None
        safe = wm.safe_path(save_path, subdir='reports')
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        fig.savefig(safe, bbox_inches='tight', dpi=150)
        log_info(f"[Visualization] 图表已保存: {safe}")
        return safe
    return None


def _close_fig(fig: Any) -> None:
    """关闭图表释放内存"""
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        # 关闭图表是 best-effort 操作；matplotlib 未安装或图已被关闭时静默跳过
        pass


# =============================================================================
# 数据探索可视化
# =============================================================================

class DataVisualizer:
    """
    数据探索可视化器
    
    提供 EDA 常用图表：
    - 数值分布直方图 + KDE
    - 类别计数图
    - 相关性热力图
    - 缺失值可视化
    - 箱线图（异常值检测）
    - 目标变量分布
    - 散点图矩阵
    """
    
    def __init__(self, color_theme: Optional[Dict] = None) -> None:
        self.colors = color_theme or _COLOR_THEME
        _init_matplotlib()
    
    def plot_distribution(self, df: pd.DataFrame, column: str,
                          hue: Optional[str] = None,
                          bins: int = 30,
                          figsize: Tuple[int, int] = (10, 5),
                          save_path: Optional[str] = None) -> Optional[Any]:
        """
        数值列分布直方图 + KDE
        
        Args:
            df: 数据框
            column: 列名
            hue: 分组列（可选）
            bins: 直方图箱数
            figsize: 图尺寸
            save_path: 保存路径（None=不保存）
        """
        if not _MPL_AVAILABLE or column not in df.columns:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左：直方图 + KDE
        ax1 = axes[0]
        if hue and hue in df.columns:
            for i, (val, subdf) in enumerate(df.groupby(hue)):
                color = self.colors['palette'][i % len(self.colors['palette'])]
                sns.histplot(subdf[column].dropna(), bins=bins, kde=True,
                            color=color, alpha=0.5, label=str(val), ax=ax1)
            ax1.legend(title=hue)
        else:
            sns.histplot(df[column].dropna(), bins=bins, kde=True,
                        color=self.colors['primary'], ax=ax1)
        ax1.set_title(f'{column} 分布')
        ax1.set_xlabel(column)
        ax1.set_ylabel('频数')
        
        # 右：箱线图
        ax2 = axes[1]
        if hue and hue in df.columns:
            sns.boxplot(data=df, x=hue, y=column, ax=ax2, palette=self.colors['palette'])
            ax2.set_title(f'{column} 按 {hue} 分组箱线图')
        else:
            sns.boxplot(y=df[column].dropna(), color=self.colors['primary'], ax=ax2)
            ax2.set_title(f'{column} 箱线图')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                  method: str = 'pearson',
                                  figsize: Optional[Tuple[int, int]] = None,
                                  save_path: Optional[str] = None) -> Optional[Any]:
        """
        相关性热力图（自适应：列数多时自动放大、关闭标注避免重叠）
        
        Args:
            df: 数据框
            method: 'pearson' / 'spearman' / 'kendall'
            figsize: 图尺寸（None 则根据列数自动计算）
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 只选数值列
        numeric_df = df.select_dtypes(include=[np.number])
        n_cols = numeric_df.shape[1]
        if n_cols < 2:
            log_warning("[Visualization] 数值列不足，无法绘制相关性热力图")
            return None
        
        corr = numeric_df.corr(method=method)
        
        # 自适应策略
        if n_cols > 40:
            # 列太多：只保留相关性绝对值较高的特征，避免图变成马赛克
            threshold = 0.3
            strong_corr = corr.abs() > threshold
            keep_cols = strong_corr.any().where(lambda x: x).dropna().index.tolist()
            if len(keep_cols) < 2:
                keep_cols = corr.abs().mean().nlargest(min(30, n_cols)).index.tolist()
            corr = corr.loc[keep_cols, keep_cols]
            n_cols = len(keep_cols)
        
        # 根据列数动态计算 figsize
        if figsize is None:
            w = max(10, n_cols * 0.55)
            h = max(8,  n_cols * 0.45)
            figsize = (min(w, 40), min(h, 32))
        
        # 列数决定是否标注数字
        annot = n_cols <= 25
        annot_kws = {"size": max(6, 10 - n_cols // 8)} if annot else {}
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 只显示下三角
        sns.heatmap(corr, mask=mask, annot=annot, fmt='.2f', cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'shrink': 0.8},
                   annot_kws=annot_kws)
        
        # 列多时缩小标签字体并旋转
        label_size = max(6, 10 - n_cols // 10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=label_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=label_size)
        ax.set_title(f'特征相关性矩阵 ({method}, {n_cols} 列)', fontsize=14, pad=20)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_missing_values(self, df: pd.DataFrame,
                            figsize: Tuple[int, int] = (12, 6),
                            save_path: Optional[str] = None) -> Optional[Any]:
        """
        缺失值可视化
        
        左：各列缺失率条形图
        右：缺失模式热力图（样本级别）
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        
        missing = df.isnull().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        
        if missing.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, '无缺失值', ha='center', va='center',
                   fontsize=16, color=self.colors['success'])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            _save_or_show(fig, save_path)
            return fig
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})
        
        # 左：缺失率条形图
        ax1 = axes[0]
        colors_bar = [self.colors['danger'] if v > 0.3 else self.colors['warning'] if v > 0.1 else self.colors['success']
                      for v in missing.values]
        bars = ax1.barh(range(len(missing)), missing.values * 100, color=colors_bar)
        ax1.set_yticks(range(len(missing)))
        ax1.set_yticklabels(missing.index, fontsize=9)
        ax1.set_xlabel('缺失率 (%)')
        ax1.set_title('各列缺失率')
        ax1.invert_yaxis()
        
        for bar, val in zip(bars, missing.values):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=8)
        
        # 右：缺失模式热力图（抽样显示）
        ax2 = axes[1]
        cols_with_missing = missing.index.tolist()
        sample_df = df[cols_with_missing].sample(min(500, len(df)), random_state=42)
        
        import seaborn as sns
        sns.heatmap(sample_df.isnull(), cbar=False, yticklabels=False,
                   cmap=['#28A745', '#C73E1D'], ax=ax2)
        ax2.set_title(f'缺失模式（抽样 {len(sample_df)} 行）')
        ax2.set_xlabel('')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_target_distribution(self, y: Union[pd.Series, np.ndarray],
                                  task_type: str = 'classification',
                                  figsize: Tuple[int, int] = (10, 4),
                                  save_path: Optional[str] = None) -> Optional[Any]:
        """
        目标变量分布图
        
        分类 → 条形图 + 饼图
        回归 → 直方图 + 箱线图
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if task_type == 'classification' or y.dtype == object or y.nunique() <= 10:
            # 分类目标
            counts = y.value_counts().sort_index()
            colors = self.colors['palette'][:len(counts)]
            
            # 条形图
            ax1 = axes[0]
            bars = ax1.bar(counts.index.astype(str), counts.values, color=colors)
            ax1.set_title('目标类别分布')
            ax1.set_xlabel('类别')
            ax1.set_ylabel('样本数')
            for bar in bars:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{int(bar.get_height())}', ha='center', va='bottom')
            
            # 饼图
            ax2 = axes[1]
            ax2.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax2.set_title('目标类别占比')
        else:
            # 回归目标
            ax1 = axes[0]
            sns.histplot(y.dropna(), kde=True, color=self.colors['primary'], ax=ax1)
            ax1.set_title('目标值分布')
            ax1.set_xlabel('目标值')
            
            ax2 = axes[1]
            sns.boxplot(y=y.dropna(), color=self.colors['primary'], ax=ax2)
            ax2.set_title('目标值箱线图')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_pairplot(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      hue: Optional[str] = None,
                      sample_n: int = 500,
                      save_path: Optional[str] = None) -> Optional[Any]:
        """
        散点图矩阵（Pair Plot）
        
        Args:
            df: 数据框
            columns: 指定列（None=所有数值列，最多6个）
            hue: 颜色分组列
            sample_n: 抽样数量（大数据集加速）
            save_path: 保存路径
        """
        if not _SNS_AVAILABLE:
            log_warning("[Visualization] seaborn 未安装，跳过 pairplot")
            return None
        
        import seaborn as sns
        
        # 选择数值列
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        
        columns = [c for c in columns if c in df.columns]
        if len(columns) < 2:
            log_warning("[Visualization] 数值列不足，无法绘制 pairplot")
            return None
        
        # 抽样
        plot_df = df[columns + ([hue] if hue and hue in df.columns else [])]
        if len(plot_df) > sample_n:
            plot_df = plot_df.sample(sample_n, random_state=42)
        
        g = sns.pairplot(plot_df, hue=hue, palette=self.colors['palette'],
                        diag_kind='kde', corner=True,
                        plot_kws={'alpha': 0.6, 's': 20},
                        diag_kws={'fill': True})
        g.fig.suptitle('特征散点图矩阵', y=1.02, fontsize=14)
        
        safe = _save_or_show(g.fig, save_path)
        return g.fig
    
    def plot_categorical_counts(self, df: pd.DataFrame, column: str,
                                 top_n: int = 15,
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None) -> Optional[Any]:
        """
        类别变量计数图（水平条形图）
        """
        if not _MPL_AVAILABLE or column not in df.columns:
            return None
        
        import matplotlib.pyplot as plt
        
        counts = df[column].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        # 向量化生成颜色，避免 Python 循环
        palette = self.colors['palette']
        n_colors = len(palette)
        colors = [palette[i % n_colors] for i in range(len(counts))]
        bars = ax.barh(range(len(counts)), counts.values, color=colors)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index.astype(str), fontsize=9)
        ax.set_xlabel('样本数')
        ax.set_title(f'{column} 类别分布（Top {len(counts)}）')
        ax.invert_yaxis()
        
        for bar in bars:
            ax.text(bar.get_width() + max(counts.values) * 0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{int(bar.get_width())}', va='center', fontsize=8)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig


# =============================================================================
# 模型结果可视化
# =============================================================================

class ModelVisualizer:
    """
    模型结果可视化器
    
    提供模型评估常用图表：
    - 特征重要性条形图
    - 学习曲线（训练/验证）
    - ROC曲线（分类）
    - PR曲线（分类）
    - 混淆矩阵热力图
    - 残差图（回归）
    - 预测 vs 真实值散点图（回归）
    - 模型排行榜对比图
    """
    
    def __init__(self, color_theme: Optional[Dict] = None) -> None:
        self.colors = color_theme or _COLOR_THEME
        _init_matplotlib()
    
    def plot_feature_importance(self, result: Any,
                                 top_n: int = 20,
                                 figsize: Tuple[int, int] = (10, 8),
                                 save_path: Optional[str] = None) -> Optional[Any]:
        """
        特征重要性水平条形图
        
        Args:
            result: ModelingResult 或 feature_importance DataFrame
            top_n: 显示前N个特征
            figsize: 图尺寸
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        
        # 提取特征重要性
        if hasattr(result, 'feature_importance') and result.feature_importance is not None:
            fi = result.feature_importance
        elif isinstance(result, pd.DataFrame):
            fi = result
        else:
            log_warning("[Visualization] 无特征重要性数据")
            return None
        
        if fi.empty:
            return None
        
        # 标准化列名
        if 'feature' in fi.columns and 'importance' in fi.columns:
            fi = fi.sort_values('importance', ascending=True).tail(top_n)
            features = fi['feature'].astype(str).tolist()
            importances = fi['importance'].values
        elif fi.shape[1] >= 2:
            cols = fi.columns.tolist()
            fi = fi.sort_values(cols[1], ascending=True).tail(top_n)
            features = fi.iloc[:, 0].astype(str).tolist()
            importances = fi.iloc[:, 1].values
        else:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = [self.colors['primary'] if v > np.median(importances) else self.colors['secondary']
                  for v in importances]
        ax.barh(range(len(features)), importances, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('重要性')
        ax.set_title(f'特征重要性 Top {len(features)}')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_leaderboard(self, leaderboard: pd.DataFrame,
                          metric_col: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None) -> Optional[Any]:
        """
        模型排行榜对比图（水平条形图）
        
        Args:
            leaderboard: ModelingResult.leaderboard DataFrame
            metric_col: 排序指标列名
            figsize: 图尺寸
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE or leaderboard is None or leaderboard.empty:
            return None
        
        import matplotlib.pyplot as plt
        
        df = leaderboard.copy()
        
        # 自动检测指标列（支持 _mean 后缀格式）
        if metric_col is None:
            candidates = ['f1_weighted_mean', 'auc_mean', 'r2_mean', 'accuracy_mean',
                          'mean_f1_weighted', 'mean_auc', 'mean_r2', 'mean_accuracy',
                          'f1_mean', 'rmse_mean']
            for col in candidates:
                if col in df.columns:
                    metric_col = col
                    break
            # 回退：查找任何包含 _mean 的数值列
            if metric_col is None:
                for col in df.columns:
                    if '_mean' in col and pd.api.types.is_numeric_dtype(df[col]):
                        metric_col = col
                        break
            if metric_col is None and len(df.columns) > 1:
                # 排除已知非指标列，找第一个数值列
                skip_cols = {'rank', 'model', 'key', 'train_time'}
                for col in df.columns:
                    if col not in skip_cols and pd.api.types.is_numeric_dtype(df[col]):
                        metric_col = col
                        break
        
        if metric_col not in df.columns:
            return None
        
        # 排序
        ascending = 'rmse' in metric_col.lower() or 'mae' in metric_col.lower() or 'loss' in metric_col.lower()
        df = df.sort_values(metric_col, ascending=ascending)
        
        # 提取模型名
        name_col = 'model' if 'model' in df.columns else df.columns[0]
        names = df[name_col].astype(str).tolist()
        values = df[metric_col].values
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = [self.colors['success'] if i == len(names)-1 else self.colors['primary']
                  for i in range(len(names))]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(metric_col)
        ax.set_title('模型排行榜')
        ax.invert_yaxis()
        
        # 标注数值
        for i, v in enumerate(values):
            ax.text(v + (max(values) - min(values)) * 0.01, i, f'{v:.4f}',
                   va='center', fontsize=8)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_confusion_matrix(self, y_true: Union[pd.Series, np.ndarray],
                               y_pred: Union[pd.Series, np.ndarray],
                               labels: Optional[List[str]] = None,
                               normalize: bool = False,
                               figsize: Tuple[int, int] = (8, 6),
                               save_path: Optional[str] = None) -> Optional[Any]:
        """
        混淆矩阵热力图
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签
            normalize: 是否归一化
            figsize: 图尺寸
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=labels or range(cm.shape[1]),
                   yticklabels=labels or range(cm.shape[0]),
                   linewidths=0.5)
        ax.set_title('混淆矩阵' + ('（归一化）' if normalize else ''))
        ax.set_xlabel('预测值')
        ax.set_ylabel('真实值')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_residuals(self, y_true: Union[pd.Series, np.ndarray],
                        y_pred: Union[pd.Series, np.ndarray],
                        figsize: Tuple[int, int] = (12, 4),
                        save_path: Optional[str] = None) -> Optional[Any]:
        """
        回归残差分析图
        
        左：残差分布直方图
        右：残差 vs 预测值散点图
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 残差分布
        ax1 = axes[0]
        sns.histplot(residuals, kde=True, color=self.colors['primary'], ax=ax1)
        ax1.axvline(0, color=self.colors['danger'], linestyle='--', linewidth=2)
        ax1.set_title('残差分布')
        ax1.set_xlabel('残差')
        
        # 残差 vs 预测
        ax2 = axes[1]
        ax2.scatter(y_pred, residuals, alpha=0.5, color=self.colors['primary'], s=20)
        ax2.axhline(0, color=self.colors['danger'], linestyle='--', linewidth=2)
        ax2.set_title('残差 vs 预测值')
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('残差')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_prediction_scatter(self, y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray],
                                 figsize: Tuple[int, int] = (7, 7),
                                 save_path: Optional[str] = None) -> Optional[Any]:
        """
        预测值 vs 真实值散点图（回归）
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        
        y_true_arr = np.array(y_true).flatten()
        y_pred_arr = np.array(y_pred).flatten()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true_arr, y_pred_arr, alpha=0.5, color=self.colors['primary'], s=20)
        
        # 理想线 y=x
        min_val = min(y_true_arr.min(), y_pred_arr.min())
        max_val = max(y_true_arr.max(), y_pred_arr.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               color=self.colors['danger'], linestyle='--', linewidth=2, label='理想线 (y=x)')
        
        # R²
        r2 = r2_score(y_true_arr, y_pred_arr)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title('预测值 vs 真实值')
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_roc_curves(self, cv_results: List,
                         y_true: Optional[np.ndarray] = None,
                         figsize: Tuple[int, int] = (8, 7),
                         save_path: Optional[str] = None) -> Optional[Any]:
        """
        多模型 ROC 曲线对比
        
        Args:
            cv_results: CVResult 列表
            y_true: 真实标签（用于从 oof_proba 计算）
            figsize: 图尺寸
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        
        plotted = False
        for i, cv in enumerate(cv_results):
            if cv.oof_proba is None or y_true is None:
                continue
            
            # 二分类：取正类概率
            proba = cv.oof_proba
            if proba.ndim > 1 and proba.shape[1] == 2:
                proba = proba[:, 1]
            elif proba.ndim > 1:
                continue  # 多分类暂不支持
            
            try:
                fpr, tpr, _ = roc_curve(y_true, proba)
                roc_auc = auc(fpr, tpr)
                color = self.colors['palette'][i % len(self.colors['palette'])]
                ax.plot(fpr, tpr, color=color, linewidth=2,
                       label=f"{cv.model_name} (AUC={roc_auc:.3f})")
                plotted = True
            except Exception:
                continue
        
        if not plotted:
            _close_fig(fig)
            log_warning("[Visualization] 无可用的 ROC 数据")
            return None
        
        ax.plot([0, 1], [0, 1], color=self.colors['neutral'], linestyle='--', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假正率 (FPR)')
        ax.set_ylabel('真正率 (TPR)')
        ax.set_title('ROC 曲线对比')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_cv_boxplot(self, cv_results: List,
                         metric: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> Optional[Any]:
        """
        各模型 CV 分数箱线图对比
        
        Args:
            cv_results: CVResult 列表
            metric: 指标名（None=自动选择）
            figsize: 图尺寸
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE or not cv_results:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 自动选择指标
        if metric is None:
            first = cv_results[0]
            if first.fold_scores:
                metric = list(first.fold_scores.keys())[0]
        
        data = []
        labels = []
        for cv in cv_results:
            scores = cv.fold_scores.get(metric, [])
            if scores:
                data.append(scores)
                labels.append(cv.model_name)
        
        if not data:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], self.colors['palette']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'各模型 {metric} CV 分数分布')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig


# =============================================================================
# 评估决策可视化
# =============================================================================

class EvaluationVisualizer:
    """
    自动评估决策可视化器
    
    将 evaluation_engine 的决策报告转化为直观图表：
    - 雷达图：多维度模型对比
    - 综合得分条形图
    - 模式对比图（不同决策模式推荐差异）
    - 风险仪表盘
    """
    
    def __init__(self, color_theme: Optional[Dict] = None) -> None:
        self.colors = color_theme or _COLOR_THEME
        _init_matplotlib()
    
    def plot_radar_comparison(self, decision_report: Any,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> Optional[Any]:
        """
        模型多维度雷达图对比
        
        Args:
            decision_report: DecisionReport 对象
            figsize: 图尺寸
            save_path: 保存路径
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt

        scores = decision_report.scores
        if not scores:
            return None
        
        # 取 Top 5 模型
        top_scores = scores[:5]
        
        # 维度
        dimensions = ['精度', '速度', '稳定性', '简单度', '泛化']
        dim_keys = ['accuracy_score', 'speed_score', 'stability_score', 'simplicity_score', 'generalization_score']
        N = len(dimensions)
        
        # 角度
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        for i, ms in enumerate(top_scores):
            values = [getattr(ms, k, 0) for k in dim_keys]
            values += values[:1]  # 闭合
            color = self.colors['palette'][i % len(self.colors['palette'])]
            
            linewidth = 3 if ms.model_key == decision_report.recommended_model else 1.5
            alpha = 1.0 if ms.model_key == decision_report.recommended_model else 0.6
            linestyle = '-' if ms.model_key == decision_report.recommended_model else '--'
            
            ax.plot(angles, values, color=color, linewidth=linewidth,
                   linestyle=linestyle, label=ms.model_name, alpha=alpha)
            ax.fill(angles, values, color=color, alpha=alpha * 0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_title(f'模型多维度评估 ({decision_report.mode_description.split("：")[0]})',
                    fontsize=14, pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_score_breakdown(self, decision_report: Any,
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> Optional[Any]:
        """
        模型综合得分条形图 + 堆叠条形图（展示各维度贡献）
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        
        scores = decision_report.scores[:8]  # Top 8
        if not scores:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左：综合得分
        ax1 = axes[0]
        names = [s.model_name for s in scores]
        composites = [s.composite_score for s in scores]
        colors = [self.colors['success'] if s.model_key == decision_report.recommended_model else self.colors['primary']
                  for s in scores]
        
        bars = ax1.barh(range(len(names)), composites, color=colors)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('综合得分')
        ax1.set_title('模型综合得分')
        ax1.invert_yaxis()
        
        for bar, val in zip(bars, composites):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=8)
        
        # 右：堆叠维度得分
        ax2 = axes[1]
        dims = ['精度', '速度', '稳定性', '简单度', '泛化']
        dim_keys = ['accuracy_score', 'speed_score', 'stability_score', 'simplicity_score', 'generalization_score']
        
        y_pos = np.arange(len(names))
        width = 0.15
        
        for i, (dim, key) in enumerate(zip(dims, dim_keys)):
            values = [getattr(s, key, 0) for s in scores]
            offset = (i - 2) * width
            ax2.barh(y_pos + offset, values, width, label=dim,
                    color=self.colors['palette'][i])
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel('维度得分')
        ax2.set_title('各维度得分分解')
        ax2.legend(fontsize=8, loc='lower right')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 105)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_mode_comparison(self, cv_results: List, task_type: str,
                              figsize: Tuple[int, int] = (14, 8),
                              save_path: Optional[str] = None) -> Optional[Any]:
        """
        不同决策模式下的推荐结果对比
        
        展示同一批模型在不同模式下的排名变化
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        from core.evaluation_engine import ModelEvaluator, AutoDecisionEngine, DecisionMode
        
        # 评估所有模型
        evaluator = ModelEvaluator()
        scores = evaluator.evaluate_all(cv_results, task_type)
        
        modes = [
            DecisionMode.ACCURACY_FIRST,
            DecisionMode.SPEED_FIRST,
            DecisionMode.STABILITY_FIRST,
            DecisionMode.SIMPLICITY_FIRST,
            DecisionMode.BALANCED,
        ]
        
        mode_labels = ['精度优先', '速度优先', '稳定优先', '简单优先', '平衡模式']
        
        # 收集各模式下的排名
        model_keys = [s.model_key for s in scores]
        rankings = {key: [] for key in model_keys}
        default_rank = len(model_keys)
        
        for mode in modes:
            engine = AutoDecisionEngine(mode=mode)
            report = engine.decide(scores)
            # 构建排名映射：O(n) 建 dict 替代 O(n²) 的 list.index() 查找
            ranked_keys = [s.model_key for s in report.scores]
            rank_map = {key: i for i, key in enumerate(ranked_keys)}
            for key in model_keys:
                rankings[key].append(rank_map.get(key, default_rank) + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (key, ranks) in enumerate(rankings.items()):
            ms = next(s for s in scores if s.model_key == key)
            color = self.colors['palette'][i % len(self.colors['palette'])]
            ax.plot(mode_labels, ranks, marker='o', linewidth=2,
                   label=ms.model_name, color=color, markersize=8)
        
        ax.set_ylabel('排名')
        ax.set_title('不同决策模式下的模型排名变化')
        ax.invert_yaxis()  # 排名1在最上面
        ax.set_ylim(len(model_keys) + 0.5, 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig
    
    def plot_risk_summary(self, decision_report: Any,
                           figsize: Tuple[int, int] = (10, 4),
                           save_path: Optional[str] = None) -> Optional[Any]:
        """
        推荐模型的风险概览（简化文本图）
        """
        if not _MPL_AVAILABLE:
            return None
        
        import matplotlib.pyplot as plt
        
        scores = decision_report.scores
        if not scores:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # 构建文本内容
        lines = []
        lines.append(f"推荐模型: {decision_report.recommended_name}")
        lines.append(f"置信度: {decision_report.confidence:.0%}")
        lines.append(f"模式: {decision_report.mode_description}")
        lines.append("")
        lines.append("各模型风险:" + "-" * 50)
        
        for ms in scores[:5]:
            risks = []
            if ms.overfit_risk.value != '低风险':
                risks.append(f"过拟合:{ms.overfit_risk.value}")
            if ms.underfit_risk.value != '低风险':
                risks.append(f"欠拟合:{ms.underfit_risk.value}")
            risk_str = ' | '.join(risks) if risks else '无风险'
            marker = "★ " if ms.model_key == decision_report.recommended_model else "  "
            lines.append(f"{marker}{ms.model_name:20s} {risk_str}")
        
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=self.colors['background'], alpha=0.8))
        ax.set_title('风险评估摘要', fontsize=14, pad=10)
        
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig


# =============================================================================
# 便捷函数
# =============================================================================

def plot_modeling_summary(result: Any,
                          X_train: Optional[pd.DataFrame] = None,
                          y_train: Optional[Union[pd.Series, np.ndarray]] = None,
                          save_dir: Optional[str] = None,
                          task_type: Optional[str] = None) -> Dict[str, str]:
    """
    一键生成建模全流程可视化摘要
    
    生成以下图表：
    1. 模型排行榜
    2. 特征重要性（如有）
    3. 决策雷达图（如有决策报告）
    4. CV分数箱线图
    
    Args:
        result: ModelingResult
        X_train: 训练数据（用于特征重要性，可选）
        y_train: 训练标签（用于混淆矩阵/残差图，可选）
        save_dir: 保存目录（None=使用默认 reports/）
        task_type: 任务类型（'classification' / 'regression'）
        
    Returns:
        Dict[str, str]: 图表名 → 保存路径
    """
    if not _MPL_AVAILABLE:
        return {}
    
    wm = get_workspace_manager()
    if save_dir is None:
        save_dir = wm.safe_path('modeling_summary', subdir='reports')
    else:
        save_dir = wm.safe_path(save_dir, subdir='reports')
    
    if not wm.check_permission("写入"):
        return {}
    
    os.makedirs(save_dir, exist_ok=True)
    saved = {}
    
    mv = ModelVisualizer()
    ev = EvaluationVisualizer()
    
    # 1. 排行榜
    if result.leaderboard is not None and not result.leaderboard.empty:
        path = os.path.join(save_dir, '01_leaderboard.png')
        mv.plot_leaderboard(result.leaderboard, save_path=path)
        saved['leaderboard'] = path
    
    # 2. 特征重要性
    if result.feature_importance is not None and not result.feature_importance.empty:
        path = os.path.join(save_dir, '02_feature_importance.png')
        mv.plot_feature_importance(result, save_path=path)
        saved['feature_importance'] = path
    
    # 3. 决策雷达图
    if result.decision_report is not None:
        path = os.path.join(save_dir, '03_decision_radar.png')
        ev.plot_radar_comparison(result.decision_report, save_path=path)
        saved['decision_radar'] = path
        
        path = os.path.join(save_dir, '04_score_breakdown.png')
        ev.plot_score_breakdown(result.decision_report, save_path=path)
        saved['score_breakdown'] = path
    
    # 4. CV箱线图
    if result.cv_results:
        path = os.path.join(save_dir, '05_cv_boxplot.png')
        mv.plot_cv_boxplot(result.cv_results, save_path=path)
        saved['cv_boxplot'] = path
    
    # 5. 回归/分类专用图
    if task_type == 'regression' and result.cv_results and y_train is not None:
        best = result.best_cv_result
        if best and best.oof_pred is not None:
            path = os.path.join(save_dir, '06_residuals.png')
            mv.plot_residuals(y_train, best.oof_pred, save_path=path)
            saved['residuals'] = path
            
            path = os.path.join(save_dir, '07_prediction_scatter.png')
            mv.plot_prediction_scatter(y_train, best.oof_pred, save_path=path)
            saved['prediction_scatter'] = path
    
    elif task_type == 'classification' and result.cv_results and y_train is not None:
        best = result.best_cv_result
        if best and best.oof_pred is not None:
            path = os.path.join(save_dir, '06_confusion_matrix.png')
            mv.plot_confusion_matrix(y_train, best.oof_pred, save_path=path)
            saved['confusion_matrix'] = path
            
            path = os.path.join(save_dir, '07_roc_curve.png')
            mv.plot_roc_curves(result.cv_results, y_true=y_train, save_path=path)
            saved['roc_curve'] = path
    
    log_info(f"[Visualization] 已生成 {len(saved)} 张图表: {save_dir}")
    return saved


def plot_data_profile(df: pd.DataFrame,
                      target: Optional[Union[pd.Series, np.ndarray, str]] = None,
                      task_type: Optional[str] = None,
                      save_dir: Optional[str] = None) -> Dict[str, str]:
    """
    一键生成数据探索可视化摘要
    
    Args:
        df: 数据框
        target: 目标列（Series 或 列名字符串）
        task_type: 任务类型
        save_dir: 保存目录
        
    Returns:
        Dict[str, str]: 图表名 → 保存路径
    """
    if not _MPL_AVAILABLE:
        return {}
    
    wm = get_workspace_manager()
    if save_dir is None:
        save_dir = wm.safe_path('data_profile', subdir='reports')
    else:
        save_dir = wm.safe_path(save_dir, subdir='reports')
    
    if not wm.check_permission("写入"):
        return {}
    
    os.makedirs(save_dir, exist_ok=True)
    saved = {}
    
    dv = DataVisualizer()
    
    # 1. 相关性热力图
    path = os.path.join(save_dir, '01_correlation.png')
    dv.plot_correlation_heatmap(df, save_path=path)
    saved['correlation'] = path
    
    # 2. 缺失值
    path = os.path.join(save_dir, '02_missing_values.png')
    dv.plot_missing_values(df, save_path=path)
    saved['missing_values'] = path
    
    # 3. 目标变量分布
    if target is not None:
        if isinstance(target, str) and target in df.columns:
            y = df[target]
        else:
            y = target
        
        if task_type is None:
            from core.modeling_engine import TaskTypeDetector
            detected = TaskTypeDetector.detect(y)
            task_type = detected.value
        
        path = os.path.join(save_dir, '03_target_distribution.png')
        dv.plot_target_distribution(y, task_type=task_type, save_path=path)
        saved['target_distribution'] = path
    
    # 4. 数值分布（前5个数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for i, col in enumerate(numeric_cols[:5]):
        path = os.path.join(save_dir, f'04_distribution_{col}.png')
        dv.plot_distribution(df, col, save_path=path)
        saved[f'distribution_{col}'] = path
    
    # 5. 类别分布（前5个类别列）
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for i, col in enumerate(cat_cols[:5]):
        path = os.path.join(save_dir, f'05_categorical_{col}.png')
        dv.plot_categorical_counts(df, col, save_path=path)
        saved[f'categorical_{col}'] = path
    
    log_info(f"[Visualization] 已生成 {len(saved)} 张数据探索图表: {save_dir}")
    return saved


# =============================================================================
# 超参优化与 AutoML 可视化
# =============================================================================

def plot_optimization_history(history: Dict[str, List[Dict]],
                               save_path: Optional[str] = None) -> Optional[str]:
    """
    绘制参数优化历史曲线
    
    Args:
        history: {model_key: [{'trial': int, 'score': float, ...}, ...]}
        save_path: 保存路径
        
    Returns:
        保存路径
    """
    if not _MPL_AVAILABLE:
        return None
    
    import matplotlib.pyplot as plt
    
    n_models = len(history)
    if n_models == 0:
        return None
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    
    for idx, (model_key, trials) in enumerate(history.items()):
        ax = axes[idx // cols, idx % cols]
        
        if not trials:
            ax.set_title(f'{model_key} (无数据)')
            continue
        
        trial_nums = [t.get('trial', i + 1) for i, t in enumerate(trials)]
        scores = [t.get('score', t.get('value', 0)) for t in trials]
        
        ax.plot(trial_nums, scores, 'b-', alpha=0.5, label='Trial score')
        
        # 累积最优
        best_so_far = []
        current_best = float('-inf')
        for s in scores:
            if s > current_best:
                current_best = s
            best_so_far.append(current_best)
        ax.plot(trial_nums, best_so_far, 'r-', linewidth=2, label='Best so far')
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Score')
        ax.set_title(f'Optimization: {model_key}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_models, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        wm = get_workspace_manager()
        safe = wm.safe_path(save_path, 'reports')
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        fig.savefig(safe, dpi=150, bbox_inches='tight')
        log_info(f"[Visualization] 优化历史图已保存: {safe}")
        plt.close(fig)
        return safe
    
    plt.close(fig)
    return None


def plot_reward_curve(rl_history: List[Dict],
                      save_path: Optional[str] = None) -> Optional[str]:
    """
    绘制 RL 优化器的 reward / epsilon 变化曲线
    
    Args:
        rl_history: [{'trial': int, 'reward': float, 'epsilon': float, ...}, ...]
        save_path: 保存路径
        
    Returns:
        保存路径
    """
    if not _MPL_AVAILABLE:
        return None
    
    import matplotlib.pyplot as plt
    
    if not rl_history:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    trials = [t.get('trial', i + 1) for i, t in enumerate(rl_history)]
    rewards = [t.get('reward', 0) for t in rl_history]
    epsilons = [t.get('epsilon', 0) for t in rl_history]
    
    # Reward 曲线
    ax1.plot(trials, rewards, 'g-', alpha=0.6)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Reward')
    ax1.set_title('RL Reward Curve')
    ax1.grid(True, alpha=0.3)
    
    # Epsilon 衰减
    ax2.plot(trials, epsilons, 'purple', linewidth=2)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Epsilon Decay')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        wm = get_workspace_manager()
        safe = wm.safe_path(save_path, 'reports')
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        fig.savefig(safe, dpi=150, bbox_inches='tight')
        log_info(f"[Visualization] RL 曲线已保存: {safe}")
        plt.close(fig)
        return safe
    
    plt.close(fig)
    return None


def plot_autoencoder_results(X_orig: np.ndarray, X_reconstructed: np.ndarray,
                             encoded: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> Optional[str]:
    """
    绘制 AutoEncoder 结果：原始 vs 重构对比、编码特征分布
    
    Args:
        X_orig: 原始特征 (n_samples, n_features)
        X_reconstructed: 重构特征
        encoded: 编码特征 (n_samples, encoding_dim)
        save_path: 保存路径
        
    Returns:
        保存路径
    """
    if not _MPL_AVAILABLE:
        return None
    
    import matplotlib.pyplot as plt
    
    n_plots = 2 if encoded is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 4), squeeze=False)
    
    # 重构误差分布
    ax = axes[0, 0]
    mse = np.mean((X_orig - X_reconstructed) ** 2, axis=1)
    ax.hist(mse, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(np.mean(mse), color='red', linestyle='--', label=f'Mean MSE: {np.mean(mse):.4f}')
    ax.set_xlabel('Reconstruction MSE')
    ax.set_ylabel('Frequency')
    ax.set_title('AutoEncoder Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 编码特征分布（PCA降维到2D）
    if encoded is not None:
        ax = axes[0, 1]
        if encoded.shape[1] >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            encoded_2d = pca.fit_transform(encoded)
            scatter = ax.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=mse, cmap='viridis', alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label='Reconstruction MSE')
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_title('Encoded Features (PCA 2D)')
        else:
            ax.hist(encoded.flatten(), bins=50, color='green', edgecolor='white', alpha=0.7)
            ax.set_xlabel('Encoded Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Encoded Feature Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        wm = get_workspace_manager()
        safe = wm.safe_path(save_path, 'reports')
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        fig.savefig(safe, dpi=150, bbox_inches='tight')
        log_info(f"[Visualization] AutoEncoder 图已保存: {safe}")
        plt.close(fig)
        return safe
    
    plt.close(fig)
    return None


# 模块初始化
_init_matplotlib()
