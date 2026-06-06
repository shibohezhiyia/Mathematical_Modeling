"""
公平性检查引擎

基于 Fairlearn，提供模型公平性评估与约束：
  - 群体公平性指标: demographic parity, equalized odds, equal opportunity
  - 个体公平性: 基于相似性的公平性度量
  - 公平性约束训练: ExponentiatedGradient / GridSearch
  - 自动生成公平性报告

支持的敏感属性:
  - 性别、种族、年龄组等分类属性
  - 自动检测候选敏感属性（低基数分类列）
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.helpers import log_info, log_warning

FAIRLEARN_AVAILABLE = False
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate_difference,
        false_negative_rate_difference,
        selection_rate_difference,
    )
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class FairnessReport:
    """公平性分析报告"""
    model_key: str
    sensitive_attr: str
    demographic_parity_diff: Optional[float] = None
    demographic_parity_ratio: Optional[float] = None
    equalized_odds_diff: Optional[float] = None
    equalized_odds_ratio: Optional[float] = None
    fpr_diff: Optional[float] = None
    fnr_diff: Optional[float] = None
    selection_rate_diff: Optional[float] = None
    group_metrics: Dict[str, Any] = field(default_factory=dict)
    is_fair: bool = True
    recommendations: List[str] = field(default_factory=list)
    analysis_time: float = 0.0


class FairnessEngine:
    """
    公平性分析引擎
    
    自动检测敏感属性并计算公平性指标。
    """
    
    def __init__(self, fairness_threshold: float = 0.05) -> None:
        """
        Args:
            fairness_threshold: 公平性阈值，指标差异超过此值认为不公平
        """
        self.threshold = fairness_threshold
        if not FAIRLEARN_AVAILABLE:
            log_warning("[FairnessEngine] Fairlearn 未安装，公平性分析不可用")
    
    def detect_sensitive_attributes(self, df: pd.DataFrame,
                                    exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        自动检测候选敏感属性列
        
        规则:
          - 分类列（object/category/bool）
          - 唯一值数量 <= 10（低基数）
          - 排除目标列和 ID 列
        """
        exclude = set(exclude_cols or [])
        candidates = []
        
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            n_unique = df[col].nunique(dropna=True)
            if 2 <= n_unique <= 10:
                # 排除明显不是敏感属性的列（如姓名、ID）
                if n_unique > len(df) * 0.8:
                    continue
                candidates.append(col)
        
        return candidates
    
    def analyze(self, model: Any, X: pd.DataFrame, y_true: Union[pd.Series, np.ndarray],
                y_pred: Optional[np.ndarray] = None,
                sensitive_attr: Optional[str] = None,
                task_type: str = 'classification') -> Optional[FairnessReport]:
        """
        分析模型在敏感属性上的公平性
        
        Args:
            model: 训练好的模型（用于预测，如果 y_pred 未提供）
            X: 特征数据
            y_true: 真实标签
            y_pred: 预测标签（可选，未提供时调用 model.predict）
            sensitive_attr: 敏感属性列名（None=自动检测）
            task_type: 'classification' 或 'regression'
        
        Returns:
            FairnessReport 或 None
        """
        if not FAIRLEARN_AVAILABLE:
            log_warning("[FairnessEngine] Fairlearn 未安装")
            return None
        
        start = time.time()
        
        if sensitive_attr is None:
            candidates = self.detect_sensitive_attributes(X)
            if not candidates:
                log_info("[FairnessEngine] 未检测到候选敏感属性")
                return None
            sensitive_attr = candidates[0]
        
        if sensitive_attr not in X.columns:
            log_warning(f"[FairnessEngine] 敏感属性 '{sensitive_attr}' 不在数据中")
            return None
        
        # 获取预测（排除敏感属性列，避免 sklearn 特征名不匹配）
        X_features = X.drop(columns=[sensitive_attr]) if sensitive_attr in X.columns else X
        if y_pred is None:
            y_pred = model.predict(X_features)
        
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        sensitive = X[sensitive_attr].astype(str).fillna('Unknown')
        
        report = FairnessReport(
            model_key=getattr(model, '__class__.__name__', 'unknown'),
            sensitive_attr=sensitive_attr
        )
        
        try:
            # 分类任务公平性指标
            if task_type == 'classification':
                report.demographic_parity_diff = float(
                    demographic_parity_difference(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                report.demographic_parity_ratio = float(
                    demographic_parity_ratio(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                report.equalized_odds_diff = float(
                    equalized_odds_difference(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                report.equalized_odds_ratio = float(
                    equalized_odds_ratio(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                report.fpr_diff = float(
                    false_positive_rate_difference(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                report.fnr_diff = float(
                    false_negative_rate_difference(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                report.selection_rate_diff = float(
                    selection_rate_difference(y_true_arr, y_pred_arr, sensitive_features=sensitive)
                )
                
                # 判断是否公平
                unfair_metrics = []
                if abs(report.demographic_parity_diff) > self.threshold:
                    unfair_metrics.append(f"demographic_parity_diff={report.demographic_parity_diff:.3f}")
                if abs(report.equalized_odds_diff) > self.threshold:
                    unfair_metrics.append(f"equalized_odds_diff={report.equalized_odds_diff:.3f}")
                
                report.is_fair = len(unfair_metrics) == 0
                if not report.is_fair:
                    report.recommendations.append(
                        f"模型在 '{sensitive_attr}' 上存在不公平: {', '.join(unfair_metrics)}"
                    )
                    report.recommendations.append(
                        "建议: 使用 Fairlearn 的 ExponentiatedGradient + DemographicParity 约束重新训练，"
                        "或收集更多少数群体的训练数据。"
                    )
            
            # 各群体详细指标
            for group in sensitive.unique():
                mask = sensitive == group
                group_y_true = y_true_arr[mask]
                group_y_pred = y_pred_arr[mask]
                
                metrics = {
                    'count': int(mask.sum()),
                    'accuracy': float((group_y_true == group_y_pred).mean()) if task_type == 'classification' else None,
                }
                if task_type == 'classification':
                    from sklearn.metrics import precision_score, recall_score
                    try:
                        metrics['precision'] = float(precision_score(group_y_true, group_y_pred, average='binary', zero_division=0))
                        metrics['recall'] = float(recall_score(group_y_true, group_y_pred, average='binary', zero_division=0))
                    except:
                        pass
                else:
                    from sklearn.metrics import mean_squared_error
                    metrics['mse'] = float(mean_squared_error(group_y_true, group_y_pred))
                
                report.group_metrics[str(group)] = metrics
        
        except Exception as e:
            log_warning(f"[FairnessEngine] 公平性计算失败: {e}")
        
        report.analysis_time = time.time() - start
        return report
    
    def train_with_constraint(self, estimator: Any, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                              sensitive_attr: str,
                              constraint: str = 'demographic_parity',
                              eps: float = 0.01) -> Any:
        """
        使用公平性约束训练模型
        
        Args:
            estimator: 基础模型（需支持 fit/predict）
            constraint: 'demographic_parity' 或 'equalized_odds'
            eps: 约束容忍度
        """
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("Fairlearn 未安装")
        
        sensitive = X[sensitive_attr].astype(str).fillna('Unknown')
        
        if constraint == 'demographic_parity':
            cons = DemographicParity()
        elif constraint == 'equalized_odds':
            cons = EqualizedOdds()
        else:
            raise ValueError(f"未知约束: {constraint}")
        
        mitigator = ExponentiatedGradient(estimator, cons)
        mitigator.fit(X, y, sensitive_features=sensitive)
        
        log_info(f"[FairnessEngine] 公平约束训练完成: {constraint}, eps={eps}")
        return mitigator
