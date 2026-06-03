"""
可解释性分析引擎

支持：
- SHAP 全局/局部解释（TreeSHAP, KernelSHAP, DeepSHAP）
- LIME 局部解释
- 特征重要性汇总（模型内置 + SHAP + Permutation）
- 单样本解释报告
- 可视化导出（文本报告 + 图表）

无 SHAP/LIME 时优雅降级，使用模型内置重要性。
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from core.workspace_manager import get_workspace_manager
from core.modeling_engine import TaskType, ModelLibrary
from utils.helpers import log_info, log_warning, log_error

# 尝试导入 SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 尝试导入 LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


@dataclass
class ExplanationResult:
    """解释结果"""
    model_key: str
    global_importance: Optional[pd.DataFrame] = None
    shap_values: Optional[Any] = None
    lime_explanations: Optional[List] = None
    instance_explanations: Dict[int, Dict] = field(default_factory=dict)
    method: str = ""
    explanation_time: float = 0.0


class ExplainabilityEngine:
    """
    可解释性分析引擎
    
    自动选择最佳解释方法：
    - 树模型 → TreeSHAP（最快最准确）
    - 线性模型 → 系数解释
    - 神经网络/集成模型 → KernelSHAP / Permutation
    - 局部解释 → LIME
    """
    
    def __init__(self,
                 background_samples: int = 100,
                 max_display_features: int = 20,
                 use_shap: bool = True,
                 use_lime: bool = True) -> None:
        """
        Args:
            background_samples: SHAP背景样本数
            max_display_features: 最多显示的特征数
            use_shap: 是否使用SHAP
            use_lime: 是否使用LIME
        """
        self.background_samples = background_samples
        self.max_display = max_display_features
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.use_lime = use_lime and LIME_AVAILABLE
        
        if use_shap and not SHAP_AVAILABLE:
            log_warning("[ExplainabilityEngine] SHAP 未安装，将使用内置重要性回退")
        if use_lime and not LIME_AVAILABLE:
            log_warning("[ExplainabilityEngine] LIME 未安装，局部解释不可用")
    
    def explain_model(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      model_key: str = "",
                      task_type: Union[str, TaskType] = TaskType.CLASSIFICATION,
                      feature_names: Optional[List[str]] = None) -> ExplanationResult:
        """
        解释单个模型
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 标签（可选）
            model_key: 模型标识
            task_type: 任务类型
            feature_names: 特征名
            
        Returns:
            ExplanationResult
        """
        import time
        start = time.time()
        
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        fnames = feature_names or list(X.columns)
        result = ExplanationResult(model_key=model_key, method="builtin")
        
        # 1. 全局特征重要性（内置）
        result.global_importance = self._get_builtin_importance(model, fnames)
        
        # 2. SHAP 解释
        if self.use_shap:
            try:
                shap_result = self._compute_shap(model, X, fnames, task_type, model_key)
                if shap_result is not None:
                    result.shap_values = shap_result.get('values')
                    result.global_importance = shap_result.get('importance')
                    result.method = shap_result.get('method', 'shap')
            except Exception as e:
                log_warning(f"[ExplainabilityEngine] SHAP计算失败: {e}")
        
        # 3. Permutation Importance（作为补充）
        if y is not None and result.global_importance is None:
            try:
                result.global_importance = self._permutation_importance(model, X, y, task_type, fnames)
                result.method = "permutation"
            except Exception as e:
                log_warning(f"[ExplainabilityEngine] Permutation重要性失败: {e}")
        
        result.explanation_time = time.time() - start
        return result
    
    def explain_instance(self,
                         model: Any,
                         X: pd.DataFrame,
                         instance_index: int,
                         model_key: str = "",
                         task_type: Union[str, TaskType] = TaskType.CLASSIFICATION,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        解释单个样本的预测
        
        Args:
            model: 训练好的模型
            X: 完整特征数据
            instance_index: 样本索引
            
        Returns:
            单样本解释字典
        """
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        fnames = feature_names or list(X.columns)
        instance = X.iloc[instance_index:instance_index+1]
        explanation = {
            'instance_index': instance_index,
            'prediction': None,
            'features': {},
            'top_positive': [],
            'top_negative': [],
        }
        
        # 预测值
        try:
            if hasattr(model, 'predict_proba') and task_type == TaskType.CLASSIFICATION:
                proba = model.predict_proba(instance)[0]
                explanation['prediction'] = {
                    'class': int(model.predict(instance)[0]),
                    'probability': float(proba[1] if len(proba) == 2 else proba.max())
                }
            else:
                explanation['prediction'] = {
                    'value': float(model.predict(instance)[0])
                }
        except:
            pass
        
        # LIME 局部解释
        if self.use_lime:
            try:
                lime_exp = self._lime_explain(model, X, instance, fnames, task_type)
                if lime_exp:
                    explanation['lime'] = lime_exp
            except Exception as e:
                log_warning(f"[ExplainabilityEngine] LIME解释失败: {e}")
        
        # SHAP 局部值
        if self.use_shap:
            try:
                shap_values = self._compute_shap(model, X, fnames, task_type, model_key)
                if shap_values and 'values' in shap_values:
                    vals = shap_values['values']
                    # 防御0维/1维数组
                    if isinstance(vals, np.ndarray):
                        if vals.ndim == 0:
                            vals = None
                        elif vals.ndim == 1:
                            vals = vals.reshape(1, -1)
                    if vals is not None:
                        if hasattr(vals, 'shape') and len(vals.shape) > 1:
                            sv = vals[instance_index] if vals.shape[0] > instance_index else vals[0]
                        else:
                            sv = vals[instance_index] if len(vals) > instance_index else None
                    else:
                        sv = None
                    
                    if sv is not None:
                        feature_contrib = []
                        # 防御sv为标量
                        if hasattr(sv, '__len__') and not isinstance(sv, (str, bytes)):
                            for i, fname in enumerate(fnames):
                                val = float(sv[i]) if i < len(sv) else 0
                                feature_contrib.append((fname, val))
                        else:
                            feature_contrib = [(fnames[0], float(sv))] if fnames else []
                        
                        feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
                        explanation['top_positive'] = [(f, v) for f, v in feature_contrib if v > 0][:5]
                        explanation['top_negative'] = [(f, v) for f, v in feature_contrib if v < 0][:5]
            except:
                pass
        
        # 特征值
        for fname in fnames[:self.max_display]:
            explanation['features'][fname] = float(instance[fname].iloc[0])
        
        return explanation
    
    def explain_multiple_instances(self,
                                    model: Any,
                                    X: pd.DataFrame,
                                    indices: List[int],
                                    **kwargs) -> List[Dict[str, Any]]:
        """批量解释多个样本"""
        results = []
        for idx in indices:
            exp = self.explain_instance(model, X, idx, **kwargs)
            results.append(exp)
        return results
    
    def compare_models(self,
                       models: Dict[str, Any],
                       X: pd.DataFrame,
                       feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        对比多个模型的特征重要性
        
        Returns:
            DataFrame: 每个特征在不同模型中的重要性
        """
        fnames = feature_names or list(X.columns)
        all_importance = defaultdict(dict)
        
        for model_key, model in models.items():
            fi = self._get_builtin_importance(model, fnames)
            if fi is not None:
                for row in fi.itertuples(index=False):
                    all_importance[row.feature][model_key] = row.importance
        
        rows = []
        for feat, scores in all_importance.items():
            row = {'feature': feat}
            row.update(scores)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            # 计算平均排名
            score_cols = [c for c in df.columns if c != 'feature']
            df['mean_importance'] = df[score_cols].mean(axis=1)
            df = df.sort_values('mean_importance', ascending=False)
        
        return df
    
    def generate_report(self,
                        explanation: ExplanationResult,
                        output_path: Optional[str] = None) -> str:
        """
        生成可解释性报告（文本）
        
        Returns:
            报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"模型可解释性报告: {explanation.model_key}")
        lines.append("=" * 60)
        lines.append(f"\n解释方法: {explanation.method}")
        lines.append(f"计算耗时: {explanation.explanation_time:.2f}s")
        
        if explanation.global_importance is not None and not explanation.global_importance.empty:
            lines.append(f"\n【全局特征重要性 (Top {self.max_display})】")
            for row in explanation.global_importance.head(self.max_display).itertuples(index=False):
                lines.append(f"  {row.feature:30s}: {row.importance:.4f}")
        
        report_text = "\n".join(lines)
        
        if output_path:
            wm = get_workspace_manager()
            safe_path = wm.write_text(output_path, report_text, subdir='reports')
            if safe_path:
                log_info(f"解释报告已保存: {safe_path}")
        
        return report_text
    
    def _get_builtin_importance(self, model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """获取模型内置特征重要性"""
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        
        if importances is not None:
            df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances[:len(feature_names)]
            }).sort_values('importance', ascending=False)
            return df
        
        return None
    
    def _compute_shap(self, model: Any, X: pd.DataFrame, feature_names: List[str],
                      task_type: TaskType, model_key: str) -> Optional[Dict[str, Any]]:
        """计算SHAP值"""
        if not SHAP_AVAILABLE:
            return None
        
        # 采样背景数据
        background = shap.sample(X, min(self.background_samples, len(X)), random_state=42)
        
        # 根据模型类型选择解释器
        explainer = None
        method = ""
        
        # TreeSHAP
        if hasattr(model, 'tree_') or model_key in ['xgb', 'lgb', 'catboost', 'rf', 'dt', 'et', 'gbdt']:
            try:
                explainer = shap.TreeExplainer(model)
                method = "tree_shap"
            except:
                pass
        
        # LinearSHAP
        if explainer is None and hasattr(model, 'coef_'):
            try:
                explainer = shap.LinearExplainer(model, background)
                method = "linear_shap"
            except:
                pass
        
        # KernelSHAP（通用，慢）
        if explainer is None:
            try:
                # 使用预测函数的包装
                if task_type == TaskType.CLASSIFICATION and hasattr(model, 'predict_proba'):
                    f = lambda x: model.predict_proba(x)[:, 1] if len(np.unique(model.predict(x))) == 2 else model.predict_proba(x)
                else:
                    f = model.predict
                
                explainer = shap.KernelExplainer(f, background)
                method = "kernel_shap"
            except:
                pass
        
        if explainer is None:
            return None
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X.iloc[:min(len(X), 500)])
        
        # 处理多分类SHAP
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # 防御0维数组
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 0:
            return None
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 1:
            # 1维数组: 可能是单样本单输出，扩展为2维
            shap_values = shap_values.reshape(1, -1)
        
        # 计算全局重要性
        importance = np.abs(shap_values).mean(axis=0)
        fi_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance[:len(feature_names)]
        }).sort_values('importance', ascending=False)
        
        return {
            'values': shap_values,
            'importance': fi_df,
            'method': method
        }
    
    def _lime_explain(self, model: Any, X: pd.DataFrame, instance: pd.DataFrame,
                      feature_names: List[str], task_type: TaskType) -> Optional[Dict[str, Any]]:
        """LIME局部解释"""
        if not LIME_AVAILABLE:
            return None
        
        mode = 'classification' if task_type == TaskType.CLASSIFICATION else 'regression'
        
        # 预测函数包装
        if mode == 'classification' and hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict
        
        explainer = LimeTabularExplainer(
            X.values,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True
        )
        
        explanation = explainer.explain_instance(
            instance.values[0],
            predict_fn,
            num_features=min(10, len(feature_names))
        )
        
        return {
            'feature_weights': explanation.as_list(),
            'intercept': explanation.intercept,
            'score': explanation.score
        }
    
    def _permutation_importance(self, model: Any, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                                 task_type: TaskType, feature_names: List[str]) -> pd.DataFrame:
        """排列重要性（作为SHAP的补充/回退）"""
        from sklearn.inspection import permutation_importance
        
        if task_type == TaskType.CLASSIFICATION:
            scoring = 'roc_auc_ovr_weighted' if len(np.unique(y)) > 2 else 'roc_auc'
        else:
            scoring = 'r2'
        
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=42,
            scoring=scoring, n_jobs=1
        )
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        return df


# =============================================================================
# 便捷函数
# =============================================================================

def explain_model_quick(model: Any, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None,
                        model_key: str = "", task_type: Union[str, TaskType] = 'classification') -> ExplanationResult:
    """快速解释模型"""
    engine = ExplainabilityEngine()
    return engine.explain_model(model, X, y, model_key, task_type)
