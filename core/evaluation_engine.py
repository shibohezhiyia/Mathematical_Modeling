"""
模型评估与自动决策引擎

核心能力：
1. 多维度评估矩阵：精度、速度、稳定性、复杂度、泛化能力
2. 自动决策：基于场景（精度优先/速度优先/稳定性优先/平衡）推荐最优模型
3. 用户干预：允许覆盖自动选择、指定偏好、查看详细对比
4. 决策报告：推荐理由、置信度、风险提示

使用方式：
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate_all(cv_results, task_type)
    
    engine = AutoDecisionEngine(mode=DecisionMode.BALANCED)
    decision = engine.decide(scores)
    
    print(decision.recommendation)      # 推荐模型及理由
    print(decision.full_comparison)     # 完整对比表格
    print(decision.user_override_guide) # 用户如何覆盖
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd

from core.modeling_engine import CVResult, TaskType, TaskTypeDetector


# =============================================================================
# 枚举与常量
# =============================================================================

class DecisionMode(Enum):
    """决策模式：不同场景下的权重偏好"""
    ACCURACY_FIRST = "accuracy_first"      # 精度优先
    SPEED_FIRST = "speed_first"            # 速度优先
    STABILITY_FIRST = "stability_first"    # 稳定性优先（低方差）
    SIMPLICITY_FIRST = "simplicity_first"  # 简单优先（防过拟合）
    BALANCED = "balanced"                  # 平衡（默认）
    CUSTOM = "custom"                      # 用户自定义权重


class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    UNKNOWN = "未知"


# 各模式权重配置 (accuracy, speed, stability, simplicity, generalization)
_MODE_WEIGHTS = {
    DecisionMode.ACCURACY_FIRST:    (0.50, 0.05, 0.20, 0.05, 0.20),
    DecisionMode.SPEED_FIRST:       (0.20, 0.45, 0.10, 0.15, 0.10),
    DecisionMode.STABILITY_FIRST:   (0.25, 0.05, 0.45, 0.10, 0.15),
    DecisionMode.SIMPLICITY_FIRST:  (0.20, 0.10, 0.15, 0.45, 0.10),
    DecisionMode.BALANCED:          (0.30, 0.15, 0.25, 0.15, 0.15),
}


# =============================================================================
# 评估结果数据结构
# =============================================================================

@dataclass
class ModelScore:
    """单个模型的评估分数"""
    model_key: str
    model_name: str
    
    # 原始指标
    primary_metric: str = ""              # 主指标名称
    primary_score: float = 0.0            # 主指标值（如AUC/RMSE）
    primary_std: float = 0.0              # 主指标标准差
    train_time: float = 0.0               # 训练时间（秒）
    n_parameters: int = 0                 # 参数数量（估计）
    
    # 归一化分数 (0-100)
    accuracy_score: float = 0.0           # 精度分数
    speed_score: float = 0.0              # 速度分数
    stability_score: float = 0.0          # 稳定性分数
    simplicity_score: float = 0.0         # 简单度分数
    generalization_score: float = 0.0     # 泛化分数
    
    # 综合分数
    composite_score: float = 0.0          # 综合得分
    rank: int = 0                         # 排名
    
    # 风险
    overfit_risk: RiskLevel = RiskLevel.UNKNOWN
    underfit_risk: RiskLevel = RiskLevel.UNKNOWN


@dataclass
class DecisionReport:
    """决策报告"""
    mode: DecisionMode
    mode_description: str = ""
    
    # 推荐
    recommended_model: str = ""           # 推荐模型key
    recommended_name: str = ""            # 推荐模型名称
    recommendation_reason: str = ""       # 推荐理由
    confidence: float = 0.0               # 置信度 (0-1)
    
    # 完整对比
    scores: List[ModelScore] = field(default_factory=list)
    comparison_table: Optional[pd.DataFrame] = None
    
    # 风险提示
    risks: List[str] = field(default_factory=list)
    
    # 用户干预指南
    override_options: Dict[str, str] = field(default_factory=dict)
    
    # 场景建议
    scenario_advice: str = ""             # 当前场景下的建议


# =============================================================================
# 模型评估器
# =============================================================================

class ModelEvaluator:
    """
    模型评估器
    
    对训练好的模型进行多维度评估，生成分数矩阵。
    
    评估维度：
    1. 精度 (Accuracy): CV主指标的表现
    2. 速度 (Speed): 训练时间的倒数
    3. 稳定性 (Stability): CV标准差的倒数
    4. 简单度 (Simplicity): 模型复杂度的倒数
    5. 泛化 (Generalization): 跨折一致性
    """
    
    def __init__(self,
                 accuracy_weight: float = 1.0,
                 speed_weight: float = 1.0,
                 stability_weight: float = 1.0,
                 simplicity_weight: float = 1.0,
                 generalization_weight: float = 1.0) -> None:
        self.weights = {
            'accuracy': accuracy_weight,
            'speed': speed_weight,
            'stability': stability_weight,
            'simplicity': simplicity_weight,
            'generalization': generalization_weight,
        }
    
    def evaluate_all(self,
                     cv_results: List[CVResult],
                     task_type: TaskType,
                     primary_metric: Optional[str] = None) -> List[ModelScore]:
        """
        评估所有模型
        
        Args:
            cv_results: 各模型的CV结果
            task_type: 任务类型
            primary_metric: 主指标（None=自动选择）
            
        Returns:
            List[ModelScore]
        """
        if not cv_results:
            return []
        
        # 确定主指标
        if primary_metric is None:
            primary_metric = TaskTypeDetector.get_primary_metric(task_type)
        
        # 收集所有原始值用于归一化
        raw_scores = defaultdict(list)
        raw_times = []
        raw_stds = []
        raw_params = []
        
        for r in cv_results:
            score = r.mean_scores.get(primary_metric, 0)
            if task_type == TaskType.REGRESSION and primary_metric == 'rmse':
                # RMSE 越小越好，需要反转
                score = -score
            raw_scores['accuracy'].append(score)
            raw_times.append(r.train_time)
            raw_stds.append(r.std_scores.get(primary_metric, 0))
            raw_params.append(self._estimate_params(r))
        
        # 计算归一化基准
        score_range = self._get_range(raw_scores['accuracy'])
        time_range = self._get_range(raw_times)
        std_range = self._get_range(raw_stds)
        param_range = self._get_range(raw_params)
        
        # 评估每个模型
        model_scores = []
        for i, r in enumerate(cv_results):
            ms = self._evaluate_single(
                r, task_type, primary_metric,
                raw_scores['accuracy'][i], raw_times[i], raw_stds[i], raw_params[i],
                score_range, time_range, std_range, param_range
            )
            model_scores.append(ms)
        
        # 按综合得分排序
        model_scores.sort(key=lambda x: x.composite_score, reverse=True)
        for i, ms in enumerate(model_scores):
            ms.rank = i + 1
        
        return model_scores
    
    def _evaluate_single(self,
                         cv_result: CVResult,
                         task_type: TaskType,
                         primary_metric: str,
                         raw_score: float,
                         raw_time: float,
                         raw_std: float,
                         raw_params: int,
                         score_range: Tuple[float, float],
                         time_range: Tuple[float, float],
                         std_range: Tuple[float, float],
                         param_range: Tuple[float, float]) -> ModelScore:
        """评估单个模型"""
        
        # 1. 精度分数 (越高越好)
        accuracy_norm = self._normalize(raw_score, score_range[0], score_range[1], higher_better=True)
        
        # 2. 速度分数 (训练时间越短越好)
        speed_norm = self._normalize(raw_time, time_range[0], time_range[1], higher_better=False)
        
        # 3. 稳定性分数 (标准差越小越好)
        stability_norm = self._normalize(raw_std, std_range[0], std_range[1], higher_better=False)
        
        # 4. 简单度分数 (参数越少越好)
        simplicity_norm = self._normalize(raw_params, param_range[0], param_range[1], higher_better=False)
        
        # 5. 泛化分数 (跨折最大最小差距越小越好)
        fold_values = cv_result.fold_scores.get(primary_metric, [])
        if len(fold_values) >= 2:
            gen_gap = max(fold_values) - min(fold_values)
            gen_norm = max(0, 100 - gen_gap * 200)  # 差距0.5 → 0分
        else:
            gen_norm = 50  # 未知
        
        # 综合得分（等权重，后续决策引擎会按模式调整）
        composite = np.mean([accuracy_norm, speed_norm, stability_norm, simplicity_norm, gen_norm])
        
        # 风险判断
        overfit_risk = self._assess_overfit_risk(cv_result, task_type, primary_metric)
        underfit_risk = self._assess_underfit_risk(cv_result, task_type, primary_metric)
        
        return ModelScore(
            model_key=cv_result.model_key,
            model_name=cv_result.model_name,
            primary_metric=primary_metric,
            primary_score=raw_score if task_type != TaskType.REGRESSION else -raw_score,
            primary_std=raw_std,
            train_time=raw_time,
            n_parameters=raw_params,
            accuracy_score=accuracy_norm,
            speed_score=speed_norm,
            stability_score=stability_norm,
            simplicity_score=simplicity_norm,
            generalization_score=gen_norm,
            composite_score=composite,
            overfit_risk=overfit_risk,
            underfit_risk=underfit_risk
        )
    
    def _estimate_params(self, cv_result: CVResult) -> int:
        """估计模型参数数量"""
        # 从最后一个fold的模型估计
        if not cv_result.fitted_models:
            return 0
        model = cv_result.fitted_models[-1]
        
        # 尝试获取参数数量
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # 估计复杂度：n_estimators * 平均树深度
                if 'n_estimators' in params:
                    return params.get('n_estimators', 1) * 100
                return len(params)
        except Exception:
            # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
            pass
        
        # 根据模型类型启发式估计
        name = cv_result.model_name.lower()
        if 'random' in name or 'extra' in name:
            return 200 * 100  # 200棵树
        elif 'gradient' in name or 'xgboost' in name or 'lightgbm' in name or 'catboost' in name:
            return 200 * 50
        elif 'mlp' in name or 'neural' in name:
            return 10000
        elif 'svm' in name:
            return 5000
        elif 'linear' in name or 'logistic' in name:
            return 100
        elif 'decision tree' in name:
            return 50
        elif 'knn' in name:
            return 10
        return 100
    
    def _get_range(self, values: List[float]) -> Tuple[float, float]:
        """获取数值范围，处理边界情况"""
        if not values:
            return (0, 1)
        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            return (min_v - 1, max_v + 1)
        return (min_v, max_v)
    
    def _normalize(self, value: float, min_v: float, max_v: float,
                   higher_better: bool = True) -> float:
        """归一化到 0-100"""
        if max_v == min_v:
            return 50.0
        
        normalized = (value - min_v) / (max_v - min_v) * 100
        
        if not higher_better:
            normalized = 100 - normalized
        
        return max(0, min(100, normalized))
    
    def _assess_overfit_risk(self, cv_result: CVResult,
                              task_type: TaskType,
                              primary_metric: str) -> RiskLevel:
        """评估过拟合风险"""
        fold_scores = cv_result.fold_scores.get(primary_metric, [])
        if len(fold_scores) < 2:
            return RiskLevel.UNKNOWN
        
        std = np.std(fold_scores)
        mean = np.mean(fold_scores)
        
        # 变异系数 > 0.1 认为高不稳定
        cv_value = std / (abs(mean) + 1e-8)
        
        if cv_value > 0.15:
            return RiskLevel.HIGH
        elif cv_value > 0.08:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _assess_underfit_risk(self, cv_result: CVResult,
                               task_type: TaskType,
                               primary_metric: str) -> RiskLevel:
        """评估欠拟合风险"""
        score = cv_result.mean_scores.get(primary_metric, 0)
        
        if task_type == TaskType.CLASSIFICATION:
            if score < 0.5:
                return RiskLevel.HIGH
            elif score < 0.7:
                return RiskLevel.MEDIUM
            return RiskLevel.LOW
        else:
            if score < 0:
                return RiskLevel.HIGH
            elif score < 0.3:
                return RiskLevel.MEDIUM
            return RiskLevel.LOW


# =============================================================================
# 自动决策引擎
# =============================================================================

class AutoDecisionEngine:
    """
    自动决策引擎
    
    基于评估分数自动推荐最优模型，同时允许用户干预。
    
    场景模式：
    - accuracy_first: 比赛打榜，精度至上
    - speed_first: 实时推理，延迟敏感
    - stability_first: 生产环境，要求稳定
    - simplicity_first: 快速迭代，简单可控
    - balanced: 一般场景，综合最优
    """
    
    def __init__(self, mode: Union[str, DecisionMode] = DecisionMode.BALANCED,
                 custom_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Args:
            mode: 决策模式
            custom_weights: 自定义权重（mode='custom'时使用）
        """
        if isinstance(mode, str):
            self.mode = DecisionMode(mode)
        else:
            self.mode = mode
        
        self.custom_weights = custom_weights or {}
        self._mode_descriptions = {
            DecisionMode.ACCURACY_FIRST: "精度优先模式：选择CV分数最高的模型，适合比赛打榜",
            DecisionMode.SPEED_FIRST: "速度优先模式：选择训练快、推断快的模型，适合实时应用",
            DecisionMode.STABILITY_FIRST: "稳定性优先模式：选择CV方差最小的模型，适合生产环境",
            DecisionMode.SIMPLICITY_FIRST: "简单优先模式：选择复杂度低的模型，防止过拟合",
            DecisionMode.BALANCED: "平衡模式：综合考虑精度、速度、稳定性，适合大多数场景",
            DecisionMode.CUSTOM: "自定义模式：使用用户指定的权重",
        }
    
    def decide(self, model_scores: List[ModelScore],
               user_override: Optional[str] = None) -> DecisionReport:
        """
        做出决策
        
        Args:
            model_scores: 模型评估分数列表
            user_override: 用户覆盖选择（模型key，None=自动决策）
            
        Returns:
            DecisionReport
        """
        if not model_scores:
            return DecisionReport(mode=self.mode, recommended_model="", confidence=0.0)
        
        report = DecisionReport(mode=self.mode)
        report.mode_description = self._mode_descriptions.get(self.mode, "")
        report.scores = model_scores
        
        # 获取权重
        weights = self._get_weights()
        
        # 计算各模型在当前模式下的得分
        scored_models = []
        for ms in model_scores:
            mode_score = (
                weights['accuracy'] * ms.accuracy_score +
                weights['speed'] * ms.speed_score +
                weights['stability'] * ms.stability_score +
                weights['simplicity'] * ms.simplicity_score +
                weights['generalization'] * ms.generalization_score
            )
            scored_models.append((ms, mode_score))
        
        # 排序
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # 构建对比表
        report.comparison_table = self._build_comparison_table(scored_models, weights)
        
        # 确定推荐
        if user_override:
            # 用户覆盖
            for ms, mode_score in scored_models:
                if ms.model_key == user_override:
                    report.recommended_model = ms.model_key
                    report.recommended_name = ms.model_name
                    report.recommendation_reason = f"用户指定选择 {ms.model_name}"
                    report.confidence = 1.0
                    break
            else:
                report.recommended_model = scored_models[0][0].model_key
                report.recommended_name = scored_models[0][0].model_name
                report.recommendation_reason = f"用户指定的 {user_override} 不可用，自动推荐 {scored_models[0][0].model_name}"
                report.confidence = 0.5
        else:
            # 自动决策
            best_ms, best_score = scored_models[0]
            second_ms, second_score = scored_models[1] if len(scored_models) > 1 else (None, 0)
            
            report.recommended_model = best_ms.model_key
            report.recommended_name = best_ms.model_name
            all_scores = [score for _, score in scored_models]
            report.confidence = self._calculate_confidence(best_score, second_score, all_scores)
            report.recommendation_reason = self._generate_reason(best_ms, self.mode, weights)
        
        # 风险分析
        report.risks = self._analyze_risks(scored_models[0][0])
        
        # 场景建议
        report.scenario_advice = self._generate_scenario_advice(scored_models)
        
        # 用户覆盖选项
        report.override_options = self._generate_override_options(scored_models)
        
        return report
    
    def _get_weights(self) -> Dict[str, float]:
        """获取当前模式的权重"""
        if self.mode == DecisionMode.CUSTOM:
            return {
                'accuracy': self.custom_weights.get('accuracy', 0.2),
                'speed': self.custom_weights.get('speed', 0.2),
                'stability': self.custom_weights.get('stability', 0.2),
                'simplicity': self.custom_weights.get('simplicity', 0.2),
                'generalization': self.custom_weights.get('generalization', 0.2),
            }
        
        w = _MODE_WEIGHTS.get(self.mode, _MODE_WEIGHTS[DecisionMode.BALANCED])
        return {
            'accuracy': w[0],
            'speed': w[1],
            'stability': w[2],
            'simplicity': w[3],
            'generalization': w[4],
        }
    
    def _build_comparison_table(self, scored_models: List[Tuple[ModelScore, float]],
                                 weights: Dict[str, float]) -> pd.DataFrame:
        """构建对比表格"""
        rows = []
        for ms, mode_score in scored_models:
            row = {
                '排名': ms.rank,
                '模型': ms.model_name,
                '模式得分': round(mode_score, 1),
                f'精度({weights["accuracy"]:.0%})': round(ms.accuracy_score, 1),
                f'速度({weights["speed"]:.0%})': round(ms.speed_score, 1),
                f'稳定性({weights["stability"]:.0%})': round(ms.stability_score, 1),
                f'简单度({weights["simplicity"]:.0%})': round(ms.simplicity_score, 1),
                f'泛化({weights["generalization"]:.0%})': round(ms.generalization_score, 1),
                '主指标': f"{ms.primary_score:.4f} ± {ms.primary_std:.4f}",
                '耗时': f"{ms.train_time:.1f}s",
                '过拟合风险': ms.overfit_risk.value,
                '欠拟合风险': ms.underfit_risk.value,
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    def _calculate_confidence(self, best_score: float, second_score: float, all_scores: Optional[List[float]] = None) -> float:
        """计算推荐置信度
        
        使用效应量（gap / std）而非 gap / best_score，避免分数接近 1 时置信度失效。
        gap = 0.5 std -> confidence = 0.625
        gap = 1.0 std -> confidence = 0.75
        gap = 2.0 std -> confidence = 0.99
        """
        gap = abs(best_score - second_score)
        
        # 如果有多个模型分数，使用标准差计算效应量
        if all_scores and len(all_scores) > 1:
            std = float(np.std(all_scores))
            if std > 1e-10:
                effect_size = gap / std
                confidence = min(0.99, 0.5 + effect_size * 0.25)
                return round(confidence, 2)
        
        # 回退到原始公式（单模型或 std=0 时）
        confidence = min(0.99, 0.5 + gap / max(abs(best_score), 1e-10))
        return round(confidence, 2)
    
    def _generate_reason(self, ms: ModelScore, mode: DecisionMode,
                         weights: Dict[str, float]) -> str:
        """生成推荐理由"""
        reasons = []
        
        # 最强维度
        scores = {
            '精度': (ms.accuracy_score, weights['accuracy']),
            '速度': (ms.speed_score, weights['speed']),
            '稳定性': (ms.stability_score, weights['stability']),
            '简单度': (ms.simplicity_score, weights['simplicity']),
            '泛化能力': (ms.generalization_score, weights['generalization']),
        }
        
        # 按加权分数排序
        sorted_dims = sorted(scores.items(), key=lambda x: x[1][0] * x[1][1], reverse=True)
        top_dim = sorted_dims[0]
        
        reasons.append(f"在{self._mode_descriptions[mode].split('：')[0]}下，")
        reasons.append(f"{ms.model_name} 的{top_dim[0]}表现最优（{top_dim[1][0]:.1f}分），")
        
        # 主指标
        reasons.append(f"{ms.primary_metric}={ms.primary_score:.4f}（±{ms.primary_std:.4f}）")
        
        # 风险补充
        if ms.overfit_risk == RiskLevel.HIGH:
            reasons.append("，但需注意过拟合风险较高")
        elif ms.overfit_risk == RiskLevel.LOW and ms.underfit_risk == RiskLevel.LOW:
            reasons.append("，且过拟合/欠拟合风险均较低")
        
        return "".join(reasons)
    
    def _analyze_risks(self, ms: ModelScore) -> List[str]:
        """分析风险"""
        risks = []
        
        if ms.overfit_risk == RiskLevel.HIGH:
            risks.append(f"[{ms.model_name}] 过拟合风险高：CV标准差较大（{ms.primary_std:.4f}），建议增加正则化或减少模型复杂度")
        elif ms.overfit_risk == RiskLevel.MEDIUM:
            risks.append(f"[{ms.model_name}] 过拟合风险中等：CV表现有一定波动")
        
        if ms.underfit_risk == RiskLevel.HIGH:
            risks.append(f"[{ms.model_name}] 欠拟合风险高：主指标得分较低（{ms.primary_score:.4f}），模型可能过于简单")
        
        if ms.train_time > 60:
            risks.append(f"[{ms.model_name}] 训练耗时较长（{ms.train_time:.1f}s），若需频繁重训请考虑更轻量模型")
        
        return risks
    
    def _generate_scenario_advice(self, scored_models: List[Tuple[ModelScore, float]]) -> str:
        """生成场景建议"""
        if not scored_models:
            return ""
        
        best_ms = scored_models[0][0]
        
        if best_ms.overfit_risk == RiskLevel.HIGH:
            return "当前最优模型有过拟合倾向，建议：1) 增加数据量 2) 使用更强的正则化 3) 考虑交叉验证更稳定的模型"
        
        if best_ms.underfit_risk == RiskLevel.HIGH:
            return "当前模型表现不佳，建议：1) 检查特征工程 2) 尝试更复杂的模型 3) 增加模型容量"
        
        return "当前推荐模型综合表现良好，可直接用于后续预测"
    
    def _generate_override_options(self, scored_models: List[Tuple[ModelScore, float]]) -> Dict[str, str]:
        """生成用户覆盖选项"""
        options = {}
        
        for ms, _ in scored_models[:5]:
            reasons = []
            if ms.accuracy_score >= 80:
                reasons.append("精度高")
            if ms.speed_score >= 80:
                reasons.append("速度快")
            if ms.stability_score >= 80:
                reasons.append("稳定性好")
            if ms.simplicity_score >= 80:
                reasons.append("模型简单")
            
            reason_str = ", ".join(reasons) if reasons else "综合表现"
            options[ms.model_key] = f"{ms.model_name}（{reason_str}）"
        
        return options


# =============================================================================
# 集成便捷函数
# =============================================================================

def auto_select(cv_results: List[CVResult],
                task_type: Union[str, TaskType],
                mode: str = 'balanced',
                user_override: Optional[str] = None,
                primary_metric: Optional[str] = None) -> DecisionReport:
    """
    一键自动选择最优模型
    
    Args:
        cv_results: 各模型CV结果
        task_type: 任务类型
        mode: 决策模式 ('accuracy_first', 'speed_first', 'stability_first', 'simplicity_first', 'balanced')
        user_override: 用户覆盖选择
        primary_metric: 主指标
        
    Returns:
        DecisionReport
        
    示例：
        report = auto_select(cv_results, 'classification', mode='balanced')
        print(report.recommended_name)
        print(report.comparison_table.to_string(index=False))
        
        # 用户覆盖
        report = auto_select(cv_results, 'classification', user_override='xgb')
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    
    # 评估
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate_all(cv_results, task_type, primary_metric)
    
    # 决策
    mode_enum = DecisionMode(mode) if mode else DecisionMode.BALANCED
    engine = AutoDecisionEngine(mode=mode_enum)
    report = engine.decide(scores, user_override=user_override)
    
    return report


def print_decision_report(report: DecisionReport) -> None:
    """打印决策报告"""
    print("\n" + "=" * 70)
    print("模型自动评估与决策报告".center(60))
    print("=" * 70)
    
    print(f"\n【决策模式】{report.mode_description}")
    
    print(f"\n【推荐结果】")
    print(f"  推荐模型: {report.recommended_name}")
    print(f"  置信度: {report.confidence:.0%}")
    print(f"  推荐理由: {report.recommendation_reason}")
    
    if report.risks:
        print(f"\n【风险提示】")
        for risk in report.risks:
            print(f"  ⚠️ {risk}")
    
    if report.scenario_advice:
        print(f"\n【场景建议】")
        print(f"  {report.scenario_advice}")
    
    print(f"\n【完整对比】")
    if report.comparison_table is not None:
        print(report.comparison_table.to_string(index=False))
    
    print(f"\n【用户覆盖选项】")
    for key, desc in report.override_options.items():
        marker = "★ 推荐" if key == report.recommended_model else "  "
        print(f"  {marker} {key:12s}: {desc}")
    
    print("\n" + "=" * 70)
    print("如需覆盖自动选择，请在 ModelingEngine 中设置 model_keys=['模型key']")
    print("=" * 70)
