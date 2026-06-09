"""
AutoML 策略推荐器

基于数据元特征自动推荐最优的优化器、模型列表和集成策略。
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from core.meta_feature_extractor import MetaFeatures
from core.modeling_engine import TaskType

# 阈值常量：将散落在各 _recommend_* 方法中的"小/中/大数据"判断统一管理
# 这些数字在 5 个方法中被引用 10+ 次，原写为裸数字不利于调整。
_BIG_DATA_SAMPLES = 100_000       # 大数据阈值
_MEDIUM_DATA_SAMPLES = 10_000     # 中等数据阈值
_SMALL_DATA_SAMPLES = 1_000       # 小数据阈值
_HIGH_COMPLEXITY = 70             # 高复杂度阈值（满分 100）
_MEDIUM_COMPLEXITY = 60           # 中等复杂度阈值

# 优化器 → 时间倍增映射：原代码每次 _estimate_time 调用都重建此 dict，提到模块级避免重复
_OPTIMIZER_TIME_MULTIPLIER = {
    'random': 1.0,
    'bayesian': 2.0,
    'tpe': 2.0,
    'cmaes': 2.5,
    'rl': 3.0,
    'genetic': 2.5,
    'hyperband': 1.5,
}


@dataclass
class StrategyRecommendation:
    """策略推荐结果"""
    optimizer: str
    model_keys: List[str]
    ensemble: str
    deep_learning: Dict[str, Any]
    expected_time: str
    reasoning: str


class AutoMLStrategy:
    """
    AutoML 策略推荐器
    
    基于元特征规则自动选择：
    - 优化器策略（贝叶斯/RL/Hyperband/遗传/随机）
    - 模型列表（传统ML / 深度学习）
    - 集成策略
    
    使用方式：
        meta = MetaFeatureExtractor().extract(X, y, task_type)
        rec = AutoMLStrategy.recommend(meta, user_preference='balanced')
        print(rec.reasoning)
    """
    
    @staticmethod
    def recommend(meta: MetaFeatures, task_type: TaskType,
                  user_preference: str = 'balanced',
                  user_optimizer: Optional[str] = None,
                  user_model_keys: Optional[List[str]] = None) -> StrategyRecommendation:
        """
        基于元特征推荐 AutoML 策略
        
        Args:
            meta: 数据集元特征
            task_type: 任务类型
            user_preference: 用户偏好 ('speed_first', 'accuracy_first', 'balanced', 'exploration')
            user_optimizer: 用户指定的优化器（None=自动推荐）
            user_model_keys: 用户指定的模型列表（None=自动推荐）
            
        Returns:
            StrategyRecommendation
        """
        reasoning_parts = []
        
        # ========== 1. 推荐优化器 ==========
        if user_optimizer:
            optimizer = user_optimizer
            reasoning_parts.append(f"用户指定优化器: {optimizer}")
        else:
            optimizer = AutoMLStrategy._recommend_optimizer(meta, user_preference)
            reasoning_parts.append(f"自动推荐优化器: {optimizer}")
        
        # ========== 2. 推荐模型列表 ==========
        if user_model_keys:
            model_keys = user_model_keys
            reasoning_parts.append(f"用户指定模型: {', '.join(model_keys)}")
        else:
            model_keys = AutoMLStrategy._recommend_models(meta, task_type, user_preference)
            reasoning_parts.append(f"自动推荐模型: {', '.join(model_keys)}")
        
        # ========== 3. 推荐集成策略 ==========
        ensemble = AutoMLStrategy._recommend_ensemble(meta, user_preference)
        
        # ========== 4. 深度学习建议 ==========
        deep_learning = AutoMLStrategy._recommend_deep_learning(meta, user_preference)
        
        # ========== 5. 预期时间 ==========
        expected_time = AutoMLStrategy._estimate_time(meta, optimizer, model_keys)
        
        # 构建推荐理由
        reasoning = AutoMLStrategy._build_reasoning(meta, optimizer, model_keys, ensemble, reasoning_parts)
        
        return StrategyRecommendation(
            optimizer=optimizer,
            model_keys=model_keys,
            ensemble=ensemble,
            deep_learning=deep_learning,
            expected_time=expected_time,
            reasoning=reasoning
        )
    
    @staticmethod
    def _recommend_optimizer(meta: MetaFeatures, preference: str) -> str:
        """推荐优化器策略"""
        # 速度优先
        if preference == 'speed_first':
            if meta.n_samples > _BIG_DATA_SAMPLES // 2:
                return 'hyperband'
            return 'random'

        # 精度优先
        if preference == 'accuracy_first':
            if meta.complexity_score > _MEDIUM_COMPLEXITY:
                return 'genetic'
            return 'bayesian'

        # 探索模式（愿意尝试新方法）
        if preference == 'exploration':
            if meta.n_samples > _MEDIUM_DATA_SAMPLES and meta.n_features > 50:
                return 'rl'
            return 'genetic'

        # 平衡模式（默认）
        if meta.n_samples > _BIG_DATA_SAMPLES:
            return 'hyperband'  # 大数据用多保真度
        elif meta.n_samples < _SMALL_DATA_SAMPLES:
            return 'bayesian'   # 小数据用贝叶斯
        elif meta.complexity_score > _HIGH_COMPLEXITY:
            return 'genetic'    # 高复杂度用遗传算法
        else:
            return 'bayesian'
    
    @staticmethod
    def _recommend_models(meta: MetaFeatures, task_type: TaskType, preference: str) -> List[str]:
        """推荐模型列表"""
        # 类别任务
        if task_type == TaskType.CLASSIFICATION:
            base_models = ['lr', 'xgb', 'lgb', 'rf']

            if preference == 'speed_first':
                return ['lr', 'dt', 'lgb']

            if preference == 'accuracy_first':
                base_models.extend(['et', 'gbdt', 'svm'])

            # 大数据：去掉慢模型
            if meta.n_samples > _BIG_DATA_SAMPLES:
                base_models = [m for m in base_models if m not in ['svm', 'knn']]

            # 高维：把线性模型放最前（'lr' 必然已在 base_models，insert(0, 'lr') 是 no-op，但保留
            # 兜底逻辑以防 base_models 初始列表未来被修改时漏掉 'lr'）
            if meta.n_features > 100 and 'lr' not in base_models:
                base_models.insert(0, 'lr')

            # 类别不平衡：增加对不平衡鲁棒的模型（同理，'xgb' 必然已在，append 是 no-op，
            # 保留以防未来 base_models 初始列表变化时漏掉 'xgb'）
            if meta.class_imbalance_ratio > 5 and 'xgb' not in base_models:
                base_models.append('xgb')

            return base_models[:5]  # 最多5个

        # 回归任务
        else:
            base_models = ['ridge', 'xgb', 'lgb', 'rf']

            if preference == 'speed_first':
                return ['ridge', 'linear', 'lgb']

            if preference == 'accuracy_first':
                base_models.extend(['et', 'gbdt', 'svr'])

            # 大数据：用 LinearSVR 替代 SVR，保留更多模型
            if meta.n_samples > _BIG_DATA_SAMPLES:
                base_models = [m for m in base_models if m not in ['svr', 'knn']]
                if 'linear_svr' not in base_models:
                    base_models.append('linear_svr')

            return base_models[:5]
    
    @staticmethod
    def _recommend_ensemble(meta: MetaFeatures, preference: str) -> str:
        """推荐集成策略"""
        if preference == 'speed_first':
            return 'best_single'
        if meta.n_samples < _SMALL_DATA_SAMPLES // 2:
            return 'best_single'  # 小数据集成容易过拟合
        if meta.complexity_score > _HIGH_COMPLEXITY:
            return 'stacking'
        return 'weighted'
    
    @staticmethod
    def _recommend_deep_learning(meta: MetaFeatures, preference: str) -> Dict[str, Any]:
        """推荐深度学习配置"""
        dl_config = {'enabled': False, 'models': []}
        
        # 只有精度优先或探索模式才启用 DL
        if preference not in ('accuracy_first', 'exploration'):
            return dl_config
        
        # 大数据 + 高维才启用
        if meta.n_samples < _SMALL_DATA_SAMPLES or meta.n_features < 10:
            return dl_config
        
        dl_config['enabled'] = True
        dl_config['models'] = ['torch_mlp']
        
        # 序列/表格特征明显时启用 CNN/LSTM
        if meta.n_features >= 20:
            dl_config['models'].append('torch_cnn1d')
        
        return dl_config
    
    @staticmethod
    def _estimate_time(meta: MetaFeatures, optimizer: str, model_keys: List[str]) -> str:
        """预估训练时间"""
        base_time = len(model_keys) * 2  # 每个模型约2分钟基础时间
        
        # 优化器倍增（dict 提到模块级 _OPTIMIZER_TIME_MULTIPLIER，避免每次重建）
        multiplier = _OPTIMIZER_TIME_MULTIPLIER.get(optimizer, 1.5)
        
        # 数据规模倍增
        if meta.n_samples > _BIG_DATA_SAMPLES:
            multiplier *= 2.0
        elif meta.n_samples > _MEDIUM_DATA_SAMPLES:
            multiplier *= 1.5
        
        total_minutes = base_time * multiplier
        
        if total_minutes < 2:
            return '很快 (< 2分钟)'
        elif total_minutes < 5:
            return '较快 (2-5分钟)'
        elif total_minutes < 15:
            return '中等 (5-15分钟)'
        elif total_minutes < 30:
            return '较慢 (15-30分钟)'
        else:
            return '很慢 (> 30分钟)'
    
    @staticmethod
    def _build_reasoning(meta: MetaFeatures, optimizer: str, models: List[str],
                         ensemble: str, parts: List[str]) -> str:
        """构建推荐理由文本"""
        reasons = []
        reasons.append(f"数据规模: {meta.n_samples} 样本 × {meta.n_features} 特征")
        reasons.append(f"复杂度评分: {meta.complexity_score:.0f}/100")
        
        if meta.missing_ratio > 0.01:
            reasons.append(f"缺失率: {meta.missing_ratio*100:.1f}%")
        if meta.class_imbalance_ratio > 3:
            reasons.append(f"类别不平衡: {meta.class_imbalance_ratio:.1f} 倍")
        if meta.sparsity > 0.1:
            reasons.append(f"稀疏度: {meta.sparsity*100:.1f}%")
        
        reasons.extend(parts)
        reasons.append(f"集成策略: {ensemble}")
        
        return '\n'.join(reasons)
