"""
元学习模型推荐器

基于数据集元特征的相似度搜索，从知识库中找到最相似的历史数据集，
推荐其表现最好的模型。支持特征类型感知和在线学习。

核心组件:
1. DatasetFingerprint - 数据集指纹（扩展元特征）
2. MetaKnowledgeBase - 元知识库（内存+磁盘持久化）
3. MetaLearningModelRecommender - 推荐引擎
"""
import json
import math
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from core.modeling_engine import TaskType
from core.meta_feature_extractor import MetaFeatureExtractor, MetaFeatures
from core.automl_strategy import AutoMLStrategy
from utils.helpers import log_warning


@dataclass
class DatasetFingerprint:
    """数据集指纹（扩展元特征，用于相似度计算）"""
    
    # 基础元特征（来自 MetaFeatureExtractor）
    n_samples: int = 0
    n_features: int = 0
    n_numeric: int = 0
    n_categorical: int = 0
    sample_feature_ratio: float = 0.0
    numeric_ratio: float = 0.0
    categorical_ratio: float = 0.0
    missing_ratio: float = 0.0
    sparsity: float = 0.0
    feature_correlation_mean: float = 0.0
    feature_correlation_max: float = 0.0
    n_classes: int = 0
    class_imbalance_ratio: float = 1.0
    target_entropy: float = 0.0
    target_std: float = 0.0
    complexity_score: float = 0.0
    
    # 扩展特征（用于更精确匹配）
    n_text_features: int = 0              # 文本型特征数
    text_ratio: float = 0.0               # 文本特征比例
    n_datetime_features: int = 0          # 时间型特征数
    datetime_ratio: float = 0.0           # 时间特征比例
    
    n_high_cardinality: int = 0           # 高基数类别特征数 (>50 unique)
    high_cardinality_ratio: float = 0.0   # 高基数比例
    
    mean_skewness: float = 0.0            # 数值特征平均偏度
    mean_kurtosis: float = 0.0            # 数值特征平均峰度
    
    # 数据质量
    outlier_ratio: float = 0.0            # 异常值比例（3-sigma）
    duplicate_ratio: float = 0.0          # 重复行比例
    
    # 任务信息
    task_type: str = ""
    
    @classmethod
    def from_meta_features(cls, meta: 'MetaFeatures', X: pd.DataFrame,
                           task_type: TaskType) -> 'DatasetFingerprint':
        """从 MetaFeatures 和原始数据构建完整指纹"""
        fp = cls(
            n_samples=meta.n_samples,
            n_features=meta.n_features,
            n_numeric=meta.n_numeric,
            n_categorical=meta.n_categorical,
            sample_feature_ratio=meta.sample_feature_ratio,
            numeric_ratio=meta.numeric_ratio,
            categorical_ratio=meta.categorical_ratio,
            missing_ratio=meta.missing_ratio,
            sparsity=meta.sparsity,
            feature_correlation_mean=meta.feature_correlation_mean,
            feature_correlation_max=meta.feature_correlation_max,
            n_classes=meta.n_classes,
            class_imbalance_ratio=meta.class_imbalance_ratio,
            target_entropy=meta.target_entropy,
            target_std=meta.target_std,
            complexity_score=meta.complexity_score,
            task_type=task_type.value if isinstance(task_type, TaskType) else str(task_type),
        )
        
        # 计算扩展特征
        fp._compute_extended_features(X)
        return fp
    
    def _compute_extended_features(self, X: pd.DataFrame) -> None:
        """计算扩展特征"""
        if X.empty:
            return

        # 一次遍历收集每列的 dtype 和 nunique
        # 原代码 N 列 × 4-5 次 dtype/nunique 查询 = O(4-5N) 重复工作
        # 修复后：单次 O(N) 收集，后续 O(1) 查表
        col_dtype = {}
        col_nunique = {}
        for c in X.columns:
            col_dtype[c] = X[c].dtype
            col_nunique[c] = X[c].nunique(dropna=True)

        # 文本特征
        text_cols = [c for c in X.columns if col_dtype[c] == object and col_nunique[c] > 10]
        self.n_text_features = len(text_cols)
        self.text_ratio = self.n_text_features / max(len(X.columns), 1)

        # 时间特征
        datetime_cols = [c for c in X.columns if 'datetime' in str(col_dtype[c])]
        self.n_datetime_features = len(datetime_cols)
        self.datetime_ratio = self.n_datetime_features / max(len(X.columns), 1)

        # 高基数特征
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(col_dtype[c])]
        high_card = [c for c in cat_cols if col_nunique[c] > 50]
        self.n_high_cardinality = len(high_card)
        self.high_cardinality_ratio = self.n_high_cardinality / max(len(cat_cols), 1)

        # 偏度和峰度
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(col_dtype[c])]
        if numeric_cols:
            skew_vals = []
            kurt_vals = []
            outlier_counts = 0
            total_numeric = 0
            for col in numeric_cols:
                vals = X[col].dropna()
                if len(vals) > 3:
                    skew_vals.append(vals.skew())
                    # pandas kurtosis 默认是 excess kurtosis
                    kurt_vals.append(vals.kurtosis())
                    # 3-sigma 异常值
                    mean, std = vals.mean(), vals.std()
                    if std > 0:
                        outlier_counts += ((vals - mean).abs() > 3 * std).sum()
                        total_numeric += len(vals)
            if skew_vals:
                self.mean_skewness = float(np.mean(skew_vals))
                self.mean_kurtosis = float(np.mean(kurt_vals))
            if total_numeric > 0:
                self.outlier_ratio = outlier_counts / total_numeric
        
        # 重复行
        if len(X) > 0:
            n_dups = X.duplicated().sum()
            self.duplicate_ratio = n_dups / len(X)
    
    def to_vector(self) -> np.ndarray:
        """转为数值向量（用于相似度计算）"""
        # 对数变换处理尺度差异大的特征
        log_samples = math.log1p(self.n_samples)
        log_features = math.log1p(self.n_features)
        
        return np.array([
            log_samples,
            log_features,
            self.sample_feature_ratio,
            self.numeric_ratio,
            self.categorical_ratio,
            self.text_ratio,
            self.datetime_ratio,
            self.high_cardinality_ratio,
            self.missing_ratio,
            self.sparsity,
            self.feature_correlation_mean,
            self.feature_correlation_max,
            self.class_imbalance_ratio,
            self.target_entropy,
            self.target_std,
            self.complexity_score / 100.0,
            self.mean_skewness,
            self.mean_kurtosis,
            self.outlier_ratio,
            self.duplicate_ratio,
        ], dtype=np.float64)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DatasetFingerprint':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelPerformanceRecord:
    """单条模型性能记录"""
    model_key: str
    score: float
    metric: str
    cv_std: float = 0.0
    train_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class KnowledgeRecord:
    """知识库单条记录"""
    fingerprint: DatasetFingerprint
    performances: List[ModelPerformanceRecord] = field(default_factory=list)
    best_model: str = ""
    best_score: float = 0.0
    dataset_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fingerprint': self.fingerprint.to_dict(),
            'performances': [
                {'model_key': p.model_key, 'score': p.score, 'metric': p.metric,
                 'cv_std': p.cv_std, 'train_time': p.train_time, 'timestamp': p.timestamp}
                for p in self.performances
            ],
            'best_model': self.best_model,
            'best_score': self.best_score,
            'dataset_name': self.dataset_name,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KnowledgeRecord':
        rec = cls(
            fingerprint=DatasetFingerprint.from_dict(d['fingerprint']),
            best_model=d.get('best_model', ''),
            best_score=d.get('best_score', 0.0),
            dataset_name=d.get('dataset_name', ''),
        )
        for p in d.get('performances', []):
            rec.performances.append(ModelPerformanceRecord(
                model_key=p['model_key'],
                score=p['score'],
                metric=p['metric'],
                cv_std=p.get('cv_std', 0.0),
                train_time=p.get('train_time', 0.0),
                timestamp=p.get('timestamp', time.time())
            ))
        return rec


class MetaKnowledgeBase:
    """
    元知识库
    
    存储和管理历史数据集指纹及其模型性能记录。
    支持内存缓存和磁盘持久化。
    """
    
    def __init__(self, disk_path: Optional[str] = None,
                 max_records: int = 500) -> None:
        self.records: List[KnowledgeRecord] = []
        self.max_records = max_records
        self.disk_path = disk_path
        
        # 特征权重（用于相似度计算，可在线学习）
        self.feature_weights: Optional[np.ndarray] = None
        
        if disk_path and os.path.exists(disk_path):
            self.load()
    
    def add(self, record: KnowledgeRecord) -> None:
        """添加记录"""
        self.records.append(record)
        
        # 限制数量，保留最新的
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
    
    def find_similar(self, fingerprint: DatasetFingerprint,
                     k: int = 5,
                     task_type_filter: Optional[str] = None) -> List[Tuple[KnowledgeRecord, float]]:
        """
        查找最相似的历史记录
        
        Returns:
            [(record, similarity_score), ...] 按相似度降序
        """
        if not self.records:
            return []
        
        query_vec = fingerprint.to_vector()
        weights = self.feature_weights

        similarities = []
        for record in self.records:
            # 任务类型过滤
            if task_type_filter and record.fingerprint.task_type != task_type_filter:
                continue

            ref_vec = record.fingerprint.to_vector()
            sim = self._cosine_similarity(query_vec, ref_vec, weights)
            similarities.append((record, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> float:
        """计算加权余弦相似度"""
        if weights is not None:
            a = a * weights
            b = b * weights
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def update_weights(self, fingerprint: DatasetFingerprint,
                       actual_best_model: str,
                       predicted_models: List[str]) -> None:
        """
        在线更新特征权重
        
        如果推荐准确，增强相关特征权重；如果不准确，微调降低。
        这是一个简单的 perceptron-style 更新。
        """
        if actual_best_model in predicted_models:
            return  # 推荐正确，不调整
        
        # 初始化权重
        # 缓存 fingerprint.to_vector()：原代码在 update_weights 内 2 次调用
        # 同一个对象 + 1 次在循环内 record.fingerprint.to_vector()，3 次冗余
        # 实际只需调 1 次（line 339 还需要 record.fingerprint.to_vector()）
        fp_vec = fingerprint.to_vector()
        dim = len(fp_vec)
        if self.feature_weights is None:
            self.feature_weights = np.ones(dim, dtype=np.float64)

        # 找到实际最佳模型对应的历史记录
        for record in self.records:
            best = record.best_model
            if best == actual_best_model:
                diff = np.abs(fp_vec - record.fingerprint.to_vector())
                # 差异大的维度权重应降低（说明这些维度不重要）
                self.feature_weights *= (1.0 - 0.01 * diff)
                self.feature_weights = np.clip(self.feature_weights, 0.1, 5.0)
                break
    
    def save(self) -> None:
        """保存到磁盘"""
        if not self.disk_path:
            return
        data = {
            'records': [r.to_dict() for r in self.records],
            'feature_weights': self.feature_weights.tolist() if self.feature_weights is not None else None,
        }
        os.makedirs(os.path.dirname(self.disk_path), exist_ok=True)
        with open(self.disk_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self) -> None:
        """从磁盘加载"""
        if not self.disk_path or not os.path.exists(self.disk_path):
            return
        try:
            with open(self.disk_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get('records', []):
                self.records.append(KnowledgeRecord.from_dict(d))
            w = data.get('feature_weights')
            if w is not None:
                self.feature_weights = np.array(w, dtype=np.float64)
        except Exception as e:
            log_warning(f"[MetaKnowledgeBase] 加载失败: {e}")
    
    def __len__(self) -> int:
        return len(self.records)


class MetaLearningModelRecommender:
    """
    元学习模型推荐器
    
    基于数据集相似度，从历史知识库中推荐最优模型。
    同时保留启发式规则作为冷启动和兜底策略。
    
    使用方式:
        recommender = MetaLearningModelRecommender()
        
        # 推荐模型
        rec = recommender.recommend(X, y, task_type)
        
        # 训练后反馈
        recommender.feedback(fingerprint, actual_results)
    """
    
    def __init__(self,
                 knowledge_base: Optional[MetaKnowledgeBase] = None,
                 min_similarity: float = 0.6,
                 fallback_to_rules: bool = True) -> None:
        self.kb = knowledge_base or MetaKnowledgeBase()
        self.min_similarity = min_similarity
        self.fallback_to_rules = fallback_to_rules
        self._extractor = MetaFeatureExtractor()
    
    def recommend(self, X: pd.DataFrame, y: Optional[pd.Series],
                  task_type: TaskType,
                  preference: str = 'balanced',
                  k: int = 5) -> Dict[str, Any]:
        """
        推荐模型列表
        
        Returns:
            {
                'model_keys': List[str],        # 推荐模型key
                'similarity': float,             # 最相似记录相似度
                'reasoning': str,                # 推荐理由
                'source': str,                   # 'meta_learning' 或 'heuristic'
                'similar_datasets': List[str],   # 相似数据集名称
                'fingerprint': DatasetFingerprint,
            }
        """
        # 提取指纹
        meta = self._extractor.extract(X, y, task_type)
        fingerprint = DatasetFingerprint.from_meta_features(meta, X, task_type)
        
        # 查询知识库
        similar = self.kb.find_similar(
            fingerprint, k=k,
            task_type_filter=task_type.value if isinstance(task_type, TaskType) else str(task_type)
        )
        
        if not similar:
            return self._heuristic_recommend(fingerprint, task_type, preference)
        
        best_sim = similar[0][1]
        if best_sim < self.min_similarity:
            # 相似度不足，混合策略
            return self._hybrid_recommend(fingerprint, similar, task_type, preference)
        
        # 纯元学习推荐
        return self._meta_learning_recommend(fingerprint, similar, task_type)
    
    def _meta_learning_recommend(self, fingerprint: DatasetFingerprint,
                                  similar: List[Tuple[KnowledgeRecord, float]],
                                  task_type: TaskType) -> Dict[str, Any]:
        """基于元学习推荐"""
        # 加权投票
        model_scores: Dict[str, List[Tuple[float, float]]] = {}
        dataset_names = []
        
        for record, sim in similar:
            dataset_names.append(record.dataset_name or f"dataset_{len(dataset_names)}")
            for perf in record.performances:
                if perf.model_key not in model_scores:
                    model_scores[perf.model_key] = []
                model_scores[perf.model_key].append((perf.score, sim))
        
        # 计算加权平均分数
        weighted_scores = {}
        for model_key, scores in model_scores.items():
            total_weight = sum(sim for _, sim in scores)
            if total_weight > 0:
                weighted_scores[model_key] = sum(s * sim for s, sim in scores) / total_weight
        
        # 排序取 top
        sorted_models = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        model_keys = [m for m, _ in sorted_models[:5]]
        
        reasoning = (
            f"元学习推荐: 找到 {len(similar)} 个相似数据集，"
            f"最高相似度 {similar[0][1]:.2f}。"
            f"历史最佳模型: {similar[0][0].best_model}"
        )
        
        return {
            'model_keys': model_keys,
            'similarity': similar[0][1],
            'reasoning': reasoning,
            'source': 'meta_learning',
            'similar_datasets': dataset_names,
            'fingerprint': fingerprint,
        }
    
    def _hybrid_recommend(self, fingerprint: DatasetFingerprint,
                          similar: List[Tuple[KnowledgeRecord, float]],
                          task_type: TaskType,
                          preference: str) -> Dict[str, Any]:
        """混合推荐（元学习 + 启发式）"""
        meta_rec = self._meta_learning_recommend(fingerprint, similar, task_type)
        heuristic_rec = self._heuristic_recommend(fingerprint, task_type, preference)
        
        # 合并模型列表（元学习优先）
        combined = meta_rec['model_keys'][:3] + [m for m in heuristic_rec['model_keys'] if m not in meta_rec['model_keys']]
        
        reasoning = (
            f"混合推荐: 相似度 {meta_rec['similarity']:.2f} 低于阈值 {self.min_similarity}，"
            f"结合元学习和启发式规则。"
        )
        
        return {
            'model_keys': combined[:5],
            'similarity': meta_rec['similarity'],
            'reasoning': reasoning,
            'source': 'hybrid',
            'similar_datasets': meta_rec.get('similar_datasets', []),
            'fingerprint': fingerprint,
        }
    
    def _heuristic_recommend(self, fingerprint: DatasetFingerprint,
                             task_type: TaskType,
                             preference: str) -> Dict[str, Any]:
        """启发式兜底推荐"""
        # 构造 MetaFeatures
        meta = MetaFeatures(
            n_samples=fingerprint.n_samples,
            n_features=fingerprint.n_features,
            n_numeric=fingerprint.n_numeric,
            n_categorical=fingerprint.n_categorical,
            sample_feature_ratio=fingerprint.sample_feature_ratio,
            numeric_ratio=fingerprint.numeric_ratio,
            categorical_ratio=fingerprint.categorical_ratio,
            missing_ratio=fingerprint.missing_ratio,
            sparsity=fingerprint.sparsity,
            feature_correlation_mean=fingerprint.feature_correlation_mean,
            feature_correlation_max=fingerprint.feature_correlation_max,
            n_classes=fingerprint.n_classes,
            class_imbalance_ratio=fingerprint.class_imbalance_ratio,
            target_entropy=fingerprint.target_entropy,
            target_std=fingerprint.target_std,
            complexity_score=fingerprint.complexity_score,
        )
        
        rec = AutoMLStrategy.recommend(meta, task_type, preference)
        
        # 特征类型感知调整
        model_keys = self._adjust_by_feature_type(fingerprint, rec.model_keys)
        
        return {
            'model_keys': model_keys,
            'similarity': 0.0,
            'reasoning': f"启发式推荐: {rec.reasoning}",
            'source': 'heuristic',
            'similar_datasets': [],
            'fingerprint': fingerprint,
        }
    
    def _adjust_by_feature_type(self, fingerprint: DatasetFingerprint,
                                 model_keys: List[str]) -> List[str]:
        """根据特征类型分布调整模型优先级"""
        adjusted = list(model_keys)
        
        # 高文本比例 → 优先线性模型（对高维稀疏友好）
        if fingerprint.text_ratio > 0.3:
            for linear in ['lr', 'ridge', 'linear', 'sgd']:
                if linear in adjusted:
                    adjusted.remove(linear)
                    adjusted.insert(0, linear)
        
        # 高时间特征比例 → 优先树模型（对时间特征交互好）
        if fingerprint.datetime_ratio > 0.2:
            for tree in ['xgb', 'lgb', 'rf']:
                if tree in adjusted and tree != adjusted[0]:
                    adjusted.remove(tree)
                    adjusted.insert(0, tree)
        
        # 高基数类别多 → 优先 LightGBM（对高基数类别处理最好）
        if fingerprint.high_cardinality_ratio > 0.3:
            if 'lgb' in adjusted and adjusted[0] != 'lgb':
                adjusted.remove('lgb')
                adjusted.insert(0, 'lgb')
        
        # 高度偏态数据 → 优先鲁棒模型
        if abs(fingerprint.mean_skewness) > 2:
            for robust in ['lgb', 'xgb', 'et']:
                if robust in adjusted and robust != adjusted[0]:
                    adjusted.remove(robust)
                    adjusted.insert(0, robust)
                    break
        
        return adjusted
    
    def feedback(self, fingerprint: DatasetFingerprint,
                 model_results: Dict[str, Dict[str, Any]],
                 dataset_name: str = "") -> None:
        """
        反馈训练结果，更新知识库
        
        Args:
            fingerprint: 数据集指纹
            model_results: {model_key: {'score': float, 'metric': str, 'cv_std': float, 'train_time': float}}
            dataset_name: 数据集名称（可选）
        """
        if not model_results:
            return
        
        performances = []
        best_model = ""
        best_score = float('-inf')
        
        for model_key, result in model_results.items():
            score = result.get('score', 0.0)
            perf = ModelPerformanceRecord(
                model_key=model_key,
                score=score,
                metric=result.get('metric', ''),
                cv_std=result.get('cv_std', 0.0),
                train_time=result.get('train_time', 0.0)
            )
            performances.append(perf)
            
            if score > best_score:
                best_score = score
                best_model = model_key
        
        record = KnowledgeRecord(
            fingerprint=fingerprint,
            performances=performances,
            best_model=best_model,
            best_score=best_score,
            dataset_name=dataset_name
        )
        
        self.kb.add(record)
        
        # 在线更新权重
        predicted = [p.model_key for p in performances]
        self.kb.update_weights(fingerprint, best_model, predicted)
    
    def save(self) -> None:
        """保存知识库"""
        self.kb.save()
