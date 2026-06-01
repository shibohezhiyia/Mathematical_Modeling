"""
元学习模型推荐器测试
"""
import unittest
import tempfile
import os

import numpy as np
import pandas as pd

from core.meta_learning_recommender import (
    DatasetFingerprint, ModelPerformanceRecord, KnowledgeRecord,
    MetaKnowledgeBase, MetaLearningModelRecommender
)
from core.meta_feature_extractor import MetaFeatures
from core.modeling_engine import TaskType


class TestDatasetFingerprint(unittest.TestCase):
    """测试数据集指纹"""
    
    def test_from_meta_features(self):
        """测试从元特征构建"""
        meta = MetaFeatures(n_samples=100, n_features=10, n_numeric=7, n_categorical=3)
        X = pd.DataFrame(np.random.randn(100, 10))
        fp = DatasetFingerprint.from_meta_features(meta, X, TaskType.CLASSIFICATION)
        self.assertEqual(fp.n_samples, 100)
        self.assertEqual(fp.task_type, 'classification')
    
    def test_vector_conversion(self):
        """测试向量转换"""
        fp = DatasetFingerprint(n_samples=100, n_features=10)
        vec = fp.to_vector()
        self.assertGreater(len(vec), 0)
        self.assertTrue(np.isfinite(vec).all())
    
    def test_serialization(self):
        """测试序列化"""
        fp = DatasetFingerprint(n_samples=100, n_features=10, task_type='classification')
        d = fp.to_dict()
        fp2 = DatasetFingerprint.from_dict(d)
        self.assertEqual(fp2.n_samples, 100)
        self.assertEqual(fp2.task_type, 'classification')


class TestMetaKnowledgeBase(unittest.TestCase):
    """测试元知识库"""
    
    def test_add_and_find(self):
        """测试添加和查找"""
        kb = MetaKnowledgeBase()
        fp = DatasetFingerprint(n_samples=1000, n_features=20, task_type='classification')
        record = KnowledgeRecord(
            fingerprint=fp,
            performances=[ModelPerformanceRecord('xgb', 0.9, 'auc')],
            best_model='xgb',
            best_score=0.9
        )
        kb.add(record)
        
        similar = kb.find_similar(fp, k=1)
        self.assertEqual(len(similar), 1)
        self.assertAlmostEqual(similar[0][1], 1.0, places=5)  # 自己与自己完全相似
    
    def test_find_similar_task_filter(self):
        """测试任务类型过滤"""
        kb = MetaKnowledgeBase()
        kb.add(KnowledgeRecord(
            fingerprint=DatasetFingerprint(n_samples=100, task_type='classification'),
            performances=[ModelPerformanceRecord('xgb', 0.9, 'auc')],
            best_model='xgb', best_score=0.9
        ))
        
        query = DatasetFingerprint(n_samples=100, task_type='regression')
        similar = kb.find_similar(query, k=5, task_type_filter='regression')
        self.assertEqual(len(similar), 0)
    
    def test_empty_kb(self):
        """测试空知识库"""
        kb = MetaKnowledgeBase()
        fp = DatasetFingerprint(n_samples=100, task_type='classification')
        similar = kb.find_similar(fp, k=5)
        self.assertEqual(len(similar), 0)
    
    def test_persistence(self):
        """测试持久化"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            kb = MetaKnowledgeBase(disk_path=path)
            fp = DatasetFingerprint(n_samples=100, task_type='classification')
            kb.add(KnowledgeRecord(
                fingerprint=fp,
                performances=[ModelPerformanceRecord('lr', 0.85, 'auc')],
                best_model='lr', best_score=0.85
            ))
            kb.save()
            
            kb2 = MetaKnowledgeBase(disk_path=path)
            self.assertEqual(len(kb2), 1)
        finally:
            os.unlink(path)


class TestMetaLearningModelRecommender(unittest.TestCase):
    """测试元学习推荐器"""
    
    def test_empty_kb_fallback(self):
        """空知识库时应回退到启发式"""
        rec = MetaLearningModelRecommender()
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        result = rec.recommend(X, y, TaskType.CLASSIFICATION)
        self.assertIn('model_keys', result)
        self.assertGreater(len(result['model_keys']), 0)
        self.assertEqual(result['source'], 'heuristic')
    
    def test_feedback_and_recommend(self):
        """测试反馈和推荐"""
        rec = MetaLearningModelRecommender()
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # 先反馈一些历史数据
        from core.meta_feature_extractor import MetaFeatureExtractor
        meta = MetaFeatureExtractor().extract(X, y, TaskType.CLASSIFICATION)
        fp = DatasetFingerprint.from_meta_features(meta, X, TaskType.CLASSIFICATION)
        
        rec.feedback(fp, {
            'xgb': {'score': 0.92, 'metric': 'auc'},
            'lr': {'score': 0.85, 'metric': 'auc'},
        })
        
        # 再次推荐
        result = rec.recommend(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(result['source'], 'meta_learning')
        self.assertIn('xgb', result['model_keys'])
    
    def test_feature_type_adjustment(self):
        """测试特征类型感知调整"""
        rec = MetaLearningModelRecommender()
        fp = DatasetFingerprint(
            n_samples=100, n_features=10,
            text_ratio=0.5,  # 高文本比例
            task_type='classification'
        )
        base_models = ['xgb', 'lr', 'rf']
        adjusted = rec._adjust_by_feature_type(fp, base_models)
        self.assertEqual(adjusted[0], 'lr')  # 线性模型应优先
    
    def test_hybrid_recommendation(self):
        """测试混合推荐"""
        rec = MetaLearningModelRecommender(min_similarity=0.99)  # 高阈值强制混合
        kb = MetaKnowledgeBase()
        kb.add(KnowledgeRecord(
            fingerprint=DatasetFingerprint(n_samples=100, task_type='classification'),
            performances=[ModelPerformanceRecord('xgb', 0.9, 'auc')],
            best_model='xgb', best_score=0.9
        ))
        rec.kb = kb
        
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        result = rec.recommend(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(result['source'], 'hybrid')


class TestCosineSimilarity(unittest.TestCase):
    """测试余弦相似度"""
    
    def test_identical_vectors(self):
        """相同向量相似度为1"""
        a = np.array([1.0, 2.0, 3.0])
        sim = MetaKnowledgeBase._cosine_similarity(a, a)
        self.assertAlmostEqual(sim, 1.0, places=5)
    
    def test_orthogonal_vectors(self):
        """正交向量相似度为0"""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = MetaKnowledgeBase._cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 0.0, places=5)
    
    def test_zero_vector(self):
        """零向量相似度为0"""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        sim = MetaKnowledgeBase._cosine_similarity(a, b)
        self.assertEqual(sim, 0.0)
    
    def test_weighted_similarity(self):
        """加权相似度"""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        weights = np.array([0.1, 10.0])
        sim = MetaKnowledgeBase._cosine_similarity(a, b, weights)
        self.assertGreater(sim, 0.0)


if __name__ == '__main__':
    unittest.main()
