"""
数据模块单元测试
"""
import os
import sys
import unittest
from core.workspace_manager import get_workspace_manager

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_module import (
    DataModule, DataLoader, TypeDetector, DataCleaner,
    DataType, ColumnProfile
)


class TestDataLoader(unittest.TestCase):
    """测试数据加载器"""
    
    def setUp(self):
        self.loader = DataLoader()
        wm = get_workspace_manager()
        self.test_dir = wm.create_temp_dir(prefix='test_data')
    
    def tearDown(self):
        # WorkspaceManager 自动清理 temp
        pass
    
    def _create_test_csv(self, filename, data):
        path = os.path.join(self.test_dir, filename)
        data.to_csv(path, index=False, encoding='utf-8')
        return path
    
    def test_load_csv(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        path = self._create_test_csv('test.csv', df)
        loaded = self.loader.load(path)
        self.assertEqual(loaded.shape, (3, 2))
    
    def test_load_excel(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        path = os.path.join(self.test_dir, 'test.xlsx')
        df.to_excel(path, index=False)
        loaded = self.loader.load(path)
        self.assertEqual(loaded.shape, (3, 2))
    
    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            self.loader.load('test.xyz')


class TestTypeDetector(unittest.TestCase):
    """测试类型检测器"""
    
    def setUp(self):
        self.detector = TypeDetector()
    
    def test_numeric_detection(self):
        s = pd.Series([1, 2, 3, 4, 5])
        dtype, profile = self.detector.detect(s, 'num_col')
        self.assertEqual(dtype, DataType.NUMERIC)
        self.assertIn('mean', profile.stats)
    
    def test_category_detection(self):
        s = pd.Series(['A', 'B', 'A', 'C', 'B'] * 20)  # 100个样本，提高类别型置信度
        dtype, profile = self.detector.detect(s, 'cat_col')
        self.assertEqual(dtype, DataType.CATEGORY)
    
    def test_boolean_detection(self):
        s = pd.Series([0, 1, 0, 1, 1])
        dtype, profile = self.detector.detect(s, 'bool_col')
        self.assertEqual(dtype, DataType.BOOLEAN)
    
    def test_datetime_detection(self):
        s = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        dtype, profile = self.detector.detect(s, 'date_col')
        self.assertEqual(dtype, DataType.DATETIME)
    
    def test_id_detection(self):
        s = pd.Series(range(100))
        dtype, profile = self.detector.detect(s, 'user_id')
        self.assertEqual(dtype, DataType.ID)
    
    def test_empty_detection(self):
        s = pd.Series([np.nan] * 10)
        dtype, profile = self.detector.detect(s, 'empty_col')
        self.assertEqual(dtype, DataType.EMPTY)
    
    def test_constant_detection(self):
        s = pd.Series([5] * 10)
        dtype, profile = self.detector.detect(s, 'const_col')
        self.assertEqual(dtype, DataType.CONSTANT)
    
    def test_numeric_with_comma(self):
        s = pd.Series(['1,000', '2,000', '3,000'])
        dtype, profile = self.detector.detect(s, 'comma_num')
        self.assertEqual(dtype, DataType.NUMERIC)


class TestDataCleaner(unittest.TestCase):
    """测试数据清洗器"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
    
    def test_drop_empty_column(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan],
            'C': ['x', 'y', 'z']
        })
        detector = TypeDetector()
        profiles = detector.analyze_dataframe(df)
        cleaned = self.cleaner.clean(df, profiles)
        self.assertNotIn('B', cleaned.columns)
    
    def test_fill_numeric_missing(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0]
        })
        detector = TypeDetector()
        profiles = detector.analyze_dataframe(df)
        cleaned = self.cleaner.clean(df, profiles)
        self.assertEqual(cleaned['A'].isnull().sum(), 0)
        self.assertEqual(cleaned['A'].iloc[2], 2.0)  # 中位数
    
    def test_clip_outliers(self):
        df = pd.DataFrame({
            'A': list(range(1, 21)) + [999]  # 999 是明显异常值
        })
        detector = TypeDetector()
        profiles = detector.analyze_dataframe(df)
        cleaned = self.cleaner.clean(df, profiles)
        # IQR: Q1=5.75, Q3=15.25, IQR=9.5, upper=15.25+1.5*9.5=29.5 -> clip到29.5左右
        self.assertLess(cleaned['A'].max(), 100)  # 999应该被截断为小于100的值


class TestDataModule(unittest.TestCase):
    """测试数据模块整体流程"""
    
    def setUp(self):
        self.module = DataModule()
        wm = get_workspace_manager()
        self.test_dir = wm.create_temp_dir(prefix='test_data')
    
    def tearDown(self):
        pass
    
    def test_full_pipeline(self):
        # 创建测试数据
        df = pd.DataFrame({
            'id': range(100),
            'age': np.random.randint(18, 80, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'income': np.random.normal(5000, 2000, 100),
            'register_date': pd.date_range('2020-01-01', periods=100),
            'empty_col': [np.nan] * 100,
            'target': np.random.randint(0, 2, 100)
        })
        # 添加一些缺失值
        df.loc[::10, 'age'] = np.nan
        df.loc[::15, 'income'] = np.nan
        
        path = os.path.join(self.test_dir, 'test_data.csv')
        df.to_csv(path, index=False)
        
        # 执行完整流程
        self.module.load(path).analyze().clean(target_col='target')
        
        # 验证
        self.assertIsNotNone(self.module.raw_data)
        self.assertIsNotNone(self.module.cleaned_data)
        self.assertIsNotNone(self.module.profiles)
        self.assertEqual(self.module.cleaned_data['target'].isnull().sum(), 0)
        self.assertNotIn('empty_col', self.module.cleaned_data.columns)
    
    def test_summary(self):
        df = pd.DataFrame({
            'A': [1, 2, 3] * 40,
            'B': ['x', 'y', 'z'] * 40  # 120样本，3唯一值，unique_rate=0.025 < 0.05
        })
        path = os.path.join(self.test_dir, 'summary_test.csv')
        df.to_csv(path, index=False)
        
        self.module.load(path).analyze()
        summary = self.module.get_summary()
        
        self.assertEqual(summary['total_columns'], 2)
        self.assertEqual(summary['total_rows'], 120)
        self.assertIn('数值型', summary['type_distribution'])
        self.assertIn('类别型', summary['type_distribution'])


if __name__ == '__main__':
    unittest.main()
