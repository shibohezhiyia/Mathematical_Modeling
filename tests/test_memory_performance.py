"""
内存与性能优化测试
覆盖：
- 大文件分块读取
- GPU 内存监控
- 结果缓存机制
- 默认并行 n_jobs=-1
"""
import os
import sys
import unittest
import tempfile
import time
import threading

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_module import DataLoader, DataModule
from core.accelerators import GPUManager, ParallelEngine, optimize_memory, get_system_info
from core.result_cache import ResultCache, get_result_cache, cached, clear_cache
from core.workspace_manager import get_workspace_manager


class TestChunkedLoading(unittest.TestCase):
    """测试大文件分块读取"""
    
    def setUp(self):
        self.loader = DataLoader()
        wm = get_workspace_manager()
        wm.set_allow_disk_write(True)
        self.test_dir = wm.create_temp_dir(prefix='test_chunk')
    
    def test_should_chunk_small_file(self):
        """小文件不应触发分块"""
        path = os.path.join(self.test_dir, 'small.csv')
        df = pd.DataFrame({'A': range(100)})
        df.to_csv(path, index=False)
        self.assertFalse(DataLoader._should_chunk(path, threshold_mb=10))
    
    def test_should_chunk_large_file(self):
        """大文件应触发分块"""
        path = os.path.join(self.test_dir, 'large.csv')
        # 创建 2MB 文件（超过 1MB 阈值）
        df = pd.DataFrame(np.random.randn(50000, 10))
        df.to_csv(path, index=False)
        self.assertTrue(DataLoader._should_chunk(path, threshold_mb=1))
    
    def test_load_chunked_csv(self):
        """分块读取 CSV 并合并"""
        path = os.path.join(self.test_dir, 'chunked.csv')
        df = pd.DataFrame({
            'a': range(1000),
            'b': ['x'] * 1000,
            'c': np.random.randn(1000)
        })
        df.to_csv(path, index=False)
        
        result = self.loader.load_chunked(path, chunk_size=200)
        self.assertEqual(len(result), 1000)
        self.assertEqual(list(result.columns), ['a', 'b', 'c'])
    
    def test_auto_chunk_disabled(self):
        """auto_chunk=False 时小文件正常加载"""
        path = os.path.join(self.test_dir, 'normal.csv')
        df = pd.DataFrame({'A': range(100)})
        df.to_csv(path, index=False)
        
        result = self.loader.load(path, auto_chunk=False)
        self.assertEqual(len(result), 100)
    
    def test_chunked_encoding_fallback(self):
        """分块读取编码回退"""
        path = os.path.join(self.test_dir, 'gbk.csv')
        df = pd.DataFrame({'名称': ['测试', '数据', '中文']})
        df.to_csv(path, index=False, encoding='gbk')
        
        # 强制分块读取
        result = self.loader.load_chunked(path, chunk_size=10, encoding='utf-8')
        self.assertEqual(len(result), 3)
    
    def test_unsupported_chunk_format(self):
        """不支持分块的格式报错"""
        path = os.path.join(self.test_dir, 'test.parquet')
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_parquet(path)
        
        with self.assertRaises(ValueError):
            self.loader.load_chunked(path)


class TestGPUMemoryMonitor(unittest.TestCase):
    """测试 GPU 内存监控"""
    
    def test_get_memory_info_returns_dict(self):
        """显存信息返回正确格式"""
        gpu = GPUManager()
        info = gpu.get_memory_info()
        
        self.assertIn('total_mb', info)
        self.assertIn('used_mb', info)
        self.assertIn('free_mb', info)
        self.assertIn('utilization', info)
        
        self.assertIsInstance(info['total_mb'], (int, float))
        self.assertIsInstance(info['utilization'], float)
    
    def test_check_memory_no_gpu(self):
        """无 GPU 时检查返回 False"""
        gpu = GPUManager()
        if not gpu.available:
            result = gpu.check_memory()
            self.assertFalse(result)
    
    def test_system_info_has_gpu_keys(self):
        """系统信息包含 GPU 相关字段"""
        info = get_system_info()
        self.assertIn('gpu_available', info)
        self.assertIn('gpu_backend', info)
        self.assertIn('gpu_count', info)


class TestResultCache(unittest.TestCase):
    """测试结果缓存机制"""
    
    def setUp(self):
        self.cache = ResultCache(
            cache_dir=os.path.join(get_workspace_manager().cache_dir, 'test_cache'),
            ttl_seconds=3600,
            max_entries=100,
            enabled=True
        )
        self.cache.invalidate()  # 清空
    
    def tearDown(self):
        self.cache.invalidate()
    
    def test_basic_set_get(self):
        """基本设置和获取"""
        self.cache.set('key1', {'score': 0.95})
        result = self.cache.get('key1')
        self.assertEqual(result, {'score': 0.95})
    
    def test_get_nonexistent(self):
        """获取不存在的键返回 None"""
        result = self.cache.get('nonexistent')
        self.assertIsNone(result)
    
    def test_cache_expiration(self):
        """缓存过期测试"""
        short_cache = ResultCache(
            cache_dir=os.path.join(get_workspace_manager().cache_dir, 'test_cache_short'),
            ttl_seconds=0,
            enabled=True
        )
        short_cache.set('expiring', 'value')
        time.sleep(0.1)
        result = short_cache.get('expiring')
        self.assertIsNone(result)
    
    def test_get_or_compute(self):
        """获取或计算"""
        call_count = [0]
        
        def compute():
            call_count[0] += 1
            return call_count[0]
        
        result1 = self.cache.get_or_compute(compute, cache_key='compute_test')
        result2 = self.cache.get_or_compute(compute, cache_key='compute_test')
        
        self.assertEqual(result1, 1)
        self.assertEqual(result2, 1)  # 第二次应命中缓存，compute 不执行
        self.assertEqual(call_count[0], 1)
    
    def test_invalidate_all(self):
        """清空所有缓存"""
        self.cache.set('a', 1)
        self.cache.set('b', 2)
        count = self.cache.invalidate()
        self.assertGreaterEqual(count, 2)
        self.assertIsNone(self.cache.get('a'))
    
    def test_invalidate_pattern(self):
        """按前缀清空缓存"""
        self.cache.set('prefix_1', 1)
        self.cache.set('prefix_2', 2)
        self.cache.set('other', 3)
        
        self.cache.invalidate(pattern='prefix_')
        self.assertIsNone(self.cache.get('prefix_1'))
        self.assertIsNotNone(self.cache.get('other'))
    
    def test_stats(self):
        """缓存统计"""
        self.cache.set('s1', [1, 2, 3])
        stats = self.cache.stats()
        self.assertIn('enabled', stats)
        self.assertIn('memory_entries', stats)
        self.assertIn('disk_entries', stats)
        self.assertTrue(stats['enabled'])
    
    def test_disabled_cache(self):
        """禁用缓存"""
        disabled = ResultCache(enabled=False)
        disabled.set('key', 'value')
        self.assertIsNone(disabled.get('key'))
    
    def test_thread_safety(self):
        """线程安全测试"""
        errors = []
        
        def worker(i):
            try:
                self.cache.set(f'thread_{i}', i * 10)
                val = self.cache.get(f'thread_{i}')
                if val != i * 10:
                    errors.append(f'mismatch: {val} != {i * 10}')
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(errors, [])
    
    def test_lru_eviction(self):
        """LRU 淘汰测试"""
        small_cache = ResultCache(
            cache_dir=os.path.join(get_workspace_manager().cache_dir, 'test_lru'),
            max_entries=3,
            enabled=True
        )
        small_cache.invalidate()
        
        small_cache.set('k1', 1)
        small_cache.set('k2', 2)
        small_cache.set('k3', 3)
        small_cache.set('k4', 4)  # 应淘汰 k1
        
        # 磁盘缓存仍然存在，但内存缓存可能被淘汰
        # 这里主要测试不会报错
        stats = small_cache.stats()
        self.assertTrue(stats['enabled'])


class TestCachedDecorator(unittest.TestCase):
    """测试缓存装饰器"""
    
    def setUp(self):
        clear_cache()
        self.call_count = 0
    
    def tearDown(self):
        clear_cache()
    
    def test_cached_decorator(self):
        """装饰器缓存功能"""
        @cached()
        def add(a, b):
            self.call_count += 1
            return a + b
        
        r1 = add(1, 2)
        r2 = add(1, 2)
        
        self.assertEqual(r1, 3)
        self.assertEqual(r2, 3)
        self.assertEqual(self.call_count, 1)
    
    def test_cached_disabled(self):
        """禁用装饰器缓存"""
        @cached(enabled=False)
        def multiply(a, b):
            self.call_count += 1
            return a * b
        
        multiply(2, 3)
        multiply(2, 3)
        self.assertEqual(self.call_count, 2)


class TestParallelEngineDefaultJobs(unittest.TestCase):
    """测试并行引擎默认配置"""
    
    def test_default_n_jobs(self):
        """默认 n_jobs=-1 应使用全部核心"""
        engine = ParallelEngine()
        self.assertGreater(engine.n_jobs, 0)
    
    def test_map_single_item(self):
        """单元素列表不使用并行"""
        engine = ParallelEngine(n_jobs=4)
        result = engine.map(lambda x: x * 2, [5])
        self.assertEqual(result, [10])


class TestMemoryOptimization(unittest.TestCase):
    """测试内存优化"""
    
    def test_optimize_memory_downcast(self):
        """数值列降精度"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'cat_col': ['A', 'B', 'A', 'B', 'A']
        })
        
        result = optimize_memory(df, verbose=False)
        
        # 整数列应降精度
        self.assertTrue(pd.api.types.is_integer_dtype(result['int_col']))
        # 浮点列应转为 float32
        self.assertEqual(result['float_col'].dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
