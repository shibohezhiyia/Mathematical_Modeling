"""
结果缓存机制：避免重复计算，提升性能

基于文件系统的持久化缓存，使用 MD5 哈希作为缓存键。
支持：
- 模型评估结果缓存（交叉验证分数）
- 数据预处理结果缓存
- EDA/特征分析结果缓存

自动清理过期缓存（LRU + TTL）。
"""

import os
import json
import hashlib
import time
import pickle
import threading
from typing import Any, Dict, Optional, Union, Callable
from pathlib import Path
from functools import wraps

from utils.helpers import log_info, log_warning, log_error


class ResultCache:
    """
    结果缓存管理器
    
    缓存策略：
    - 键：基于输入数据+配置的 MD5 哈希
    - 值：任意可序列化对象（JSON 或 pickle）
    - TTL：默认 7 天
    - 最大条目数：默认 1000（LRU 淘汰）
    """
    
    def __init__(self,
                 cache_dir: Optional[str] = None,
                 ttl_seconds: int = 604800,  # 7天
                 max_entries: int = 1000,
                 enabled: bool = True) -> None:
        """
        Args:
            cache_dir: 缓存目录，默认使用 workspace/cache/result_cache
            ttl_seconds: 缓存过期时间
            max_entries: 最大缓存条目数
            enabled: 是否启用缓存
        """
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._memory_cache: Dict[str, Any] = {}
        self._memory_meta: Dict[str, Dict] = {}  # 元数据：时间戳、访问次数
        
        if cache_dir is None:
            # 延迟导入避免循环依赖
            try:
                from core.workspace_manager import get_workspace_manager
                wm = get_workspace_manager()
                cache_dir = os.path.join(wm.cache_dir, 'result_cache')
            except Exception:
                cache_dir = os.path.join(os.getcwd(), 'workspace', 'cache', 'result_cache')
        
        self.cache_dir = cache_dir
        if enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._cleanup_expired()
    
    def _make_key(self, *args, **kwargs) -> str:
        """基于输入生成 MD5 缓存键"""
        # 使用 pickle 序列化后取哈希，支持 numpy/pandas 对象
        try:
            key_data = pickle.dumps((args, sorted(kwargs.items())), protocol=4)
        except (TypeError, pickle.PicklingError):
            # 回退：使用字符串 repr
            key_data = repr((args, sorted(kwargs.items()))).encode('utf-8')
        return hashlib.md5(key_data).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 分两层目录避免单个目录文件过多
        return os.path.join(self.cache_dir, key[:2], f"{key}.json")
    
    def _get_meta_path(self, key: str) -> str:
        """获取元数据文件路径"""
        return os.path.join(self.cache_dir, key[:2], f"{key}.meta.json")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，不存在或过期返回 None"""
        if not self.enabled:
            return None
        
        with self._lock:
            # 1. 检查内存缓存
            if key in self._memory_cache:
                meta = self._memory_meta.get(key, {})
                if time.time() - meta.get('timestamp', 0) < self.ttl_seconds:
                    meta['access_count'] = meta.get('access_count', 0) + 1
                    return self._memory_cache[key]
                else:
                    # 内存缓存过期
                    del self._memory_cache[key]
                    del self._memory_meta[key]
            
            # 2. 检查磁盘缓存
            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)
            
            if not os.path.exists(cache_path):
                return None
            
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if time.time() - meta.get('timestamp', 0) > self.ttl_seconds:
                    # 磁盘缓存过期
                    self._remove_files(key)
                    return None
                
                # 加载数据
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 回写内存缓存
                self._memory_cache[key] = data
                self._memory_meta[key] = {
                    'timestamp': meta['timestamp'],
                    'access_count': meta.get('access_count', 0) + 1
                }
                
                return data
            
            except Exception as e:
                log_warning(f"[ResultCache] 读取缓存失败: {e}")
                self._remove_files(key)
                return None
    
    def set(self, key: str, value: Any) -> bool:
        """设置缓存值"""
        if not self.enabled:
            return False
        
        with self._lock:
            # 写入内存缓存
            self._memory_cache[key] = value
            self._memory_meta[key] = {
                'timestamp': time.time(),
                'access_count': 0
            }
            
            # 写入磁盘缓存
            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)
            
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, default=str)
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': time.time(),
                        'access_count': 0,
                        'size_bytes': os.path.getsize(cache_path)
                    }, f)
                
                # 检查是否需要清理
                self._enforce_lru()
                return True
            
            except Exception as e:
                log_warning(f"[ResultCache] 写入缓存失败: {e}")
                return False
    
    def get_or_compute(self,
                       compute_fn: Callable,
                       *args,
                       cache_key: Optional[str] = None,
                       **kwargs) -> Any:
        """
        获取缓存值，不存在则计算并缓存
        
        Args:
            compute_fn: 计算函数
            *args, **kwargs: 传给 compute_fn 的参数
            cache_key: 自定义缓存键（默认自动生成）
        
        Returns:
            缓存值或计算结果
        """
        key = cache_key or self._make_key(compute_fn.__name__, *args, **kwargs)
        
        cached = self.get(key)
        if cached is not None:
            return cached
        
        result = compute_fn(*args, **kwargs)
        self.set(key, result)
        return result
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        使缓存失效
        
        Args:
            pattern: 匹配键的模式（前缀匹配），None 表示全部清除
        
        Returns:
            清除的条目数
        """
        with self._lock:
            count = 0
            
            # 清除内存缓存
            if pattern is None:
                count += len(self._memory_cache)
                self._memory_cache.clear()
                self._memory_meta.clear()
            else:
                keys_to_remove = [k for k in self._memory_cache if k.startswith(pattern)]
                for k in keys_to_remove:
                    del self._memory_cache[k]
                    del self._memory_meta[k]
                count += len(keys_to_remove)
            
            # 清除磁盘缓存
            if os.path.exists(self.cache_dir):
                for root, dirs, files in os.walk(self.cache_dir):
                    for f in files:
                        if f.endswith('.json'):
                            key = f.replace('.json', '').replace('.meta', '')
                            if pattern is None or key.startswith(pattern):
                                try:
                                    os.remove(os.path.join(root, f))
                                    count += 1
                                except Exception:
                                    pass
            
            log_info(f"[ResultCache] 清除 {count} 个缓存条目")
            return count
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            disk_count = 0
            disk_size = 0
            
            if os.path.exists(self.cache_dir):
                for root, dirs, files in os.walk(self.cache_dir):
                    for f in files:
                        if f.endswith('.json') and not f.endswith('.meta.json'):
                            disk_count += 1
                            fp = os.path.join(root, f)
                            try:
                                disk_size += os.path.getsize(fp)
                            except Exception:
                                pass
            
            return {
                'enabled': self.enabled,
                'memory_entries': len(self._memory_cache),
                'disk_entries': disk_count,
                'disk_size_mb': disk_size / (1024 ** 2),
                'ttl_seconds': self.ttl_seconds,
                'max_entries': self.max_entries,
                'cache_dir': self.cache_dir
            }
    
    def _remove_files(self, key: str) -> None:
        """删除缓存文件"""
        for path in [self._get_cache_path(key), self._get_meta_path(key)]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存"""
        if not os.path.exists(self.cache_dir):
            return
        
        now = time.time()
        removed = 0
        
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.endswith('.meta.json'):
                    meta_path = os.path.join(root, f)
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                        if now - meta.get('timestamp', 0) > self.ttl_seconds:
                            key = f.replace('.meta.json', '')
                            self._remove_files(key)
                            removed += 1
                    except Exception:
                        pass
        
        if removed > 0:
            log_info(f"[ResultCache] 清理 {removed} 个过期缓存")
    
    def _enforce_lru(self) -> None:
        """执行 LRU 淘汰"""
        total = len(self._memory_cache)
        if total <= self.max_entries:
            return
        
        # 按访问次数+时间排序，淘汰最少访问的
        items = [(k, v) for k, v in self._memory_meta.items()]
        items.sort(key=lambda x: (x[1].get('access_count', 0), x[1].get('timestamp', 0)))
        
        to_remove = items[:total - self.max_entries]
        for k, _ in to_remove:
            if k in self._memory_cache:
                del self._memory_cache[k]
                del self._memory_meta[k]


# =============================================================================
# 全局缓存实例与装饰器
# =============================================================================

_result_cache: Optional[ResultCache] = None
_cache_lock = threading.Lock()


def get_result_cache() -> ResultCache:
    """获取全局结果缓存实例"""
    global _result_cache
    if _result_cache is None:
        with _cache_lock:
            if _result_cache is None:
                _result_cache = ResultCache()
    return _result_cache


def cached(ttl_seconds: Optional[int] = None,
           key_fn: Optional[Callable] = None,
           enabled: bool = True) -> Callable[[Callable], Callable]:
    """
    缓存装饰器
    
    用法：
        @cached()
        def expensive_function(a, b):
            return a + b
        
        @cached(ttl_seconds=3600)
        def model_evaluate(model, X, y):
            return cross_val_score(model, X, y)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return func(*args, **kwargs)
            
            cache = get_result_cache()
            
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = cache._make_key(func.__qualname__, *args, **kwargs)
            
            cached_val = cache.get(key)
            if cached_val is not None:
                return cached_val
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator


def clear_cache(pattern: Optional[str] = None) -> int:
    """清除缓存"""
    cache = get_result_cache()
    return cache.invalidate(pattern)
