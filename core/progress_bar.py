"""
进度条封装模块

提供统一的进度条接口，tqdm 未安装时自动回退到无进度条模式。
支持通过 verbose=False 禁用。
"""
import os
from typing import Optional, Iterable, Any

# 尝试导入 tqdm
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class DummyTqdm:
    """tqdm 不可用时或 verbose=False 时的占位符"""
    
    def __init__(self, iterable=None, total=None, desc=None, **kwargs):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.n = 0
    
    def __iter__(self):
        if self.iterable is not None:
            for item in self.iterable:
                self.n += 1
                yield item
        else:
            for i in range(self.total or 0):
                self.n = i + 1
                yield i
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n=1):
        self.n += n
    
    def set_description(self, desc):
        self.desc = desc
    
    def set_postfix(self, **kwargs):
        pass
    
    def close(self):
        pass


def get_progress_bar(iterable=None, total=None, desc=None,
                     disable=False, **kwargs):
    """
    获取进度条实例
    
    Args:
        iterable: 可迭代对象
        total: 总数
        desc: 描述文字
        disable: 是否禁用进度条
        **kwargs: 传给 tqdm 的额外参数
    
    Returns:
        tqdm 或 DummyTqdm 实例
    """
    # 环境变量可全局禁用
    if os.environ.get('DISABLE_TQDM', '').lower() in ('1', 'true', 'yes'):
        disable = True
    
    if disable or not _TQDM_AVAILABLE:
        return DummyTqdm(iterable=iterable, total=total, desc=desc, **kwargs)
    
    return _tqdm(iterable=iterable, total=total, desc=desc, **kwargs)


def progress_iter(iterable, desc=None, disable=False, **kwargs):
    """迭代器包装进度条"""
    return get_progress_bar(iterable=iterable, desc=desc, disable=disable, **kwargs)


def progress_range(total, desc=None, disable=False, **kwargs):
    """range 包装进度条"""
    return get_progress_bar(iterable=range(total), total=total, desc=desc, disable=disable, **kwargs)
