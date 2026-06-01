"""
通用辅助函数
"""
import time
import logging
from functools import wraps
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class LogStore:
    """
    内存日志存储器（环形缓冲区，最大保留 5000 条）
    
    支持按级别分类：INFO / WARNING / ERROR
    支持按模块/类别标签分类
    """
    MAX_SIZE = 5000
    
    def __init__(self):
        self._logs: deque = deque(maxlen=self.MAX_SIZE)
        self._counters = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
    
    def add(self, level: str, message: str, category: str = ""):
        """添加一条日志"""
        entry = {
            "id": self._counters.get(level, 0) + len(self._logs) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level.upper(),
            "category": category or "系统",
            "message": message,
        }
        self._logs.append(entry)
        self._counters[level.upper()] = self._counters.get(level.upper(), 0) + 1
        
        # 同时输出到标准日志
        if level.upper() == "INFO":
            logging.info(message)
        elif level.upper() == "WARNING":
            logging.warning(message)
        elif level.upper() == "ERROR":
            logging.error(message)
        else:
            logging.info(message)
    
    def get_logs(self, level: Optional[str] = None, limit: int = 200, offset: int = 0) -> List[Dict]:
        """获取日志列表，支持级别筛选"""
        logs = list(self._logs)
        if level and level.upper() != "ALL":
            logs = [l for l in logs if l["level"] == level.upper()]
        # 倒序返回（最新的在前）
        logs = logs[::-1]
        total = len(logs)
        return logs[offset:offset + limit], total
    
    def get_stats(self) -> Dict:
        """获取日志统计"""
        return {
            "total": len(self._logs),
            "debug": sum(1 for l in self._logs if l["level"] == "DEBUG"),
            "info": sum(1 for l in self._logs if l["level"] == "INFO"),
            "warning": sum(1 for l in self._logs if l["level"] == "WARNING"),
            "error": sum(1 for l in self._logs if l["level"] == "ERROR"),
            "critical": sum(1 for l in self._logs if l["level"] == "CRITICAL"),
        }
    
    def clear(self):
        """清空日志"""
        self._logs.clear()
        self._counters = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}


# 全局日志存储实例
_log_store = LogStore()


def get_log_store() -> LogStore:
    """获取全局日志存储实例"""
    return _log_store


def log_info(msg: str, category: str = ""):
    """记录信息日志"""
    _log_store.add("INFO", msg, category)


def log_warning(msg: str, category: str = ""):
    """记录警告日志"""
    _log_store.add("WARNING", msg, category)


def log_error(msg: str, category: str = ""):
    """记录错误日志"""
    _log_store.add("ERROR", msg, category)


def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        log_info(f"[{func.__name__}] 执行耗时: {elapsed:.3f}s", category="性能")
        return result
    return wrapper
