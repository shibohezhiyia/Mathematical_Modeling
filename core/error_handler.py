"""
错误处理与恢复模块

提供统一的异常体系、安全执行上下文、用户友好错误格式化
以及自动重试/回退机制。
"""
import time
import functools
import traceback
from typing import Any, Callable, Optional, Type, Dict, List, Union, Tuple
from contextlib import contextmanager

from utils.helpers import log_info, log_warning, log_error


# ---------------------------------------------------------------------------
# 1. 自定义异常体系
# ---------------------------------------------------------------------------

class AutoMLException(Exception):
    """AutoML 基础异常"""
    error_code: str = "E0000"
    default_message: str = "未知错误"

    def __init__(self, message: Optional[str] = None, *, details: Optional[Dict[str, Any]] = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class DataLoadError(AutoMLException):
    """数据加载失败"""
    error_code = "E1001"
    default_message = "数据加载失败"


class ModelTrainingError(AutoMLException):
    """模型训练失败"""
    error_code = "E2001"
    default_message = "模型训练失败"


class OptimizationError(AutoMLException):
    """超参优化失败"""
    error_code = "E3001"
    default_message = "超参优化失败"


class ConfigurationError(AutoMLException):
    """配置错误"""
    error_code = "E4001"
    default_message = "配置错误"


class ResourceError(AutoMLException):
    """资源不足（内存/显存）"""
    error_code = "E5001"
    default_message = "系统资源不足"


class ValidationError(AutoMLException):
    """输入验证失败"""
    error_code = "E6001"
    default_message = "输入验证失败"


# ---------------------------------------------------------------------------
# 2. ErrorRecoveryContext 上下文管理器
# ---------------------------------------------------------------------------

class ErrorRecoveryContext:
    """
    错误恢复上下文管理器。

    用法::
        with ErrorRecoveryContext(fallback_value=default_params):
            result = risky_operation()
    """

    def __init__(
        self,
        fallback_value: Any = None,
        reraise: bool = False,
        log_category: str = "错误恢复",
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.fallback_value = fallback_value
        self.reraise = reraise
        self.log_category = log_category
        self.on_error = on_error
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.success: bool = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error = exc_val
            self.success = False
            msg = f"操作失败: {exc_val}"
            log_error(msg, category=self.log_category)
            if self.on_error:
                try:
                    self.on_error(exc_val)
                except Exception as cb_err:
                    log_warning(f"错误回调执行失败: {cb_err}", category=self.log_category)
            self.result = self.fallback_value
            log_info(f"已回退到 fallback_value", category=self.log_category)
            return not self.reraise
        self.success = True
        return False


# ---------------------------------------------------------------------------
# 3. FriendlyErrorFormatter 用户友好格式化器
# ---------------------------------------------------------------------------

class FriendlyErrorFormatter:
    """
    将技术异常转换为用户友好的中文消息，并提供恢复建议。
    """

    # 内置错误映射: error_code -> (中文消息, 恢复建议)
    _BUILT_IN: Dict[str, tuple] = {
        "E1001": ("数据文件无法读取或格式不正确", "请检查文件路径、编码格式及文件完整性。"),
        "E2001": ("模型训练过程中发生异常", "请检查数据质量、特征维度及超参数设置是否合理。"),
        "E3001": ("超参数搜索或优化失败", "请缩小搜索空间、减少迭代次数或检查目标函数。"),
        "E4001": ("配置文件或参数设置有误", "请核对配置文件格式、必填项及参数类型。"),
        "E5001": ("系统内存/显存不足", "请减少数据批量大小、降低模型复杂度或释放系统资源。"),
        "E6001": ("输入数据未通过验证", "请检查数据类型、缺失值及数值范围是否符合要求。"),
        "E0000": ("发生未知错误", "请查看详细日志或联系技术支持。"),
    }

    # 异常类型 -> error_code 的兜底映射
    _TYPE_MAP: Dict[Type[Exception], str] = {
        MemoryError: "E5001",
        OSError: "E1001",
        ValueError: "E6001",
        TypeError: "E6001",
        KeyError: "E4001",
        ImportError: "E4001",
    }

    @classmethod
    def format(
        cls,
        exc: Exception,
        user_context: str = "",
    ) -> Dict[str, Any]:
        """
        格式化异常为结构化用户友好消息。

        Returns:
            {
                "error_code": str,
                "title": str,
                "message": str,
                "suggestion": str,
                "detail": str,
                "context": str,
            }
        """
        error_code = getattr(exc, "error_code", None)
        if error_code is None:
            for exc_type, code in cls._TYPE_MAP.items():
                if isinstance(exc, exc_type):
                    error_code = code
                    break
            else:
                error_code = "E0000"

        title, suggestion = cls._BUILT_IN.get(error_code, cls._BUILT_IN["E0000"])
        detail = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()

        return {
            "error_code": error_code,
            "title": title,
            "message": str(exc) if str(exc) else title,
            "suggestion": suggestion,
            "detail": detail,
            "traceback": trace,
            "context": user_context,
        }

    @classmethod
    def to_string(cls, exc: Exception, user_context: str = "") -> str:
        """格式化为单条可读字符串。"""
        info = cls.format(exc, user_context)
        lines = [
            f"[{info['error_code']}] {info['title']}",
            f"消息: {info['message']}",
            f"建议: {info['suggestion']}",
        ]
        if info["context"]:
            lines.append(f"上下文: {info['context']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. SafeExecutor 安全执行器
# ---------------------------------------------------------------------------

class SafeExecutor:
    """
    安全执行器：自动重试 + 回退 + 详细日志。
    """

    @staticmethod
    def run(
        func: Callable,
        *args,
        fallback: Any = None,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        log_category: str = "SafeExecutor",
        **kwargs,
    ) -> Any:
        """
        安全执行函数，失败时自动重试，最终失败返回 fallback。

        Args:
            func: 要执行的函数
            *args: 位置参数
            fallback: 失败后的回退值
            max_retries: 最大重试次数（不含第一次）
            retry_delay: 重试间隔（秒）
            retry_exceptions: 允许重试的异常类型，默认所有 Exception
            log_category: 日志分类标签
            **kwargs: 关键字参数
        """
        if retry_exceptions is None:
            retry_exceptions = (Exception,)

        last_exception: Optional[Exception] = None
        attempt = 0
        total_attempts = max_retries + 1

        while attempt < total_attempts:
            attempt += 1
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    log_info(
                        f"函数 {func.__name__} 在第 {attempt} 次重试后成功",
                        category=log_category,
                    )
                return result
            except Exception as exc:
                last_exception = exc
                if isinstance(exc, retry_exceptions):
                    if attempt < total_attempts:
                        log_warning(
                            f"函数 {func.__name__} 第 {attempt} 次执行失败: {exc}，"
                            f"{retry_delay}s 后进行第 {attempt + 1} 次重试",
                            category=log_category,
                        )
                        time.sleep(retry_delay)
                    else:
                        log_error(
                            f"函数 {func.__name__} 在 {total_attempts} 次尝试后仍失败: {exc}",
                            category=log_category,
                        )
                else:
                    # 非允许重试的异常，直接返回 fallback
                    log_error(
                        f"函数 {func.__name__} 遇到不可重试的异常: {exc}",
                        category=log_category,
                    )
                    break

        # 所有尝试均失败，返回 fallback
        log_info(
            f"函数 {func.__name__} 已回退到 fallback={fallback!r}",
            category=log_category,
        )
        return fallback


# ---------------------------------------------------------------------------
# 5. 全局异常装饰器
# ---------------------------------------------------------------------------

def handle_errors(
    fallback: Any = None,
    user_message: str = "",
    log_category: str = "异常装饰器",
    reraise: bool = False,
    max_retries: int = 0,
    retry_delay: float = 0.5,
    retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    全局异常处理装饰器。

    Args:
        fallback: 异常时返回值
        user_message: 附加用户消息上下文
        log_category: 日志分类
        reraise: 是否重新抛出异常
        max_retries: 最大重试次数（不含首次）
        retry_delay: 重试间隔
        retry_exceptions: 允许重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _execute():
                return func(*args, **kwargs)

            if max_retries > 0:
                result = SafeExecutor.run(
                    _execute,
                    fallback=fallback,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_exceptions=retry_exceptions,
                    log_category=log_category,
                )
            else:
                try:
                    result = _execute()
                except Exception as exc:
                    friendly = FriendlyErrorFormatter.to_string(exc, user_context=user_message)
                    log_error(f"{func.__name__} 执行异常\n{friendly}", category=log_category)
                    if reraise:
                        raise
                    result = fallback

            return result
        return wrapper
    return decorator
