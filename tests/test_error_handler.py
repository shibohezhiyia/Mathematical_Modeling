"""
错误处理与恢复模块测试
"""
import time
import pytest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.error_handler import (
    AutoMLException,
    DataLoadError,
    ModelTrainingError,
    OptimizationError,
    ConfigurationError,
    ResourceError,
    ValidationError,
    ErrorRecoveryContext,
    FriendlyErrorFormatter,
    SafeExecutor,
    handle_errors,
)


# ---------------------------------------------------------------------------
# 自定义异常测试
# ---------------------------------------------------------------------------

class TestCustomExceptions:
    def test_base_exception_default(self):
        exc = AutoMLException()
        assert exc.error_code == "E0000"
        assert str(exc) == "未知错误"
        assert exc.details == {}

    def test_base_exception_custom_message(self):
        exc = AutoMLException("custom msg", details={"key": "val"})
        assert str(exc) == "custom msg"
        assert exc.details == {"key": "val"}

    def test_data_load_error(self):
        exc = DataLoadError("文件不存在")
        assert exc.error_code == "E1001"
        assert isinstance(exc, AutoMLException)

    def test_model_training_error(self):
        exc = ModelTrainingError("训练 diverged")
        assert exc.error_code == "E2001"
        assert isinstance(exc, AutoMLException)

    def test_optimization_error(self):
        exc = OptimizationError("搜索超时")
        assert exc.error_code == "E3001"

    def test_configuration_error(self):
        exc = ConfigurationError("缺少必填项")
        assert exc.error_code == "E4001"

    def test_resource_error(self):
        exc = ResourceError("OOM")
        assert exc.error_code == "E5001"

    def test_validation_error(self):
        exc = ValidationError("类型不匹配")
        assert exc.error_code == "E6001"


# ---------------------------------------------------------------------------
# ErrorRecoveryContext 测试
# ---------------------------------------------------------------------------

class TestErrorRecoveryContext:
    def test_success_path(self):
        with ErrorRecoveryContext(fallback_value="fallback") as ctx:
            result = 42
        assert ctx.success is True
        assert ctx.error is None

    def test_failure_returns_fallback(self):
        fallback = {"default": True}
        with ErrorRecoveryContext(fallback_value=fallback, log_category="测试") as ctx:
            raise ValueError("boom")
        assert ctx.success is False
        assert ctx.error is not None
        assert ctx.result is fallback

    def test_failure_reraise(self):
        with pytest.raises(ValueError, match="boom"):
            with ErrorRecoveryContext(fallback_value="fb", reraise=True):
                raise ValueError("boom")

    def test_on_error_callback(self):
        called_with = []
        def callback(exc):
            called_with.append(exc)
        with ErrorRecoveryContext(fallback_value=None, on_error=callback) as ctx:
            raise RuntimeError("err")
        assert len(called_with) == 1
        assert isinstance(called_with[0], RuntimeError)

    def test_on_error_callback_exception_swallowed(self):
        def bad_callback(exc):
            raise RuntimeError("callback failed")
        # 回调异常不应影响上下文回退
        with ErrorRecoveryContext(fallback_value="safe", on_error=bad_callback) as ctx:
            raise ValueError("original")
        assert ctx.result == "safe"


# ---------------------------------------------------------------------------
# FriendlyErrorFormatter 测试
# ---------------------------------------------------------------------------

class TestFriendlyErrorFormatter:
    def test_builtin_code(self):
        exc = ResourceError("OOM")
        info = FriendlyErrorFormatter.format(exc)
        assert info["error_code"] == "E5001"
        assert "内存" in info["title"]
        assert "suggestion" in info
        assert "traceback" in info

    def test_type_fallback_mapping(self):
        exc = MemoryError("out of memory")
        info = FriendlyErrorFormatter.format(exc)
        assert info["error_code"] == "E5001"

    def test_unknown_exception(self):
        exc = Exception("something weird")
        info = FriendlyErrorFormatter.format(exc)
        assert info["error_code"] == "E0000"
        assert "未知" in info["title"]

    def test_user_context(self):
        exc = ValueError("bad")
        info = FriendlyErrorFormatter.format(exc, user_context="步骤3")
        assert info["context"] == "步骤3"

    def test_to_string_format(self):
        exc = DataLoadError("file missing")
        text = FriendlyErrorFormatter.to_string(exc, user_context="导入CSV")
        assert "E1001" in text
        assert "导入CSV" in text
        assert "消息:" in text
        assert "建议:" in text


# ---------------------------------------------------------------------------
# SafeExecutor 测试
# ---------------------------------------------------------------------------

class TestSafeExecutor:
    def test_success_no_retry(self):
        def add(a, b):
            return a + b
        result = SafeExecutor.run(add, 1, 2)
        assert result == 3

    def test_fallback_on_failure(self):
        def fail():
            raise RuntimeError("always fail")
        result = SafeExecutor.run(fail, fallback="safe")
        assert result == "safe"

    def test_retry_then_success(self):
        call_count = 0
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("flaky")
            return "ok"
        result = SafeExecutor.run(flaky, max_retries=3, retry_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_retry_exhausted(self):
        call_count = 0
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")
        result = SafeExecutor.run(always_fail, fallback="fb", max_retries=2, retry_delay=0.01)
        assert result == "fb"
        assert call_count == 3  # 1 original + 2 retries

    def test_retry_with_specific_exception(self):
        call_count = 0
        def specific():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("retry me")
            return "done"
        result = SafeExecutor.run(
            specific,
            max_retries=1,
            retry_delay=0.01,
            retry_exceptions=(ValueError,),
        )
        assert result == "done"

    def test_retry_ignores_unconfigured_exception(self):
        def raise_type_error():
            raise TypeError("not retryable")
        result = SafeExecutor.run(
            raise_type_error,
            fallback="fb",
            max_retries=2,
            retry_delay=0.01,
            retry_exceptions=(ValueError,),
        )
        # TypeError 不在 retry_exceptions 中，应直接 fallback
        assert result == "fb"

    def test_kwargs_passed(self):
        def greet(name, greeting="Hello"):
            return f"{greeting} {name}"
        result = SafeExecutor.run(greet, "World", greeting="Hi")
        assert result == "Hi World"


# ---------------------------------------------------------------------------
# handle_errors 装饰器测试
# ---------------------------------------------------------------------------

class TestHandleErrorsDecorator:
    def test_successful_execution(self):
        @handle_errors()
        def normal(x):
            return x * 2
        assert normal(5) == 10

    def test_fallback_on_error(self):
        @handle_errors(fallback="default")
        def broken():
            raise ValueError("oops")
        assert broken() == "default"

    def test_user_message_logged(self):
        @handle_errors(fallback=-1, user_message="计算评分", log_category="测试")
        def broken():
            raise KeyError("missing key")
        assert broken() == -1

    def test_reraise_true(self):
        @handle_errors(reraise=True)
        def broken():
            raise RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            broken()

    def test_retry_then_success(self):
        attempt = 0
        @handle_errors(max_retries=2, retry_delay=0.01)
        def flaky():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise RuntimeError("retry")
            return "ok"
        assert flaky() == "ok"
        assert attempt == 2

    def test_retry_exhausted_fallback(self):
        @handle_errors(fallback="fb", max_retries=1, retry_delay=0.01)
        def always_fail():
            raise RuntimeError("fail")
        assert always_fail() == "fb"

    def test_preserves_function_metadata(self):
        @handle_errors()
        def my_func():
            """docstring"""
            pass
        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "docstring"

    def test_retry_with_specific_exceptions(self):
        call_count = 0
        @handle_errors(
            fallback="fb",
            max_retries=2,
            retry_delay=0.01,
            retry_exceptions=(ValueError,),
        )
        def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")
        assert raise_type_error() == "fb"
        assert call_count == 1  # 不重试

    def test_retry_specific_then_success(self):
        call_count = 0
        @handle_errors(max_retries=2, retry_delay=0.01, retry_exceptions=(ValueError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("retry")
            return "ok"
        assert flaky() == "ok"
        assert call_count == 2
