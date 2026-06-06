"""
配置管理器

支持 YAML/JSON 读写、点号路径 get/set、深度合并、验证。
"""

import copy
import json
import os
from typing import Any, Dict, Optional

import yaml


# 默认配置结构
DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "encoding": "utf-8",
        "chunk_threshold_mb": 100,
    },
    "modeling": {
        "n_splits": 5,
        "optimizer": "bayesian",
        "hyperparam_trials": 20,
        "auto_sample": True,
        "max_samples": 50000,
        "feature_selection": "mi",
        "ensemble": "weighted",
        "encoding": "auto",
    },
    "deep_learning": {
        "enabled": False,
        "models": ["torch_mlp"],
        "use_amp": False,
    },
    "performance": {
        "strategy_preference": "balanced",
        "n_jobs": -1,
        "use_gpu": "auto",
        "enable_kernel_approximation": True,
        "enable_precomputed_kernel_cache": True,
    },
    "cache": {
        "enabled": True,
        "ttl_seconds": 604800,
        "max_entries": 1000,
    },
}

# 默认验证 schema（仅做类型/存在性校验）
DEFAULT_SCHEMA: Dict[str, Any] = {
    "data": {
        "encoding": str,
        "chunk_threshold_mb": int,
    },
    "modeling": {
        "n_splits": int,
        "optimizer": str,
        "hyperparam_trials": int,
        "auto_sample": bool,
        "max_samples": int,
        "feature_selection": str,
        "ensemble": str,
        "encoding": str,
    },
    "deep_learning": {
        "enabled": bool,
        "models": list,
        "use_amp": bool,
    },
    "performance": {
        "strategy_preference": str,
        "n_jobs": int,
        "use_gpu": str,
        "enable_kernel_approximation": bool,
        "enable_precomputed_kernel_cache": bool,
    },
    "cache": {
        "enabled": bool,
        "ttl_seconds": int,
        "max_entries": int,
    },
}


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        if config_path:
            self.load(config_path)
        else:
            self.config = self._default_config()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return copy.deepcopy(DEFAULT_CONFIG)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并两个字典，override 优先级更高。"""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def _split_key(key: str):
        """将点号路径拆分为列表。"""
        return key.split(".") if key else []

    @staticmethod
    def _validate_node(value: Any, schema_node: Any, path: str) -> None:
        """递归校验单个节点。"""
        if isinstance(schema_node, dict):
            if not isinstance(value, dict):
                raise TypeError(f"配置项 '{path}' 应为 dict，实际为 {type(value).__name__}")
            for sub_key, sub_schema in schema_node.items():
                if sub_key not in value:
                    raise ValueError(f"配置项 '{path}' 缺少必需字段 '{sub_key}'")
                ConfigManager._validate_node(value[sub_key], sub_schema, f"{path}.{sub_key}")
        else:
            # schema_node 为期望的 type
            if not isinstance(value, schema_node):
                raise TypeError(
                    f"配置项 '{path}' 类型错误，期望 {schema_node.__name__}，"
                    f"实际为 {type(value).__name__}"
                )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def load(self, path: str) -> "ConfigManager":
        """从 YAML 或 JSON 文件加载配置。"""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        self.config_path = path
        return self

    def save(self, path: Optional[str] = None) -> str:
        """保存配置到 YAML 或 JSON 文件。"""
        target = path or self.config_path
        if not target:
            raise ValueError("未提供保存路径且实例未关联配置文件路径")
        ext = os.path.splitext(target)[1].lower()
        if ext in (".yaml", ".yml"):
            with open(target, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self.config,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
        elif ext == ".json":
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                f.write("\n")
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        self.config_path = target
        return target

    def get(self, key: str, default: Any = None) -> Any:
        """支持点号路径，如 'modeling.n_jobs'。"""
        parts = self._split_key(key)
        node = self.config
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, key: str, value: Any) -> "ConfigManager":
        """支持点号路径，如 'modeling.n_jobs'；中间层级不存在时自动创建 dict。"""
        parts = self._split_key(key)
        if not parts:
            return self
        node = self.config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        return self

    def merge(self, other: Dict[str, Any]) -> "ConfigManager":
        """深度合并外部字典到当前配置。"""
        self.config = self._deep_merge(copy.deepcopy(self.config), other)
        return self

    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置有效性。schema 为 None 时使用内置默认 schema。"""
        if not isinstance(self.config, dict):
            raise ValueError("配置根节点必须是字典")
        effective_schema = schema if schema is not None else DEFAULT_SCHEMA
        self._validate_node(self.config, effective_schema, "<root>")
        return True

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.config)

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "ConfigManager":
        """从现有 IntegratedPipeline 导出配置。"""
        # 本地导入避免循环依赖（原 IntegratedPipeline import 是 no-op 副作用，round 69 清理）
        def _dl_cfg():
            dl = getattr(pipeline, "deep_learning", None)
            if isinstance(dl, dict):
                return {
                    "enabled": dl.get("enabled", False),
                    "models": dl.get("models", ["torch_mlp"]),
                }
            return {"enabled": False, "models": ["torch_mlp"]}

        config = {
            "data": {
                "encoding": getattr(pipeline, "encoding", "auto"),
                "chunk_threshold_mb": 100,
            },
            "modeling": {
                "n_splits": getattr(pipeline, "n_splits", 5),
                "optimizer": getattr(pipeline, "optimizer", "bayesian"),
                "hyperparam_trials": getattr(pipeline, "hyperparam_trials", 20),
                "auto_sample": getattr(pipeline, "auto_sample", True),
                "max_samples": getattr(pipeline, "max_samples", 50000),
                "feature_selection": getattr(pipeline, "feature_selection", "mi"),
                "ensemble": getattr(pipeline, "ensemble", "weighted"),
                "encoding": getattr(pipeline, "encoding", "auto"),
            },
            "deep_learning": _dl_cfg(),
            "performance": {
                "strategy_preference": getattr(pipeline, "strategy_preference", None),
                "n_jobs": -1,
                "use_gpu": "auto",
                "enable_kernel_approximation": getattr(pipeline, "enable_kernel_approximation", True),
                "enable_precomputed_kernel_cache": getattr(pipeline, "enable_precomputed_kernel_cache", True),
            },
            "cache": {
                "enabled": True,
                "ttl_seconds": 604800,
                "max_entries": 1000,
            },
            "pipeline": {
                "target_col": getattr(pipeline, "target_col", None),
                "task_type": getattr(pipeline, "user_task_type", None),
                "model_keys": getattr(pipeline, "model_keys", None),
                "allow_disk_write": getattr(pipeline, "allow_disk_write", True),
                "optimize_hyperparams": getattr(pipeline, "optimize_hyperparams", False),
                "hyperparam_sampler": getattr(pipeline, "hyperparam_sampler", "tpe"),
                "explainability": getattr(pipeline, "explainability", False),
                "auto_decision_mode": getattr(pipeline, "auto_decision_mode", "balanced"),
                "user_override_model": getattr(pipeline, "user_override_model", None),
                "visualization": getattr(pipeline, "visualization", False),
                "dim_reduction": getattr(pipeline, "dim_reduction", "none"),
            },
        }
        inst = cls()
        inst.config = config
        return inst


# ----------------------------------------------------------------------
# 全局单例
# ----------------------------------------------------------------------
_config_manager: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """获取全局 ConfigManager 单例。"""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager
