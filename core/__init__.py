"""
建模比赛智能分析引擎 - 核心模块
"""
from .data_module import DataModule, DataLoader, TypeDetector, DataCleaner
from .missing_engine import (
    MissingPatternClassifier, MissingValueHandler,
    MissingPattern, MissingStrategy, StructuralRule,
    ColumnMissingProfile, MissingReport,
    FastMissingClassifier, export_missing_report
)
from .auto_pipeline import AutoMissingPipeline, PipelineConfig
from .performance_scheduler import (
    PerformanceScheduler, ExecutionPlan, StrategyLevel,
    DataScaleMetrics, HardwareProfile, HardwareDetector,
    DataScaleEvaluator, auto_schedule
)
from .accelerators import (
    ParallelEngine, GPUManager, get_gpu_manager,
    auto_gpu_model, GPUDataTransformer, optimize_memory,
    get_system_info, gpu_fallback, parallelize
)
from .parallel_modeling import (
    ParallelModelingEngine, ModelRegistry, ModelConfig,
    ModelResult, Metrics, HyperparameterSearch,
    quick_model
)
from .integrated_pipeline import IntegratedPipeline, PipelineResult, quick_run
from .workspace_manager import WorkspaceManager, get_workspace_manager, set_workspace_config
from .config_manager import ConfigManager, get_config
from .modeling_engine import (
    ModelingEngine, ModelingResult,
    TaskType, TaskTypeDetector,
    EncodingType, FeatureSelectionStrategy, EnsembleMethod,
    AutoEncoder, AutoFeatureSelector,
    ModelLibrary, ModelSpec,
    CrossValidator, CVResult,
    EnsembleBuilder
)
from .hyperparameter_optimizer import (
    HyperparameterOptimizer, OptimizationResult,
    SamplerType, quick_optimize
)
from .explainability import (
    ExplainabilityEngine, ExplanationResult,
    explain_model_quick
)
try:
    from .deep_learning import (
        TorchMLP, TabNetWrapper,
        register_deep_learning_models
    )
except ImportError:
    pass

__all__ = [
    # 数据模块
    'DataModule', 'DataLoader', 'TypeDetector', 'DataCleaner',
    # 缺失分析
    'MissingPatternClassifier', 'MissingValueHandler',
    'MissingPattern', 'MissingStrategy', 'StructuralRule',
    'ColumnMissingProfile', 'MissingReport',
    'FastMissingClassifier', 'export_missing_report',
    'AutoMissingPipeline', 'PipelineConfig',
    # 性能调度
    'PerformanceScheduler', 'ExecutionPlan', 'StrategyLevel',
    'DataScaleMetrics', 'HardwareProfile', 'HardwareDetector',
    'DataScaleEvaluator', 'auto_schedule',
    # 加速层
    'ParallelEngine', 'GPUManager', 'get_gpu_manager',
    'auto_gpu_model', 'GPUDataTransformer', 'optimize_memory',
    'get_system_info', 'gpu_fallback', 'parallelize',
    # 并行建模
    'ParallelModelingEngine', 'ModelRegistry', 'ModelConfig',
    'ModelResult', 'Metrics', 'HyperparameterSearch',
    'quick_model',
    # 集成流水线
    'IntegratedPipeline', 'PipelineResult',
    # 工作空间管理
    'WorkspaceManager', 'get_workspace_manager', 'set_workspace_config',
    'ConfigManager', 'get_config',
    # 建模引擎
    'ModelingEngine', 'ModelingResult',
    'TaskType', 'TaskTypeDetector',
    'EncodingType', 'FeatureSelectionStrategy', 'EnsembleMethod',
    'AutoEncoder', 'AutoFeatureSelector',
    'ModelLibrary', 'ModelSpec',
    'CrossValidator', 'CVResult',
    'EnsembleBuilder',
    'quick_run',
    # 超参优化
    'HyperparameterOptimizer', 'OptimizationResult',
    'SamplerType', 'quick_optimize',
    # 可解释性
    'ExplainabilityEngine', 'ExplanationResult',
    'explain_model_quick',
]
