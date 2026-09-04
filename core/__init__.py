"""建模比赛智能分析引擎的轻量公共入口。

核心包曾在导入任意子模块时一次性加载深度学习、解释器和全部流水线，
导致仅做数据画像也要等待大型依赖初始化。这里使用 PEP 562 延迟导入：
公开 API 保持不变，只有真正访问某个能力时才加载对应模块。
"""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DataModule": ("data_module", "DataModule"),
    "DataLoader": ("data_module", "DataLoader"),
    "TypeDetector": ("data_module", "TypeDetector"),
    "DataCleaner": ("data_module", "DataCleaner"),
    "MissingPatternClassifier": ("missing_engine", "MissingPatternClassifier"),
    "MissingValueHandler": ("missing_engine", "MissingValueHandler"),
    "MissingPattern": ("missing_engine", "MissingPattern"),
    "MissingStrategy": ("missing_engine", "MissingStrategy"),
    "StructuralRule": ("missing_engine", "StructuralRule"),
    "ColumnMissingProfile": ("missing_engine", "ColumnMissingProfile"),
    "MissingReport": ("missing_engine", "MissingReport"),
    "FastMissingClassifier": ("missing_engine", "FastMissingClassifier"),
    "export_missing_report": ("missing_engine", "export_missing_report"),
    "AutoMissingPipeline": ("auto_pipeline", "AutoMissingPipeline"),
    "PipelineConfig": ("auto_pipeline", "PipelineConfig"),
    "PerformanceScheduler": ("performance_scheduler", "PerformanceScheduler"),
    "ExecutionPlan": ("performance_scheduler", "ExecutionPlan"),
    "StrategyLevel": ("performance_scheduler", "StrategyLevel"),
    "DataScaleMetrics": ("performance_scheduler", "DataScaleMetrics"),
    "HardwareProfile": ("performance_scheduler", "HardwareProfile"),
    "HardwareDetector": ("performance_scheduler", "HardwareDetector"),
    "DataScaleEvaluator": ("performance_scheduler", "DataScaleEvaluator"),
    "auto_schedule": ("performance_scheduler", "auto_schedule"),
    "ParallelEngine": ("accelerators", "ParallelEngine"),
    "GPUManager": ("accelerators", "GPUManager"),
    "get_gpu_manager": ("accelerators", "get_gpu_manager"),
    "auto_gpu_model": ("accelerators", "auto_gpu_model"),
    "GPUDataTransformer": ("accelerators", "GPUDataTransformer"),
    "optimize_memory": ("accelerators", "optimize_memory"),
    "get_system_info": ("accelerators", "get_system_info"),
    "gpu_fallback": ("accelerators", "gpu_fallback"),
    "parallelize": ("accelerators", "parallelize"),
    "ParallelModelingEngine": ("parallel_modeling", "ParallelModelingEngine"),
    "ModelRegistry": ("parallel_modeling", "ModelRegistry"),
    "ModelConfig": ("parallel_modeling", "ModelConfig"),
    "ModelResult": ("parallel_modeling", "ModelResult"),
    "Metrics": ("parallel_modeling", "Metrics"),
    "HyperparameterSearch": ("parallel_modeling", "HyperparameterSearch"),
    "quick_model": ("parallel_modeling", "quick_model"),
    "IntegratedPipeline": ("integrated_pipeline", "IntegratedPipeline"),
    "PipelineResult": ("integrated_pipeline", "PipelineResult"),
    "quick_run": ("integrated_pipeline", "quick_run"),
    "WorkspaceManager": ("workspace_manager", "WorkspaceManager"),
    "get_workspace_manager": ("workspace_manager", "get_workspace_manager"),
    "set_workspace_config": ("workspace_manager", "set_workspace_config"),
    "ConfigManager": ("config_manager", "ConfigManager"),
    "get_config": ("config_manager", "get_config"),
    "ModelingEngine": ("modeling_engine", "ModelingEngine"),
    "ModelingResult": ("modeling_engine", "ModelingResult"),
    "TaskType": ("modeling_engine", "TaskType"),
    "TaskTypeDetector": ("modeling_engine", "TaskTypeDetector"),
    "EncodingType": ("modeling_engine", "EncodingType"),
    "FeatureSelectionStrategy": ("modeling_engine", "FeatureSelectionStrategy"),
    "EnsembleMethod": ("modeling_engine", "EnsembleMethod"),
    "AutoEncoder": ("modeling_engine", "AutoEncoder"),
    "AutoFeatureSelector": ("modeling_engine", "AutoFeatureSelector"),
    "ModelLibrary": ("modeling_engine", "ModelLibrary"),
    "ModelSpec": ("modeling_engine", "ModelSpec"),
    "CrossValidator": ("modeling_engine", "CrossValidator"),
    "CVResult": ("modeling_engine", "CVResult"),
    "EnsembleBuilder": ("modeling_engine", "EnsembleBuilder"),
    "DatasetProfile": ("modeling_assistant", "DatasetProfile"),
    "DatasetRelation": ("modeling_assistant", "DatasetRelation"),
    "InteractionFinding": ("modeling_assistant", "InteractionFinding"),
    "ResearchResult": ("modeling_assistant", "ResearchResult"),
    "MathModelingAssistant": ("modeling_assistant", "MathModelingAssistant"),
    "run_modeling_study": ("modeling_assistant", "run_modeling_study"),
    "RunArtifactManager": ("artifact_manager", "RunArtifactManager"),
    "create_run_id": ("artifact_manager", "create_run_id"),
    "ARTIFACT_SCHEMA_VERSION": ("artifact_manager", "ARTIFACT_SCHEMA_VERSION"),
    "UnitDimension": ("mathematical_reasoning", "UnitDimension"),
    "parse_unit": ("mathematical_reasoning", "parse_unit"),
    "extract_column_unit": ("mathematical_reasoning", "extract_column_unit"),
    "check_expression_dimensions": ("mathematical_reasoning", "check_expression_dimensions"),
    "check_equation_dimensions": ("mathematical_reasoning", "check_equation_dimensions"),
    "classify_expression_structure": ("mathematical_reasoning", "classify_expression_structure"),
    "compile_linear_expression": ("mathematical_reasoning", "compile_linear_expression"),
    "MathematicalModelSpec": ("mathematical_reasoning", "MathematicalModelSpec"),
    "EvidenceBundle": ("mathematical_reasoning", "EvidenceBundle"),
    "MathematicalReasoningEngine": ("mathematical_reasoning", "MathematicalReasoningEngine"),
    "MechanisticModelingEngine": ("mechanistic_modeling", "MechanisticModelingEngine"),
    "MechanisticOperatorRegistry": ("mechanistic_modeling", "MechanisticOperatorRegistry"),
    "OperatorDefinition": ("mechanistic_modeling", "OperatorDefinition"),
    "FourLayerModelingPipeline": ("four_layer_modeling", "FourLayerModelingPipeline"),
    "SemanticContractLayer": ("four_layer_modeling", "SemanticContractLayer"),
    "UnifiedMathematicalIRLayer": ("four_layer_modeling", "UnifiedMathematicalIRLayer"),
    "StructureAwareSolverPlanner": ("four_layer_modeling", "StructureAwareSolverPlanner"),
    "IndependentResultAuditor": ("four_layer_modeling", "IndependentResultAuditor"),
    "MathematicalStructureDefinition": ("four_layer_modeling", "MathematicalStructureDefinition"),
    "MathematicalStructureRegistry": ("four_layer_modeling", "MathematicalStructureRegistry"),
    "SolverSpecification": ("four_layer_modeling", "SolverSpecification"),
    "UniversalRelationValidator": ("universal_math_solvers", "UniversalRelationValidator"),
    "UniversalSolverRegistry": ("universal_math_solvers", "UniversalSolverRegistry"),
    "HierarchicalDecisionCompiler": (
        "hierarchical_decision_compiler", "HierarchicalDecisionCompiler"
    ),
    "SemanticCompilerConfig": ("semantic_model_compiler", "SemanticCompilerConfig"),
    "SemanticModelCompiler": ("semantic_model_compiler", "SemanticModelCompiler"),
    "SemanticCompletionBackend": ("semantic_model_compiler", "SemanticCompletionBackend"),
    "CallableSemanticBackend": ("semantic_model_compiler", "CallableSemanticBackend"),
    "HttpSemanticBackend": ("semantic_model_compiler", "HttpSemanticBackend"),
    "TableTransformError": ("table_transformer", "TableTransformError"),
    "TransformationResult": ("table_transformer", "TransformationResult"),
    "TableTransformationEngine": ("table_transformer", "TableTransformationEngine"),
    "TableTransformationPlanner": ("table_transformer", "TableTransformationPlanner"),
    "MathematicalDataCompiler": ("mathematical_data_compiler", "MathematicalDataCompiler"),
    "CompiledDataView": ("mathematical_data_compiler", "CompiledDataView"),
    "InteractiveVisualizationCompiler": (
        "interactive_visualization", "InteractiveVisualizationCompiler"
    ),
    "InteractiveVisualizationError": (
        "interactive_visualization", "InteractiveVisualizationError"
    ),
    "HyperparameterOptimizer": ("hyperparameter_optimizer", "HyperparameterOptimizer"),
    "OptimizationResult": ("hyperparameter_optimizer", "OptimizationResult"),
    "SamplerType": ("hyperparameter_optimizer", "SamplerType"),
    "quick_optimize": ("hyperparameter_optimizer", "quick_optimize"),
    "ExplainabilityEngine": ("explainability", "ExplainabilityEngine"),
    "ExplanationResult": ("explainability", "ExplanationResult"),
    "explain_model_quick": ("explainability", "explain_model_quick"),
    "TorchMLP": ("deep_learning", "TorchMLP"),
    "TabNetWrapper": ("deep_learning", "TabNetWrapper"),
    "register_deep_learning_models": ("deep_learning", "register_deep_learning_models"),
}

__all__ = [
    name for name in _EXPORTS
    if name not in {"TorchMLP", "TabNetWrapper", "register_deep_learning_models"}
]


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
