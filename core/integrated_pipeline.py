"""
集成流水线（V2）

将性能调度器 + 数据模块 + 缺失分析 + 建模引擎 串联为端到端方案。
基于 ModelingEngine 构建，支持自动任务类型判断、智能编码、特征选择、K折CV、多模型融合。

使用方式：
    pipeline = IntegratedPipeline()
    result = pipeline.run(df)
    result.predictions  # 测试集预测
    result.leaderboard  # 模型排行榜
    result.report       # 完整报告
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

import pandas as pd
import numpy as np

from core.performance_scheduler import PerformanceScheduler, ExecutionPlan, StrategyLevel
from core.auto_pipeline import AutoMissingPipeline, PipelineConfig
from core.modeling_engine import (
    ModelingEngine, ModelingResult,
    TaskType, TaskTypeDetector,
    EncodingType, FeatureSelectionStrategy, EnsembleMethod
)
from core.workspace_manager import get_workspace_manager, set_workspace_config
from utils.helpers import log_info, log_warning, timer

# 可选特性模块的懒加载助手：3 个 try/except 块用到的内部函数
# 都从这几个模块导入，没有循环依赖问题。提升到模块级后，4 个函数内
# import 全部消除。
from core.drift_detection import detect_drift
from core.meta_feature_extractor import MetaFeatureExtractor
from core.automl_strategy import AutoMLStrategy
from core.visualization import plot_modeling_summary, plot_data_profile


@dataclass
class PipelineResult:
    """流水线执行结果"""
    # 数据
    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    X_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    X_test: Optional[pd.DataFrame] = None
    
    # 建模
    predictions: Optional[np.ndarray] = None
    oof_predictions: Optional[np.ndarray] = None
    leaderboard: Optional[pd.DataFrame] = None
    feature_importance: Optional[pd.DataFrame] = None
    ensemble_weights: Optional[Dict[str, float]] = None
    
    # 报告
    execution_plan: Optional[ExecutionPlan] = None
    missing_report: Optional[Any] = None
    modeling_result: Optional[ModelingResult] = None
    decision_report: Optional[Any] = None
    visualization_paths: Optional[Dict[str, str]] = None
    
    # 元信息
    target_col: Optional[str] = None
    task_type: str = ""
    total_time: float = 0.0
    strategy: str = ""


class IntegratedPipeline:
    """
    集成流水线：端到端建模比赛解决方案（V2）
    
    自动执行：
    1. 性能调度 → 选择最优策略
    2. 数据加载与类型识别
    3. 缺失值智能分析（真缺失/结构性缺失/目标缺失）
    4. 智能编码（OneHot/Label/Target/Ordinal）
    5. 自动特征选择
    6. K折交叉验证训练（多模型）
    7. 自动评估与决策
    8. 模型融合与预测
    
    用户只需：pipeline.run(df)
    
    用户覆盖：
    - model_keys=['lr', 'xgb']  指定模型（跳过自动评估）
    - user_override_model='xgb' 覆盖自动推荐
    - auto_decision_mode='accuracy_first'  切换决策模式
    """
    
    def __init__(self,
                 strategy_preference: Optional[str] = None,
                 target_col: Optional[str] = None,
                 task_type: Optional[str] = None,
                 model_keys: Optional[List[str]] = None,
                 allow_disk_write: bool = True,
                 encoding: str = 'auto',
                 feature_selection: str = 'mi',
                 ensemble: str = 'weighted',
                 n_splits: int = 5,
                 optimize_hyperparams: bool = False,
                 hyperparam_trials: int = 20,
                 hyperparam_sampler: str = 'tpe',
                 explainability: bool = False,
                 auto_decision_mode: str = 'balanced',
                 user_override_model: Optional[str] = None,
                 visualization: bool = False,
                 auto_sample: bool = True,
                 max_samples: int = 50_000,
                 deep_learning: Optional[Dict] = None,
                 optimizer: str = 'bayesian',
                 dim_reduction: str = 'none',
                 enable_kernel_approximation: bool = True,
                 enable_precomputed_kernel_cache: bool = True,
                 progress_callback: Optional[callable] = None,
                 **kwargs) -> None:
        """
        Args:
            strategy_preference: 策略偏好 ('standard', 'fast', 'ultra', None=自动)
            target_col: 目标列名（None=自动识别）
            task_type: 任务类型 ('classification', 'regression', 'clustering', None=自动推断)
            model_keys: 指定模型列表（None=全部可用模型）
            allow_disk_write: 是否允许磁盘写入（默认True，可设为False禁止一切C盘/磁盘操作）
            encoding: 编码策略 ('auto', 'onehot', 'label', 'target', 'none')
            feature_selection: 特征选择 ('mi', 'variance', 'rfe', 'model_based', 'correlation', 'pca', 'none')
            ensemble: 融合策略 ('weighted', 'voting_hard', 'voting_soft', 'stacking', 'best_single')
            n_splits: K折交叉验证折数
            optimize_hyperparams: 是否启用超参优化
            hyperparam_trials: 超参搜索次数
            hyperparam_sampler: 'tpe', 'cmaes', 'random'
            explainability: 是否启用可解释性分析
            auto_decision_mode: 自动决策模式 ('accuracy_first', 'speed_first', 'stability_first', 'simplicity_first', 'balanced')
            user_override_model: 用户覆盖的模型key（None=接受自动推荐）
            visualization: 是否生成可视化图表
            auto_sample: 是否启用自动降采样
            max_samples: 自动降采样的最大样本数
            deep_learning: 深度学习配置 {'enabled': bool, 'models': List[str]}
            optimizer: 优化器 'bayesian', 'rl', 'both'
            dim_reduction: 降维 'none', 'pca', 'autoencoder'
            progress_callback: 进度回调函数，签名为 (step: str, current: int, total: int, message: str) -> None
        """
        self.strategy_preference = strategy_preference
        self.target_col = target_col
        self.user_task_type = task_type
        self.model_keys = model_keys
        self.allow_disk_write = allow_disk_write
        self.encoding = encoding
        self.feature_selection = feature_selection
        self.ensemble = ensemble
        self.n_splits = n_splits
        self.optimize_hyperparams = optimize_hyperparams
        self.hyperparam_trials = hyperparam_trials
        self.hyperparam_sampler = hyperparam_sampler
        self.explainability = explainability
        self.auto_decision_mode = auto_decision_mode
        self.user_override_model = user_override_model
        self.visualization = visualization
        self.auto_sample = auto_sample
        self.max_samples = max_samples
        self.deep_learning = deep_learning
        self.optimizer = optimizer
        self.dim_reduction = dim_reduction
        self.enable_kernel_approximation = enable_kernel_approximation
        self.enable_precomputed_kernel_cache = enable_precomputed_kernel_cache
        self.progress_callback = progress_callback
        
        # 初始化工作空间管理器
        set_workspace_config(allow_disk_write=allow_disk_write)
        
        # 吸收 web/app.py 传来的额外参数，避免 TypeError
        if kwargs:
            from utils.helpers import log_warning
            log_warning(f"[IntegratedPipeline] 忽略未识别的参数: {list(kwargs.keys())}")
        
        self.result = PipelineResult()
    
    @timer
    def run(self, df: pd.DataFrame) -> PipelineResult:
        """
        执行完整流水线
        
        Args:
            df: 原始数据
                - 监督学习：train + test 合并，目标列测试集部分为NaN
                - 聚类：无目标列，全部用于聚类
            
        Returns:
            PipelineResult
        """
        overall_start = time.time()
        log_info("=" * 70)
        log_info("集成流水线 V2 启动".center(60))
        log_info("=" * 70)
        
        def _notify(step, current, total, message):
            if self.progress_callback:
                self.progress_callback(step, current, total, message)
        
        # ============================================================
        # Phase 1: 性能调度
        # ============================================================
        _notify('preprocessing', 1, 6, '性能调度...')
        log_info("\n[Phase 1/6] 性能调度...")
        scheduler = PerformanceScheduler(
            user_preference=StrategyLevel(self.strategy_preference) if self.strategy_preference else None
        )
        plan = scheduler.schedule(df)
        self.result.execution_plan = plan
        self.result.strategy = plan.strategy.value
        log_info(f"  策略: {plan.strategy.value} | n_jobs: {plan.n_jobs} | GPU: {plan.use_gpu}")
        _notify('preprocessing', 1, 6, f'性能调度完成: {plan.strategy.value}模式')
        
        # ============================================================
        # Phase 2: 数据预处理（类型识别 + 缺失分析）
        # ============================================================
        _notify('preprocessing', 2, 6, '数据预处理（类型识别 + 缺失分析）...')
        log_info("\n[Phase 2/6] 数据预处理...")
        
        # 判断是否为聚类任务（无标签）
        is_clustering = self.user_task_type == 'clustering' if self.user_task_type else False
        has_target = self.target_col is not None or not is_clustering
        
        # 如果有目标列，先尝试识别
        if has_target and self.target_col is None:
            # 启发式：缺失率在10%-90%之间的最后一列可能是目标
            for col in reversed(df.columns):
                missing_rate = df[col].isnull().sum() / len(df)
                if 0.05 < missing_rate < 0.95:
                    self.target_col = col
                    log_info(f"  自动识别目标列: {col} (缺失率: {missing_rate:.1%})")
                    break
        
        # 使用自动缺失分析流程
        if has_target and self.target_col and self.target_col in df.columns:
            missing_config = PipelineConfig(
                target_col=self.target_col,
                fast_mode=(plan.strategy in [StrategyLevel.FAST, StrategyLevel.ULTRA]),
                sample_size=plan.missing_sample_size,
                structural_threshold=plan.missing_structural_threshold,
                drop_col_threshold=0.95,
                allow_disk_write=self.allow_disk_write
            )
            
            missing_pipeline = AutoMissingPipeline(missing_config)
            train_df, test_df, missing_report = missing_pipeline.run(df)
            self.result.missing_report = missing_report
            self.result.target_col = self.target_col
            
            log_info(f"  目标列: {self.target_col}")
            log_info(f"  训练集: {train_df.shape}, 测试集: {test_df.shape if test_df is not None else None}")
            missing_summary = f"缺失值处理完成"
            if missing_report and hasattr(missing_report, 'imputed_count'):
                missing_summary += f"，填充 {missing_report.imputed_count} 个值"
            _notify('preprocessing', 2, 6, missing_summary)
        else:
            # 聚类：全部数据
            train_df = df.copy()
            test_df = None
            self.result.target_col = None
            log_info("  聚类模式：无目标列")
            _notify('preprocessing', 2, 6, '聚类模式：无目标列')
        
        # ============================================================
        # Phase 3: 分离特征与标签
        # ============================================================
        _notify('preprocessing', 3, 6, '分离特征与标签...')
        log_info("\n[Phase 3/6] 分离特征与标签...")
        
        if self.target_col and self.target_col in train_df.columns:
            y_train = train_df[self.target_col]
            X_train = train_df.drop(columns=[self.target_col])
        else:
            y_train = None
            X_train = train_df.copy()
        
        X_test = None
        if test_df is not None:
            X_test = test_df.drop(columns=[self.target_col], errors='ignore')
        
        _notify('preprocessing', 3, 6, f'特征: {X_train.shape[1]} 列, 样本: {X_train.shape[0]} 行')
        
        # ============================================================
        # Phase 3.5: 分布漂移检测
        # ============================================================
        if X_test is not None and len(X_test) > 0:
            try:
                drift_report = detect_drift(X_train, X_test, method='auto', threshold=0.05)
                self.result.drift_report = drift_report.to_dict()
                log_info(f"[DriftDetection] {drift_report}")
                if drift_report.is_drifted:
                    log_warning(f"[DriftDetection] ⚠️ 检测到分布漂移！训练集与测试集分布差异明显")
                    log_warning(f"[DriftDetection] 建议：检查数据划分策略，或启用领域适配/样本重加权")
                else:
                    log_info("[DriftDetection] ✅ 训练集与测试集分布一致")
            except Exception as e:
                log_warning(f"[DriftDetection] 漂移检测失败: {e}")
        
        # ============================================================
        # Phase 4: 启动建模引擎
        # ============================================================
        _notify('preprocessing', 4, 6, '启动建模引擎...')
        log_info("\n[Phase 4/6] 启动建模引擎...")
        
        # 元特征提取与 AutoML 策略推荐
        auto_recommendation = None
        if self.optimizer == 'auto' or (self.deep_learning and self.deep_learning.get('enabled') == 'auto'):
            try:
                meta = MetaFeatureExtractor().extract(X_train, y_train,
                    TaskTypeDetector.detect(y_train, X_train, self.user_task_type))
                log_info(f"[AutoML] 元特征: n={meta.n_samples}, features={meta.n_features}, complexity={meta.complexity_score:.0f}")
                
                auto_recommendation = AutoMLStrategy.recommend(
                    meta=meta,
                    task_type=TaskTypeDetector.detect(y_train, X_train, self.user_task_type),
                    user_preference=self.auto_decision_mode,
                    user_optimizer=None if self.optimizer == 'auto' else self.optimizer,
                    user_model_keys=self.model_keys
                )
                log_info(f"[AutoML] 推荐优化器: {auto_recommendation.optimizer}")
                log_info(f"[AutoML] 推荐模型: {auto_recommendation.model_keys}")
                log_info(f"[AutoML] 预计耗时: {auto_recommendation.expected_time}")
                
                self.result.automl_recommendation = auto_recommendation
            except Exception as e:
                log_warning(f"[AutoML] 策略推荐失败: {e}")
        
        # 解析参数
        enc = EncodingType.ONEHOT if self.encoding == 'onehot' else \
              EncodingType.LABEL if self.encoding == 'label' else \
              EncodingType.TARGET if self.encoding == 'target' else \
              EncodingType.NONE if self.encoding == 'none' else EncodingType.AUTO
        
        fs = FeatureSelectionStrategy.VARIANCE if self.feature_selection == 'variance' else \
             FeatureSelectionStrategy.RFE if self.feature_selection == 'rfe' else \
             FeatureSelectionStrategy.MODEL_BASED if self.feature_selection == 'model_based' else \
             FeatureSelectionStrategy.CORRELATION if self.feature_selection == 'correlation' else \
             FeatureSelectionStrategy.PCA_DIM if self.feature_selection == 'pca' else \
             FeatureSelectionStrategy.NONE if self.feature_selection == 'none' else \
             FeatureSelectionStrategy.MI
        
        ens = EnsembleMethod.VOTING_HARD if self.ensemble == 'voting_hard' else \
              EnsembleMethod.VOTING_SOFT if self.ensemble == 'voting_soft' else \
              EnsembleMethod.STACKING if self.ensemble == 'stacking' else \
              EnsembleMethod.BEST_SINGLE if self.ensemble == 'best_single' else \
              EnsembleMethod.WEIGHTED
        
        # 应用 AutoML 推荐
        effective_optimizer = auto_recommendation.optimizer if auto_recommendation and self.optimizer == 'auto' else self.optimizer
        effective_model_keys = auto_recommendation.model_keys if auto_recommendation and self.model_keys is None else self.model_keys
        effective_ensemble = auto_recommendation.ensemble if auto_recommendation and self.ensemble == 'weighted' else self.ensemble
        effective_dl = auto_recommendation.deep_learning if auto_recommendation and self.deep_learning and self.deep_learning.get('enabled') == 'auto' else self.deep_learning
        
        # Fast/Ultra 模式：精简模型列表 + 降低 CV 折数 + 减少 trials
        effective_n_splits = self.n_splits
        effective_trials = self.hyperparam_trials
        if plan.strategy in [StrategyLevel.FAST, StrategyLevel.ULTRA]:
            effective_n_splits = plan.cv_folds
            effective_trials = min(self.hyperparam_trials or plan.hyperparameter_trials, plan.hyperparameter_trials)
            # 用户未指定模型时，优先使用轻量模型；若用户已指定则保留（由底层超时保护）
            if effective_model_keys is None:
                n_samples = X_train.shape[0]
                # 检测任务类型以区分回归/分类可用模型
                detected_task = TaskTypeDetector.detect(y_train, X_train, self.user_task_type)
                is_classification = detected_task == TaskType.CLASSIFICATION
                if n_samples > 30000:
                    # 大数据快速模型：树模型优先（同时支持回归/分类），线性模型补充
                    # 大数据用 hist_gb 替代 sklearn 原生 gbdt（速度提升 5-10x）
                    if is_classification:
                        fast_models = ['lgb', 'xgb', 'catboost', 'rf', 'et', 'hist_gb',
                                       'ridge', 'lasso', 'elastic', 'linear', 'bayesian_ridge']
                    else:
                        fast_models = ['lgb', 'xgb', 'catboost', 'rf', 'et', 'hist_gb',
                                       'ridge', 'lasso', 'elastic', 'linear', 'bayesian_ridge',
                                       'huber', 'linear_svm']
                    effective_model_keys = fast_models[:plan.max_models]
                    log_info(f"[Performance] Fast模式大数据精简模型 ({detected_task.value}): {effective_model_keys}")
                else:
                    # 小数据可用更多模型，但仍限制数量
                    # 小数据保留 gbdt，大数据 hist_gb 已在上面处理
                    if is_classification:
                        all_fast = ['lgb', 'xgb', 'catboost', 'rf', 'et', 'hist_gb', 'gbdt', 'dt',
                                    'ridge', 'lasso', 'elastic', 'linear', 'bayesian_ridge',
                                    'svm', 'knn', 'mlp']
                    else:
                        all_fast = ['lgb', 'xgb', 'catboost', 'rf', 'et', 'hist_gb', 'gbdt', 'dt',
                                    'ridge', 'lasso', 'elastic', 'linear', 'bayesian_ridge',
                                    'huber', 'pls', 'svm', 'knn', 'torch_mlp']
                    effective_model_keys = all_fast[:plan.max_models]
                    log_info(f"[Performance] Fast模式精简模型 ({detected_task.value}): {effective_model_keys}")
        
        effective_max_samples = self.max_samples
        if self.auto_sample and plan.sample_size is not None:
            effective_max_samples = min(self.max_samples, plan.sample_size)
            if effective_max_samples != self.max_samples:
                log_info(
                    f"[Performance] 按 {plan.strategy.value} 资源预算限制建模样本: "
                    f"{self.max_samples:,} → {effective_max_samples:,}"
                )

        engine = ModelingEngine(
            task_type=self.user_task_type,
            model_keys=effective_model_keys,
            n_splits=effective_n_splits,
            encoding=enc,
            feature_selection=fs,
            ensemble=ens,
            random_state=42,
            n_jobs=plan.n_jobs,
            use_gpu=plan.use_gpu,
            optimize_hyperparams=self.optimize_hyperparams,
            hyperparam_trials=effective_trials,
            hyperparam_sampler=self.hyperparam_sampler,
            explainability=self.explainability,
            auto_decision_mode=self.auto_decision_mode,
            user_override_model=self.user_override_model,
            auto_sample=self.auto_sample,
            max_samples=effective_max_samples,
            deep_learning=effective_dl,
            optimizer=effective_optimizer,
            dim_reduction=self.dim_reduction,
            enable_kernel_approximation=self.enable_kernel_approximation,
            enable_precomputed_kernel_cache=self.enable_precomputed_kernel_cache,
            progress_callback=self.progress_callback,
            pipeline_notify=_notify
        )
        
        # ============================================================
        # Phase 5: 训练
        # ============================================================
        _notify('training', 5, 6, '开始训练所有模型...')
        log_info("\n[Phase 5/6] 训练所有模型...")
        modeling_result = engine.fit(X_train, y_train, X_test)
        self.result.modeling_result = modeling_result
        _notify('training', 5, 6, f'训练完成，{len(modeling_result.cv_results)} 个模型成功')
        
        # 传递决策报告
        if modeling_result and modeling_result.decision_report:
            self.result.decision_report = modeling_result.decision_report
        
        # ============================================================
        # Phase 6: 可视化（可选）
        # ============================================================
        _notify('preprocessing', 6, 6, '生成可视化图表...')
        if self.visualization and modeling_result:
            log_info("\n[Phase 6/7] 生成可视化图表...")
            try:
                # 数据探索图
                viz_paths = plot_data_profile(
                    df, target=self.target_col, task_type=self.user_task_type,
                    save_dir='pipeline_viz/data'
                )
                
                # 模型结果图
                model_viz = plot_modeling_summary(
                    modeling_result,
                    X_train=X_train, y_train=y_train,
                    save_dir='pipeline_viz/models',
                    task_type=self.user_task_type or modeling_result.task_type.value
                )
                viz_paths.update(model_viz)
                
                self.result.visualization_paths = viz_paths
                log_info(f"[Pipeline] 已生成 {len(viz_paths)} 张可视化图表")
            except Exception as e:
                log_warning(f"[Pipeline] 可视化生成失败: {e}")
        
        # ============================================================
        # Phase 7: 汇总结果
        # ============================================================
        _notify('done', 7, 7, '汇总结果...')
        log_info("\n[Phase 7/7] 汇总结果...")
        
        self.result.train_df = train_df
        self.result.test_df = test_df
        self.result.X_train = X_train
        self.result.y_train = y_train
        self.result.X_test = X_test
        self.result.task_type = modeling_result.task_type.value
        self.task_type = self.result.task_type
        self.result.leaderboard = modeling_result.leaderboard
        self.result.feature_importance = modeling_result.feature_importance
        self.result.oof_predictions = modeling_result.cv_results[0].oof_pred if modeling_result.cv_results else None
        
        if modeling_result.ensemble_result:
            self.result.ensemble_weights = modeling_result.ensemble_result.get('weights')
            self.result.predictions = modeling_result.ensemble_result.get('test')
            _notify('done', 7, 7, f'融合完成，推荐模型: {modeling_result.best_model_key}')
        else:
            _notify('done', 7, 7, f'推荐模型: {modeling_result.best_model_key}')
        
        self.result.total_time = time.time() - overall_start
        
        log_info("=" * 70)
        log_info(f"流水线完成，总耗时: {self.result.total_time:.1f}s".center(60))
        log_info("=" * 70)
        
        return self.result
    
    def print_summary(self) -> None:
        """打印完整摘要"""
        r = self.result
        print("\n" + "=" * 70)
        print("集成流水线执行报告".center(60))
        print("=" * 70)
        
        print(f"\n【执行策略】")
        print(f"  策略级别: {r.strategy}")
        print(f"  任务类型: {r.task_type}")
        print(f"  目标列: {r.target_col or '无（聚类）'}")
        print(f"  总耗时: {r.total_time:.1f}s")
        
        if r.execution_plan:
            p = r.execution_plan
            print(f"\n【资源配置】")
            print(f"  n_jobs: {p.n_jobs}")
            print(f"  GPU: {'启用' if p.use_gpu else '禁用'}")
            print(f"  CV折数: {self.n_splits}")
        
        print(f"\n【数据规模】")
        if r.train_df is not None:
            print(f"  训练集: {r.train_df.shape}")
        if r.test_df is not None:
            print(f"  测试集: {r.test_df.shape}")
        
        # 预处理信息
        if r.modeling_result and r.modeling_result.preprocessing_info:
            info = r.modeling_result.preprocessing_info
            print(f"\n【预处理】")
            print(f"  原始特征: {info.get('original_features')}")
            print(f"  编码后: {info.get('encoded_features')}")
            print(f"  选择后: {info.get('selected_features')}")
        
        # 编码报告
        if r.modeling_result and r.modeling_result.encoding_report is not None:
            print(f"\n【编码策略】")
            print(r.modeling_result.encoding_report.to_string(index=False))
        
        print(f"\n【模型排行榜】")
        if r.leaderboard is not None and not r.leaderboard.empty:
            print(r.leaderboard.to_string(index=False))
        else:
            print("  无")
        
        # 融合权重
        if r.ensemble_weights:
            print(f"\n【融合权重】")
            for model, weight in sorted(r.ensemble_weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model:20s}: {weight:.3f}")
        
        if r.feature_importance is not None and not r.feature_importance.empty:
            print(f"\n【Top 10 重要特征】")
            for row in r.feature_importance.head(10).itertuples(index=False):
                print(f"  {row.feature:30s}: {row.importance:.4f}")
        
        print("\n" + "=" * 70)
    
    def export_predictions(self, path: str, id_col: Optional[str] = None) -> Optional[str]:
        """导出预测结果"""
        wm = get_workspace_manager()
        
        if self.result.predictions is None:
            raise ValueError("没有预测结果")
        
        df = pd.DataFrame()
        if id_col and self.result.test_df is not None and id_col in self.result.test_df.columns:
            df[id_col] = self.result.test_df[id_col].values
        
        df['prediction'] = self.result.predictions
        
        if self.task_type == 'classification' and self.result.predictions.ndim == 1:
            df['prediction_label'] = (self.result.predictions > 0.5).astype(int)
        
        safe_path = wm.save_dataframe(df, path, subdir='reports')
        if safe_path:
            log_info(f"预测结果已导出: {safe_path}")
        return safe_path


# =============================================================================
# 便捷接口
# =============================================================================

def quick_run(df: pd.DataFrame,
              target_col: Optional[str] = None,
              task_type: Optional[str] = None,
              model_keys: Optional[List[str]] = None,
              ensemble: str = 'weighted',
              n_splits: int = 5) -> PipelineResult:
    """
    快速运行完整流水线
    
    Args:
        df: 原始数据
        target_col: 目标列
        task_type: 任务类型
        model_keys: 指定模型
        ensemble: 融合策略
        n_splits: CV折数
        
    Returns:
        PipelineResult
    """
    pipeline = IntegratedPipeline(
        target_col=target_col,
        task_type=task_type,
        model_keys=model_keys,
        ensemble=ensemble,
        n_splits=n_splits
    )
    result = pipeline.run(df)
    pipeline.print_summary()
    return result
