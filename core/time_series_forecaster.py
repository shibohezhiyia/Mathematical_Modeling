# -*- coding: utf-8 -*-
"""
普适性时间序列预测模块 (TimeSeriesForecaster)

基于项目 ModelingEngine 构建，支持：
- 自动时序特征工程（滞后、滚动统计、日历特征）
- 外部因素合并（天气、节假日、活动日等）
- 递归多步预测
- 多序列批量建模
- 类别级聚合预测
- 自动可视化与报告

普适性设计：不依赖具体业务字段，通过配置字典映射列名。

使用示例：
    config = {
        'date_col': '日期',
        'target_col': '销量',
        'id_cols': ['门店编号', '商品代码'],
        'category_col': '类别名称',
        'external_df': weather_df,
        'external_cols': ['天气', '温度', '节日', '活动日', '工休日'],
        'external_date_col': '日期',
    }
    forecaster = TimeSeriesForecaster(config)
    results = forecaster.fit_predict(daily_sales, future_dates)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from core.modeling_engine import ModelingEngine, TaskType, FeatureSelectionStrategy
from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


@dataclass
class TSConfig:
    """时间序列预测配置"""
    date_col: str = 'date'
    target_col: str = 'target'
    id_cols: List[str] = field(default_factory=list)
    category_col: Optional[str] = None
    
    # 外部数据
    external_df: Optional[pd.DataFrame] = None
    external_cols: List[str] = field(default_factory=list)
    external_date_col: str = 'date'
    
    # 特征工程参数
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    
    # 建模参数
    model_keys: List[str] = field(default_factory=lambda: ['ridge', 'rf', 'lgb'])
    n_splits: int = 5
    ensemble: str = 'weighted'
    n_jobs: int = -1
    
    # 预测参数
    forecast_horizon: int = 7
    
    # 输出目录（自动创建 表格/文字/图片 子目录）
    output_dir: str = 'temp'
    
    # 其他
    verbose: bool = True
    random_state: int = 42


class TimeSeriesFeatureEngineer:
    """时序特征工程师：纯特征工程，无状态"""
    
    def __init__(self, config: TSConfig):
        self.cfg = config
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建日历特征"""
        df = df.copy()
        dt = pd.to_datetime(df[self.cfg.date_col])
        df['ts_dow'] = dt.dt.dayofweek
        df['ts_month'] = dt.dt.month
        df['ts_day'] = dt.dt.day
        df['ts_is_weekend'] = (df['ts_dow'] >= 5).astype(int)
        df['ts_is_month_start'] = dt.dt.is_month_start.astype(int)
        df['ts_is_month_end'] = dt.dt.is_month_end.astype(int)
        df['ts_quarter'] = dt.dt.quarter
        df['ts_weekofyear'] = dt.dt.isocalendar().week.astype(int)
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建滞后特征"""
        for lag in self.cfg.lags:
            df[f'ts_lag_{lag}'] = df[self.cfg.target_col].shift(lag)
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建滚动统计特征"""
        for w in self.cfg.rolling_windows:
            df[f'ts_roll_mean_{w}'] = df[self.cfg.target_col].shift(1).rolling(w, min_periods=1).mean()
            df[f'ts_roll_std_{w}'] = df[self.cfg.target_col].shift(1).rolling(w, min_periods=1).std()
            df[f'ts_roll_max_{w}'] = df[self.cfg.target_col].shift(1).rolling(w, min_periods=1).max()
            df[f'ts_roll_min_{w}'] = df[self.cfg.target_col].shift(1).rolling(w, min_periods=1).min()
        return df
    
    def merge_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """合并外部数据"""
        if self.cfg.external_df is None:
            return df
        ext = self.cfg.external_df.copy()
        ext[self.cfg.external_date_col] = pd.to_datetime(ext[self.cfg.external_date_col]).dt.date
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col]).dt.date
        df = df.merge(ext, left_on=self.cfg.date_col, right_on=self.cfg.external_date_col, how='left')
        if self.cfg.external_date_col != self.cfg.date_col and self.cfg.external_date_col in df.columns:
            df.drop(columns=[self.cfg.external_date_col], inplace=True)
        return df
    
    def encode_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码外部特征（天气类别等）"""
        # 天气类型 One-Hot（如果存在）
        if '天气' in df.columns:
            for val in df['天气'].dropna().unique():
                df[f'weather_{val}'] = (df['天气'] == val).astype(int)
        # 节日编码
        if '节日' in df.columns:
            df['ts_is_festival'] = (df['节日'] != 0).astype(int)
        # 温度差
        if '温度' in df.columns and '温度.1' in df.columns:
            df['ts_temp_diff'] = df['温度'] - df['温度.1']
        return df
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整特征工程管道"""
        df = df.sort_values(self.cfg.date_col).reset_index(drop=True)
        df = self.create_calendar_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.merge_external(df)
        df = self.encode_external(df)
        return df
    
    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """获取特征列（排除目标、ID、日期等）"""
        exclude = {self.cfg.date_col, self.cfg.target_col}
        exclude.update(self.cfg.id_cols)
        exclude.update(['天气', '节日', '星期几'])
        return [c for c in df.columns if c not in exclude and df[c].dtype.kind in 'iufcb']


class TimeSeriesForecaster:
    """
    普适性时间序列预测器
    
    支持批量多序列建模、递归多步预测、外部因素融合。
    输出自动分类到 output_dir/表格、output_dir/文字、output_dir/图片
    """
    
    def __init__(self, config: TSConfig):
        self.cfg = config
        self.engineer = TimeSeriesFeatureEngineer(config)
        self.models: Dict[tuple, Any] = {}
        self.feature_cols: Dict[tuple, List[str]] = {}
        self.cv_scores: Dict[tuple, Dict] = {}
        
        # 创建分类输出目录
        self.dir_tables = os.path.join(config.output_dir, '表格')
        self.dir_texts = os.path.join(config.output_dir, '文字')
        self.dir_images = os.path.join(config.output_dir, '图片')
        for d in [self.dir_tables, self.dir_texts, self.dir_images]:
            os.makedirs(d, exist_ok=True)
    
    def _prepare_series(self, df: pd.DataFrame, key_vals: tuple) -> pd.DataFrame:
        """筛选单条序列并补全日期"""
        mask = pd.Series(True, index=df.index)
        for col, val in zip(self.cfg.id_cols, key_vals):
            mask &= (df[col] == val)
        ts = df[mask].copy()
        
        # 补全日期
        if len(ts) == 0:
            return ts
        
        all_dates = pd.date_range(ts[self.cfg.date_col].min(), ts[self.cfg.date_col].max(), freq='D')
        full = pd.DataFrame({self.cfg.date_col: all_dates.date})
        ts[self.cfg.date_col] = pd.to_datetime(ts[self.cfg.date_col]).dt.date
        full = full.merge(ts, on=self.cfg.date_col, how='left')
        
        # 填充ID列
        for col, val in zip(self.cfg.id_cols, key_vals):
            full[col] = full[col].fillna(val)
        
        # 填充目标为0（无销售记录）
        full[self.cfg.target_col] = full[self.cfg.target_col].fillna(0)
        
        # 填充其他列为前向/后向填充
        for col in full.columns:
            if col not in [self.cfg.date_col, self.cfg.target_col] + list(self.cfg.id_cols):
                full[col] = full[col].ffill().bfill()
        
        return full
    
    def _train_single(self, ts_df: pd.DataFrame) -> Tuple[Any, List[str], Dict]:
        """训练单条序列模型"""
        ts_df = self.engineer.build(ts_df)
        feat_cols = self.engineer.get_feature_cols(ts_df)
        
        # 删除含NaN的行（主要是滞后特征开头）
        train_df = ts_df.dropna(subset=feat_cols + [self.cfg.target_col])
        
        if len(train_df) < 30:
            raise ValueError(f"样本过少: {len(train_df)}")
        
        X = train_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = train_df[self.cfg.target_col]
        
        # 快速模式：n_splits=1 时直接训练，跳过 ModelingEngine 的复杂流程
        if self.cfg.n_splits <= 1:
            from core.modeling_engine import ModelLibrary, TaskType
            task_type = TaskType.REGRESSION
            key = self.cfg.model_keys[0] if self.cfg.model_keys else 'ridge'
            model = ModelLibrary.create_model(key, task_type)
            if model is None:
                raise ValueError(f"模型创建失败: {key}")
            model.fit(X, y)
            return model, feat_cols, {'rmse_cv': None, 'model_result': None}
        
        engine = ModelingEngine(
            task_type='regression',
            model_keys=self.cfg.model_keys,
            n_splits=self.cfg.n_splits,
            encoding='none',
            feature_selection=FeatureSelectionStrategy.NONE,
            ensemble=self.cfg.ensemble,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            # 滞后/滚动特征只能用过去数据验证；随机 K 折会让未来样本进入
            # 训练集，产生时间穿越并高估离线得分。
            fold_type='time',
            auto_decision_mode='balanced'
        )
        result = engine.fit(X, y)
        
        cv_rmse = None
        if result.leaderboard is not None and not result.leaderboard.empty:
            # ModelingEngine 的排行榜列以 "<metric>_mean" 命名。
            rmse_col = 'rmse_mean' if 'rmse_mean' in result.leaderboard.columns else 'rmse'
            if rmse_col in result.leaderboard.columns:
                cv_rmse = float(result.leaderboard.iloc[0][rmse_col])
        
        # 获取最佳模型（最后一个fold的拟合模型）
        best_model = result.best_cv_result.fitted_models[-1] if result.best_cv_result and result.best_cv_result.fitted_models else None
        if best_model is None:
            raise ValueError("模型训练失败，未能获取最佳模型")
        return best_model, feat_cols, {'rmse_cv': cv_rmse, 'model_result': result}
    
    def fit(self, df: pd.DataFrame) -> 'TimeSeriesForecaster':
        """
        批量训练所有序列模型
        
        Args:
            df: 原始数据，必须包含 date_col, target_col, id_cols
        """
        if not self.cfg.id_cols:
            raise ValueError("id_cols 不能为空")
        
        keys = df[self.cfg.id_cols].drop_duplicates().values.tolist()
        log_info(f"[TSForecaster] 发现 {len(keys)} 条序列，开始批量训练...")
        
        for key_vals in keys:
            key_tuple = tuple(key_vals)
            try:
                ts = self._prepare_series(df, key_tuple)
                model, feat_cols, scores = self._train_single(ts)
                self.models[key_tuple] = model
                self.feature_cols[key_tuple] = feat_cols
                self.cv_scores[key_tuple] = scores
                if self.cfg.verbose:
                    log_info(f"  训练成功 {key_tuple}, RMSE_CV={scores.get('rmse_cv')}")
            except Exception as e:
                log_warning(f"  训练失败 {key_tuple}: {e}")
                self.models[key_tuple] = None
        
        success = sum(1 for m in self.models.values() if m is not None)
        log_info(f"[TSForecaster] 训练完成: {success}/{len(keys)} 成功")
        return self
    
    def _predict_recursive(self, model, ts_df: pd.DataFrame, feat_cols: List[str],
                           future_dates: List[datetime.date]) -> List[float]:
        """递归预测未来多步"""
        predictions = []
        current = ts_df.copy()
        cfg = self.cfg
        
        for d in future_dates:
            new_row = pd.DataFrame({cfg.date_col: [d]})
            
            # 日历特征
            new_row = self.engineer.create_calendar_features(new_row)
            
            # 外部特征
            if cfg.external_df is not None:
                ext_row = cfg.external_df[cfg.external_df[cfg.external_date_col] == pd.Timestamp(d)]
                if not ext_row.empty:
                    for c in cfg.external_cols:
                        if c in ext_row.columns:
                            new_row[c] = ext_row[c].values[0]
                new_row = self.engineer.encode_external(new_row)
            
            # 滞后特征
            if len(current) > 0:
                hist = current[cfg.target_col]
                for lag in cfg.lags:
                    val = hist.iloc[-lag] if len(hist) >= lag else hist.iloc[-1]
                    new_row[f'ts_lag_{lag}'] = val
            else:
                for lag in cfg.lags:
                    new_row[f'ts_lag_{lag}'] = 0
            
            # 滚动特征
            hist = current[cfg.target_col]
            for w in cfg.rolling_windows:
                tail = hist.tail(w) if len(hist) >= 1 else pd.Series([0])
                new_row[f'ts_roll_mean_{w}'] = tail.mean()
                new_row[f'ts_roll_std_{w}'] = tail.std() if len(tail) > 1 else 0
                new_row[f'ts_roll_max_{w}'] = tail.max()
                new_row[f'ts_roll_min_{w}'] = tail.min()
            
            # 对齐列
            for c in feat_cols:
                if c not in new_row.columns:
                    new_row[c] = 0
            
            X_pred = new_row[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            pred = model.predict(X_pred)[0]
            pred = max(0, float(pred))
            predictions.append(pred)
            
            new_row[cfg.target_col] = pred
            current = pd.concat([current, new_row], ignore_index=True, copy=False)
        
        return predictions
    
    def predict(self, df: pd.DataFrame, future_dates: List[datetime.date]) -> pd.DataFrame:
        """
        预测未来多步
        
        Args:
            df: 历史数据（用于构建滞后特征）
            future_dates: 未来日期列表
        
        Returns:
            DataFrame: 预测结果
        """
        results = []
        keys = df[self.cfg.id_cols].drop_duplicates().values.tolist()
        
        for key_vals in keys:
            key_tuple = tuple(key_vals)
            model = self.models.get(key_tuple)
            feat_cols = self.feature_cols.get(key_tuple)
            
            if model is None or feat_cols is None:
                # 使用历史均值作为备选
                ts = self._prepare_series(df, key_tuple)
                mean_val = ts[self.cfg.target_col].mean()
                for d in future_dates:
                    row = {cfg: val for cfg, val in zip(self.cfg.id_cols, key_vals)}
                    row[self.cfg.date_col] = d
                    row[f'{self.cfg.target_col}_pred'] = mean_val
                    results.append(row)
                continue
            
            ts = self._prepare_series(df, key_tuple)
            preds = self._predict_recursive(model, ts, feat_cols, future_dates)
            
            for i, d in enumerate(future_dates):
                row = {cfg: val for cfg, val in zip(self.cfg.id_cols, key_vals)}
                row[self.cfg.date_col] = d
                row[f'{self.cfg.target_col}_pred'] = round(preds[i], 2)
                results.append(row)
        
        return pd.DataFrame(results)
    
    def fit_predict(self, df: pd.DataFrame, future_dates: List[datetime.date]) -> pd.DataFrame:
        """训练并预测"""
        return self.fit(df).predict(df, future_dates)
    
    def aggregate_by_category(self, df: pd.DataFrame, future_dates: List[datetime.date]) -> pd.DataFrame:
        """
        按类别聚合后预测（问题二模式）
        
        先按 id_cols + category_col 聚合为日销量，再建模预测。
        """
        if self.cfg.category_col is None:
            raise ValueError("category_col 未配置")
        
        # 按 id_cols + category + date 聚合
        agg_df = df.groupby(self.cfg.id_cols + [self.cfg.category_col, self.cfg.date_col])[self.cfg.target_col].sum().reset_index()
        
        # 临时替换 id_cols 为 id_cols + category
        old_id_cols = self.cfg.id_cols
        self.cfg.id_cols = old_id_cols + [self.cfg.category_col]
        
        # 重新训练预测
        result = self.fit_predict(agg_df, future_dates)
        
        # 恢复配置
        self.cfg.id_cols = old_id_cols
        return result
    
    def analyze_external_impact(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        分析外部因素对销量的影响（问题三模式）
        
        基于特征重要性均值评估各外部因素的影响程度。
        """
        if not self.models:
            log_warning("[TSForecaster] 尚未训练模型，先执行 fit()")
            return pd.DataFrame()
        
        impact_records = []
        
        for key_tuple, model in self.models.items():
            if model is None:
                continue
            feat_cols = self.feature_cols.get(key_tuple, [])
            
            # 获取特征重要性
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
            
            if importances is None:
                continue
            
            fi_df = pd.DataFrame({'feature': feat_cols, 'importance': importances})
            
            # 分类统计外部因素重要性
            external_prefixes = ['weather_', 'ts_is_festival', 'ts_temp_diff', '活动日', '工休日', '风力', '温度']
            for prefix in external_prefixes:
                matched = fi_df[fi_df['feature'].str.startswith(prefix) | (fi_df['feature'] == prefix)]
                if not matched.empty:
                    impact_records.append({
                        'series_key': str(key_tuple),
                        'factor_group': prefix,
                        'mean_importance': matched['importance'].mean(),
                        'max_importance': matched['importance'].max(),
                        'feature_count': len(matched)
                    })
            
            # 时间因素
            time_feats = [c for c in feat_cols if c.startswith('ts_dow') or c.startswith('ts_month') 
                          or c.startswith('ts_is_weekend') or c.startswith('ts_quarter')]
            if time_feats:
                matched = fi_df[fi_df['feature'].isin(time_feats)]
                impact_records.append({
                    'series_key': str(key_tuple),
                    'factor_group': 'calendar_time',
                    'mean_importance': matched['importance'].mean(),
                    'max_importance': matched['importance'].max(),
                    'feature_count': len(matched)
                })
            
            # 历史滞后
            lag_feats = [c for c in feat_cols if c.startswith('ts_lag_')]
            if lag_feats:
                matched = fi_df[fi_df['feature'].isin(lag_feats)]
                impact_records.append({
                    'series_key': str(key_tuple),
                    'factor_group': 'lag_history',
                    'mean_importance': matched['importance'].mean(),
                    'max_importance': matched['importance'].max(),
                    'feature_count': len(matched)
                })
        
        impact_df = pd.DataFrame(impact_records)
        if not impact_df.empty:
            summary = impact_df.groupby('factor_group').agg({
                'mean_importance': 'mean',
                'max_importance': 'mean',
                'feature_count': 'sum'
            }).reset_index().sort_values('mean_importance', ascending=False)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # 保存到文字目录
                text_dir = os.path.join(save_dir, '文字')
                os.makedirs(text_dir, exist_ok=True)
                summary.to_csv(os.path.join(text_dir, 'external_impact_summary.csv'), index=False, encoding='utf-8-sig')
                impact_df.to_csv(os.path.join(text_dir, 'external_impact_detail.csv'), index=False, encoding='utf-8-sig')
            
            return summary
        return impact_df
    
    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[tuple, Dict]:
        """
        时间序列交叉验证（滚动原点验证）
        
        对每个序列做滚动原点验证，返回各序列的CV评分。
        """
        scores = {}
        keys = df[self.cfg.id_cols].drop_duplicates().values.tolist()
        
        for key_vals in keys:
            key_tuple = tuple(key_vals)
            ts = self._prepare_series(df, key_tuple)
            ts = self.engineer.build(ts)
            feat_cols = self.engineer.get_feature_cols(ts)
            ts = ts.dropna(subset=feat_cols + [self.cfg.target_col])
            
            if len(ts) < 60:
                continue
            
            X = ts[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = ts[self.cfg.target_col]
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            rmse_list, mae_list = [], []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                engine = ModelingEngine(
                    task_type='regression',
                    model_keys=self.cfg.model_keys[:2],
                    n_splits=3,
                    encoding='none',
                    feature_selection='none',
                    ensemble='best_single',
                    random_state=self.cfg.random_state,
                    n_jobs=self.cfg.n_jobs
                )
                result = engine.fit(X_train, y_train)
                preds = result.best_model.predict(X_test)
                rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
                mae_list.append(mean_absolute_error(y_test, preds))
            
            scores[key_tuple] = {
                'rmse_mean': np.mean(rmse_list),
                'rmse_std': np.std(rmse_list),
                'mae_mean': np.mean(mae_list),
                'mae_std': np.std(mae_list)
            }
        
        return scores
    
    def plot_series(self, df: pd.DataFrame, predictions: pd.DataFrame, 
                    n_samples: int = 6, filename: str = 'series_preview.png'):
        """绘制部分序列的历史+预测图，自动保存到图片目录"""
        save_path = os.path.join(self.dir_images, filename)
        try:
            import matplotlib.pyplot as plt
            from core.visualization import _init_matplotlib
            _init_matplotlib()
            
            keys = df[self.cfg.id_cols].drop_duplicates().values.tolist()
            sample_keys = keys[:n_samples]
            
            n_rows = (len(sample_keys) + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
            if n_rows == 1:
                axes = np.array([axes]).reshape(1, -1)
            axes = axes.flatten()
            
            for idx, key_vals in enumerate(sample_keys):
                ax = axes[idx]
                key_tuple = tuple(key_vals)
                
                ts = self._prepare_series(df, key_tuple)
                ts['date_dt'] = pd.to_datetime(ts[self.cfg.date_col])
                
                pred_mask = pd.Series(True, index=predictions.index)
                for col, val in zip(self.cfg.id_cols, key_vals):
                    pred_mask &= (predictions[col] == val)
                pred = predictions[pred_mask].copy()
                
                if not pred.empty:
                    pred['date_dt'] = pd.to_datetime(pred[self.cfg.date_col])
                    ax.plot(ts['date_dt'], ts[self.cfg.target_col], label='历史', alpha=0.7)
                    ax.plot(pred['date_dt'], pred[f'{self.cfg.target_col}_pred'], 'r--', marker='o', label='预测')
                    ax.set_title(str(key_tuple))
                    ax.legend()
            
            for idx in range(len(sample_keys), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log_info(f"[TSForecaster] 序列图已保存: {save_path}")
            plt.close(fig)
        except Exception as e:
            log_warning(f"[TSForecaster] 绘图失败: {e}")
    
    def save_report(self, predictions: pd.DataFrame, subdir_name: str = 'report'):
        """保存预测结果和报告到分类目录"""
        # 表格
        predictions.to_csv(os.path.join(self.dir_tables, f'{subdir_name}_predictions.csv'), index=False, encoding='utf-8-sig')
        
        # CV评分
        if self.cv_scores:
            cv_df = pd.DataFrame([
                {'series': str(k), **v} for k, v in self.cv_scores.items()
            ])
            cv_df.to_csv(os.path.join(self.dir_tables, f'{subdir_name}_cv_scores.csv'), index=False, encoding='utf-8-sig')
