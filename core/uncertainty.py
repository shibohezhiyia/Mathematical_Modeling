"""
不确定性量化模块：概率校准 + 共形预测

解决业务痛点：
- 模型只给点预测（销量=120），业务需要知道"大概范围"
- 分类概率不准（说 0.9 实际命中率远不到 90%）
- 需要置信区间做风控/库存决策

提供：
1. ProbabilityCalibrator: 分类概率校准（Platt / Isotonic / Temperature）
2. ConformalPredictor: 回归/分类共形预测（带统计保证的预测区间）
3. PredictionInterval: 统一接口，输出点预测 + 区间
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from scipy import stats

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class ProbabilityCalibrator:
    """
    分类概率校准器

    用法：
        calibrator = ProbabilityCalibrator(method='isotonic')
        calibrator.fit(model, X_val, y_val)  # 用验证集校准
        proba_calibrated = calibrator.predict_proba(X_test)
    """

    def __init__(self, method: str = 'isotonic', random_state: int = 42) -> None:
        """
        Args:
            method: 'platt' | 'isotonic' | 'temperature'
            random_state: 随机种子
        """
        self.method = method
        self.random_state = random_state
        self.calibrator_: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, model: Any, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'ProbabilityCalibrator':
        """用验证集拟合校准器"""
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        # 获取原始概率
        if hasattr(model, 'predict_proba'):
            raw_proba = model.predict_proba(X)
        else:
            raise ValueError("模型必须支持 predict_proba()")

        if self.method == 'platt':
            self._fit_platt(raw_proba, y)
        elif self.method == 'isotonic':
            self._fit_isotonic(raw_proba, y)
        elif self.method == 'temperature':
            self._fit_temperature(raw_proba, y)
        else:
            raise ValueError(f"未知校准方法: {self.method}")

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], model: Any) -> np.ndarray:
        """对测试集输出校准后的概率"""
        raw_proba = model.predict_proba(X)
        return self._calibrate(raw_proba)

    def _fit_platt(self, raw_proba: np.ndarray, y: np.ndarray) -> None:
        """Platt Scaling: 用 Logistic Regression 拟合概率映射"""
        # 二分类：取正类概率
        if raw_proba.shape[1] == 2:
            scores = raw_proba[:, 1]
            self.calibrator_ = LogisticRegression(C=1e10, max_iter=1000)
            self.calibrator_.fit(scores.reshape(-1, 1), y)
        else:
            # 多分类：每个类别单独拟合一个 Platt
            self.calibrator_ = []
            for i, cls in enumerate(self.classes_):
                scores = raw_proba[:, i]
                lr = LogisticRegression(C=1e10, max_iter=1000)
                lr.fit(scores.reshape(-1, 1), (y == cls).astype(int))
                self.calibrator_.append(lr)

    def _fit_isotonic(self, raw_proba: np.ndarray, y: np.ndarray) -> None:
        """Isotonic Regression: 单调非参数校准"""
        if raw_proba.shape[1] == 2:
            scores = raw_proba[:, 1]
            self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
            self.calibrator_.fit(scores, (y == self.classes_[1]).astype(float))
        else:
            self.calibrator_ = []
            for i, cls in enumerate(self.classes_):
                scores = raw_proba[:, i]
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(scores, (y == cls).astype(float))
                self.calibrator_.append(iso)

    def _fit_temperature(self, raw_proba: np.ndarray, y: np.ndarray) -> None:
        """Temperature Scaling: 用单参数 T 平滑 softmax"""
        # 简化为从 logits 出发，但 sklearn 模型通常没有 logits
        # 这里用近似：T = argmin NLL，通过简单搜索
        best_T = 1.0
        best_nll = float('inf')
        for T in np.linspace(0.5, 3.0, 50):
            # 从概率反推 logits 再重新 softmax
            eps = 1e-10
            logits = np.log(np.clip(raw_proba, eps, 1 - eps))
            scaled = np.exp(logits / T)
            scaled /= scaled.sum(axis=1, keepdims=True)
            # 计算 NLL
            y_idx = np.searchsorted(self.classes_, y)
            nll = -np.log(np.clip(scaled[np.arange(len(y)), y_idx], eps, 1)).mean()
            if nll < best_nll:
                best_nll = nll
                best_T = T
        self.calibrator_ = {'T': best_T}
        log_info(f"[ProbabilityCalibrator] Temperature Scaling: T={best_T:.3f}")

    def _calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        if self.calibrator_ is None:
            return raw_proba

        if self.method == 'platt':
            if raw_proba.shape[1] == 2:
                scores = raw_proba[:, 1]
                calibrated = self.calibrator_.predict_proba(scores.reshape(-1, 1))[:, 1]
                return np.column_stack([1 - calibrated, calibrated])
            else:
                calibrated = np.zeros_like(raw_proba)
                for i, lr in enumerate(self.calibrator_):
                    scores = raw_proba[:, i]
                    calibrated[:, i] = lr.predict_proba(scores.reshape(-1, 1))[:, 1]
                # 归一化
                calibrated /= calibrated.sum(axis=1, keepdims=True)
                return calibrated

        elif self.method == 'isotonic':
            if raw_proba.shape[1] == 2:
                scores = raw_proba[:, 1]
                calibrated = self.calibrator_.predict(scores)
                calibrated = np.clip(calibrated, 0, 1)
                return np.column_stack([1 - calibrated, calibrated])
            else:
                calibrated = np.zeros_like(raw_proba)
                for i, iso in enumerate(self.calibrator_):
                    scores = raw_proba[:, i]
                    calibrated[:, i] = iso.predict(scores)
                calibrated = np.clip(calibrated, 0, 1)
                calibrated /= calibrated.sum(axis=1, keepdims=True)
                return calibrated

        elif self.method == 'temperature':
            T = self.calibrator_['T']
            eps = 1e-10
            logits = np.log(np.clip(raw_proba, eps, 1 - eps))
            scaled = np.exp(logits / T)
            scaled /= scaled.sum(axis=1, keepdims=True)
            return scaled

        return raw_proba


class ConformalPredictor:
    """
    共形预测器（Split Conformal Prediction）

    为回归/分类提供有统计保证的预测区间：
        P(y ∈ C(x)) ≥ 1 - α

    用法（回归）：
        cp = ConformalPredictor(alpha=0.1)  # 90% 置信区间
        cp.fit(model, X_calib, y_calib)
        intervals = cp.predict_interval(X_test)
        # intervals = [[lower, upper], [lower, upper], ...]

    用法（分类）：
        cp = ConformalPredictor(alpha=0.1)
        cp.fit(model, X_calib, y_calib, task_type='classification')
        sets = cp.predict_set(X_test)
        # sets = [{label1, label2}, {label3}, ...]
    """

    def __init__(self, alpha: float = 0.1, random_state: int = 42) -> None:
        """
        Args:
            alpha: 错误率（0.1 = 90% 置信区间）
            random_state: 随机种子
        """
        self.alpha = alpha
        self.random_state = random_state
        self.model_: Optional[Any] = None
        self.q_hat_: Optional[float] = None
        self.task_type_: Optional[str] = None
        self.scores_: Optional[np.ndarray] = None

    def _compute_q_hat(self, scores: np.ndarray) -> float:
        """根据非一致性分数计算共形分位数 q_hat。

        公式：q_level = ceil((n+1)*(1-alpha)) / n
        修正：分位数水平 cap 在 [0, 1]，避免 np.quantile 在水平 > 1 时抛 Warning。

        提取为辅助方法：fit() 在回归/分类两个分支都需要此计算，
        之前是两段几乎相同的代码。
        """
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        return float(np.quantile(scores, min(q_level, 1.0)))

    def fit(self, model: Any,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            task_type: str = 'regression') -> 'ConformalPredictor':
        """用校准集计算共形分位数"""
        X = np.array(X)
        y = np.array(y).ravel()
        self.model_ = model
        self.task_type_ = task_type

        if task_type == 'regression':
            pred = model.predict(X)
            # 非一致性分数：|y - ŷ|
            self.scores_ = np.abs(y - pred)
            self.q_hat_ = self._compute_q_hat(self.scores_)
            log_info(f"[ConformalPredictor] 回归校准完成: q_hat={self.q_hat_:.4f}, alpha={self.alpha}, n_calib={len(self.scores_)}")
        else:
            # 分类：使用 softmax 分数的非一致性度量
            if not hasattr(model, 'predict_proba'):
                raise ValueError("分类共形预测需要 predict_proba()")
            proba = model.predict_proba(X)
            # 非一致性分数：1 - ŷ_y（真实标签的概率）
            y_idx = np.searchsorted(model.classes_, y)
            self.scores_ = 1 - proba[np.arange(len(y)), y_idx]
            self.q_hat_ = self._compute_q_hat(self.scores_)
            log_info(f"[ConformalPredictor] 分类校准完成: q_hat={self.q_hat_:.4f}, alpha={self.alpha}, n_calib={len(self.scores_)}")

        return self

    def predict_interval(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        回归：返回预测区间 [lower, upper]
        
        Returns:
            ndarray of shape (n_samples, 2)
        """
        if self.task_type_ != 'regression':
            raise ValueError("predict_interval() 仅适用于回归任务")
        if self.q_hat_ is None:
            raise ValueError("请先调用 fit()")

        X = np.array(X)
        pred = self.model_.predict(X)
        lower = pred - self.q_hat_
        upper = pred + self.q_hat_
        return np.column_stack([lower, upper])

    def predict_set(self, X: Union[pd.DataFrame, np.ndarray]) -> List[set]:
        """
        分类：返回预测集合（可能包含多个标签）
        
        Returns:
            List[set]: 每个样本的预测标签集合
        """
        if self.task_type_ != 'classification':
            raise ValueError("predict_set() 仅适用于分类任务")
        if self.q_hat_ is None:
            raise ValueError("请先调用 fit()")

        X = np.array(X)
        proba = self.model_.predict_proba(X)
        classes = self.model_.classes_

        # 向量化：列表推导式替代显式循环
        threshold = 1 - self.q_hat_
        result_sets = [set(classes[p >= threshold]) for p in proba]
        return result_sets

    def predict_with_interval(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """回归：同时返回点预测和区间"""
        X = np.array(X)
        pred = self.model_.predict(X)
        intervals = self.predict_interval(X)
        return {
            'point': pred,
            'lower': intervals[:, 0],
            'upper': intervals[:, 1],
            'interval_width': intervals[:, 1] - intervals[:, 0],
        }

    def get_coverage_guarantee(self) -> str:
        """返回覆盖率保证说明"""
        return f"保证覆盖率为 ≥ {100 * (1 - self.alpha):.1f}%"


class PredictionIntervalReporter:
    """
    预测区间报告生成器

    把共形预测结果格式化为业务友好的报告。
    """

    @staticmethod
    def regression_report(y_true: Optional[np.ndarray],
                          prediction_result: Dict[str, np.ndarray],
                          alpha: float = 0.1) -> Dict:
        """生成回归预测区间报告"""
        point = prediction_result['point']
        lower = prediction_result['lower']
        upper = prediction_result['upper']
        width = prediction_result['interval_width']

        report = {
            'n_samples': len(point),
            'alpha': alpha,
            'confidence_level': f"{100*(1-alpha):.0f}%",
            'mean_prediction': float(np.mean(point)),
            'mean_interval_width': float(np.mean(width)),
            'median_interval_width': float(np.median(width)),
            'max_interval_width': float(np.max(width)),
            'min_interval_width': float(np.min(width)),
            'interval_width_ratio': float(np.mean(width) / (np.mean(np.abs(point)) + 1e-6)),
        }

        if y_true is not None:
            # 计算实际覆盖率
            covered = (y_true >= lower) & (y_true <= upper)
            report['actual_coverage'] = float(np.mean(covered))
            report['coverage_error'] = float(report['actual_coverage'] - (1 - alpha))

        return report

    @staticmethod
    def classification_report(y_true: Optional[np.ndarray],
                               prediction_sets: List[set],
                               alpha: float = 0.1) -> Dict:
        """生成分类预测集合报告"""
        avg_set_size = np.mean([len(s) for s in prediction_sets])
        report = {
            'n_samples': len(prediction_sets),
            'alpha': alpha,
            'confidence_level': f"{100*(1-alpha):.0f}%",
            'mean_set_size': float(avg_set_size),
            'empty_sets': sum(1 for s in prediction_sets if len(s) == 0),
            'singleton_sets': sum(1 for s in prediction_sets if len(s) == 1),
        }

        if y_true is not None:
            covered = [yt in ps for yt, ps in zip(y_true, prediction_sets)]
            report['actual_coverage'] = float(np.mean(covered))
            report['coverage_error'] = float(report['actual_coverage'] - (1 - alpha))

        return report


def calibrate_and_conformal(model: Any,
                            X_train: Union[pd.DataFrame, np.ndarray],
                            y_train: Union[pd.Series, np.ndarray],
                            X_test: Union[pd.DataFrame, np.ndarray],
                            task_type: str = 'regression',
                            alpha: float = 0.1,
                            calib_size: float = 0.2,
                            random_state: int = 42) -> Dict[str, Any]:
    """
    一键完成：校准 + 共形预测

    用法：
        result = calibrate_and_conformal(model, X_train, y_train, X_test)
        print(result['point'])        # 点预测
        print(result['lower'])        # 区间下限
        print(result['upper'])        # 区间上限
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()

    # 划分训练/校准集
    if calib_size > 0:
        X_tr, X_calib, y_tr, y_calib = train_test_split(
            X_train, y_train, test_size=calib_size, random_state=random_state,
        )
        model = clone(model)
        model.fit(X_tr, y_tr)
    else:
        X_calib = X_train
        y_calib = y_train

    cp = ConformalPredictor(alpha=alpha, random_state=random_state)
    cp.fit(model, X_calib, y_calib, task_type=task_type)

    if task_type == 'regression':
        result = cp.predict_with_interval(X_test)
        result['model'] = model
        return result
    else:
        result = {
            'prediction_sets': cp.predict_set(X_test),
            'point': model.predict(X_test),
            'model': model,
        }
        return result
