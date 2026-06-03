"""
分布漂移检测模块

用途：训练前/上线前检测训练集与测试集/新数据之间的分布差异，
      如果漂移严重则发出警告，避免模型失效。

支持方法：
- KS Test（数值特征逐列比较）
- PSI（Population Stability Index，风控常用）
- Adversarial Validation（用分类器判断 train/test 是否可分）
- Wasserstein Distance（最优传输距离）
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class DriftReport:
    """漂移检测报告"""

    def __init__(self,
                 is_drifted: bool,
                 overall_score: float,
                 threshold: float,
                 method: str,
                 feature_scores: Optional[Dict[str, float]] = None,
                 details: Optional[Dict] = None) -> None:
        self.is_drifted = is_drifted
        self.overall_score = overall_score
        self.threshold = threshold
        self.method = method
        self.feature_scores = feature_scores or {}
        self.details = details or {}

    def to_dict(self) -> Dict:
        return {
            'is_drifted': self.is_drifted,
            'overall_score': float(self.overall_score),
            'threshold': float(self.threshold),
            'method': self.method,
            'feature_scores': {k: float(v) for k, v in self.feature_scores.items()},
            'details': self.details,
        }

    def __repr__(self) -> str:
        flag = "⚠️ DRIFTED" if self.is_drifted else "✅ OK"
        return f"DriftReport({flag}, score={self.overall_score:.4f}, threshold={self.threshold:.4f}, method={self.method})"


class DriftDetector:
    """
    分布漂移检测器

    用法：
        detector = DriftDetector(method='auto', threshold=0.05)
        detector.fit_reference(X_train)
        report = detector.detect(X_test)
        if report.is_drifted:
            print("警告：数据发生漂移！")
    """

    def __init__(self,
                 method: str = 'auto',
                 threshold: float = 0.05,
                 numerical_method: str = 'ks',
                 categorical_method: str = 'chi2',
                 adversarial_estimator: Any = None,
                 random_state: int = 42) -> None:
        """
        Args:
            method: 'auto' | 'ks' | 'psi' | 'adversarial' | 'wasserstein'
            threshold: 漂移阈值（越小越敏感）
            numerical_method: 数值特征比较方法
            categorical_method: 类别特征比较方法
            adversarial_estimator: 对抗验证用的分类器
            random_state: 随机种子
        """
        self.method = method
        self.threshold = threshold
        self.numerical_method = numerical_method
        self.categorical_method = categorical_method
        self.adversarial_estimator = adversarial_estimator or RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1
        )
        self.random_state = random_state
        self._reference: Optional[pd.DataFrame] = None
        self._ref_stats: Optional[Dict] = None
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []

    def fit_reference(self, X: Union[pd.DataFrame, np.ndarray]) -> 'DriftDetector':
        """记录参考分布（通常是训练集）"""
        X = self._to_df(X)
        self._reference = X.copy()
        self._num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self._cat_cols = [c for c in X.columns if c not in self._num_cols]

        # 预计算参考分布统计量
        self._ref_stats = {}
        for col in self._num_cols:
            self._ref_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'q25': X[col].quantile(0.25),
                'q75': X[col].quantile(0.75),
                'hist': np.histogram(X[col].dropna(), bins=20, range=(X[col].min(), X[col].max()))[0],
            }
        for col in self._cat_cols:
            self._ref_stats[col] = X[col].value_counts(normalize=True).to_dict()

        log_info(f"[DriftDetector] 参考分布已记录: {len(self._num_cols)} 数值列, {len(self._cat_cols)} 类别列")
        return self

    def detect(self, X: Union[pd.DataFrame, np.ndarray],
               method: Optional[str] = None) -> DriftReport:
        """检测新数据是否相对于参考分布发生漂移"""
        if self._reference is None:
            raise ValueError("请先调用 fit_reference()")

        X = self._to_df(X)
        method = method or self.method

        if method == 'auto':
            return self._detect_auto(X)
        elif method == 'ks':
            return self._detect_ks(X)
        elif method == 'psi':
            return self._detect_psi(X)
        elif method == 'adversarial':
            return self._detect_adversarial(X)
        elif method == 'wasserstein':
            return self._detect_wasserstein(X)
        else:
            raise ValueError(f"未知漂移检测方法: {method}")

    def _detect_auto(self, X: pd.DataFrame) -> DriftReport:
        """自动选择最佳检测方法"""
        # 小数据用 KS，大数据用 adversarial
        n = len(self._reference)
        if n < 5000:
            return self._detect_ks(X)
        elif n < 50000:
            return self._detect_adversarial(X)
        else:
            # 超大数据：采样后做 adversarial
            return self._detect_adversarial(X, sample_size=20000)

    def _detect_ks(self, X: pd.DataFrame) -> DriftReport:
        """KS Test：逐列比较，Bonferroni 校正"""
        feature_scores = {}
        drifted_features = []
        pvalues = []

        for col in self._num_cols:
            ref_vals = self._reference[col].dropna()
            new_vals = X[col].dropna() if col in X.columns else pd.Series([], dtype=float)
            if len(ref_vals) == 0 or len(new_vals) == 0:
                continue
            stat, pvalue = stats.ks_2samp(ref_vals, new_vals)
            feature_scores[col] = pvalue
            pvalues.append(pvalue)
            if pvalue < self.threshold:
                drifted_features.append(col)

        # Bonferroni 校正
        n_tests = max(len(pvalues), 1)
        corrected_threshold = self.threshold / n_tests
        is_drifted = any(p < corrected_threshold for p in pvalues)
        overall_score = 1.0 - min(pvalues) if pvalues else 0.0

        return DriftReport(
            is_drifted=is_drifted,
            overall_score=overall_score,
            threshold=1.0 - corrected_threshold,
            method='ks_bonferroni',
            feature_scores=feature_scores,
            details={
                'drifted_features': drifted_features,
                'n_tests': n_tests,
                'corrected_threshold': corrected_threshold,
            }
        )

    def _detect_psi(self, X: pd.DataFrame, bins: int = 10) -> DriftReport:
        """PSI（Population Stability Index）"""
        feature_scores = {}
        total_psi = 0.0
        valid_cols = 0

        for col in self._num_cols:
            ref_vals = self._reference[col].dropna()
            new_vals = X[col].dropna() if col in X.columns else pd.Series([], dtype=float)
            if len(ref_vals) == 0 or len(new_vals) == 0:
                continue

            # 用参考分布的等频分箱
            bin_edges = np.percentile(ref_vals, np.linspace(0, 100, bins + 1))
            bin_edges[0] -= 1e-6
            bin_edges[-1] += 1e-6

            ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
            new_counts, _ = np.histogram(new_vals, bins=bin_edges)

            ref_pct = ref_counts / ref_counts.sum()
            new_pct = new_counts / new_counts.sum()

            # 避免除零
            ref_pct = np.where(ref_pct == 0, 1e-10, ref_pct)
            new_pct = np.where(new_pct == 0, 1e-10, new_pct)

            psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
            feature_scores[col] = float(psi)
            total_psi += psi
            valid_cols += 1

        avg_psi = total_psi / valid_cols if valid_cols > 0 else 0.0
        # PSI 解释: <0.1 正常, 0.1~0.25 轻微漂移, >0.25 严重漂移
        is_drifted = avg_psi > 0.25

        return DriftReport(
            is_drifted=is_drifted,
            overall_score=avg_psi,
            threshold=0.25,
            method='psi',
            feature_scores=feature_scores,
            details={'avg_psi': avg_psi, 'valid_cols': valid_cols}
        )

    def _detect_adversarial(self, X: pd.DataFrame, sample_size: Optional[int] = None) -> DriftReport:
        """对抗验证：用分类器判断能否区分 train/test"""
        ref = self._reference.copy()
        new = X.copy()

        if sample_size and len(ref) > sample_size:
            ref = ref.sample(n=sample_size, random_state=self.random_state)
        if sample_size and len(new) > sample_size:
            new = new.sample(n=sample_size, random_state=self.random_state)

        # 对齐列
        common_cols = [c for c in ref.columns if c in new.columns]
        ref = ref[common_cols]
        new = new[common_cols]

        # 构建 train=0, test=1 的标签
        X_combined = pd.concat([ref, new], ignore_index=True, copy=False)
        y_combined = np.array([0] * len(ref) + [1] * len(new))

        # 编码类别特征
        for col in X_combined.columns:
            if not pd.api.types.is_numeric_dtype(X_combined[col]):
                X_combined[col] = X_combined[col].astype(str)
                le = LabelEncoder()
                X_combined[col] = le.fit_transform(X_combined[col])

        # 填充缺失值
        X_combined = X_combined.fillna(X_combined.median(numeric_only=True))

        # 用简单分类器做对抗验证
        estimator = clone(self.adversarial_estimator)
        try:
            scores = cross_val_score(estimator, X_combined, y_combined, cv=3,
                                     scoring='roc_auc', n_jobs=-1)
            mean_auc = float(np.mean(scores))
        except Exception as e:
            log_warning(f"[DriftDetector] 对抗验证失败: {e}，回退到 LogisticRegression")
            estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
            scores = cross_val_score(estimator, X_combined, y_combined, cv=3,
                                     scoring='roc_auc', n_jobs=-1)
            mean_auc = float(np.mean(scores))

        # AUC > 0.7 说明 train/test 很容易被区分，存在漂移
        # AUC ≈ 0.5 说明无法区分，分布一致
        is_drifted = mean_auc > (1.0 - self.threshold)
        # score 定义为漂移程度：0=无漂移, 1=严重漂移
        drift_score = max(0.0, (mean_auc - 0.5) * 2.0)

        return DriftReport(
            is_drifted=is_drifted,
            overall_score=drift_score,
            threshold=(1.0 - self.threshold - 0.5) * 2.0,
            method='adversarial',
            feature_scores={'mean_auc': mean_auc},
            details={'cv_auc_scores': scores.tolist(), 'mean_auc': mean_auc}
        )

    def _detect_wasserstein(self, X: pd.DataFrame) -> DriftReport:
        """Wasserstein Distance（最优传输距离）"""
        feature_scores = {}
        total_wd = 0.0
        valid_cols = 0

        for col in self._num_cols:
            ref_vals = self._reference[col].dropna()
            new_vals = X[col].dropna() if col in X.columns else pd.Series([], dtype=float)
            if len(ref_vals) == 0 or len(new_vals) == 0:
                continue

            # 标准化后计算 Wasserstein-1 距离
            ref_std = ref_vals.std() or 1.0
            wd = stats.wasserstein_distance(ref_vals / ref_std, new_vals / ref_std)
            feature_scores[col] = float(wd)
            total_wd += wd
            valid_cols += 1

        avg_wd = total_wd / valid_cols if valid_cols > 0 else 0.0
        # 启发式阈值：平均标准化 Wasserstein 距离 > 0.2 认为漂移
        is_drifted = avg_wd > 0.2

        return DriftReport(
            is_drifted=is_drifted,
            overall_score=avg_wd,
            threshold=0.2,
            method='wasserstein',
            feature_scores=feature_scores,
            details={'avg_wd': avg_wd, 'valid_cols': valid_cols}
        )

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def detect_drift(X_train: Union[pd.DataFrame, np.ndarray],
                 X_test: Union[pd.DataFrame, np.ndarray],
                 method: str = 'auto',
                 threshold: float = 0.05) -> DriftReport:
    """
    便捷函数：一键检测 train/test 漂移

    用法：
        report = detect_drift(X_train, X_test, method='auto')
        if report.is_drifted:
            print(f"⚠️ 数据漂移 detected! score={report.overall_score:.4f}")
        else:
            print(f"✅ 分布一致")
    """
    detector = DriftDetector(method=method, threshold=threshold)
    detector.fit_reference(X_train)
    return detector.detect(X_test)
