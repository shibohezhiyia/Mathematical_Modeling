"""
高级统计分析模块

支持主成分分析(PCA)、多元描述统计、相关性分析、
方差分析(ANOVA)、卡方检验、异常值检测等。
"""

import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from utils.helpers import log_info

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DescriptiveResult:
    """描述统计结果"""
    column: str
    dtype: str
    count: int
    missing: int
    missing_rate: float
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    median: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    ci_lower: Optional[float]  # 95% 置信区间下限
    ci_upper: Optional[float]  # 95% 置信区间上限
    cv: Optional[float] = None           # 变异系数
    range_val: Optional[float] = None    # 极差
    iqr: Optional[float] = None          # 四分位距
    distribution: Optional[str] = None   # 分布形态判断


@dataclass
class PCAResult:
    """PCA分析结果"""
    n_components: int
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    components: List[Dict[str, float]]  # 每个主成分的特征权重
    scores: Optional[List[List[float]]]  # 样本得分（前3个主成分）
    feature_names: List[str]
    original_features: int
    interpretation: Optional[str] = None  # 结果解释


@dataclass
class FactorAnalysisResult:
    """因子分析结果"""
    n_factors: int
    n_samples: int
    n_features: int
    method: str
    rotation: Optional[str]
    kmo: Optional[float]          # KMO检验值
    kmo_acceptable: bool
    bartlett_chi2: Optional[float] # Bartlett检验χ²
    bartlett_pvalue: Optional[float]
    bartlett_significant: bool
    loadings: List[Dict[str, Any]]  # 因子载荷矩阵
    communalities: Dict[str, float] # 共同度
    variance_explained: List[Dict[str, Any]]  # 方差解释
    rotated: bool
    scores: Optional[List[List[float]]]  # 因子得分
    interpretation: Optional[str] = None  # 适用性判断


@dataclass
class CorrelationResult:
    """相关性分析结果"""
    method: str
    matrix: Dict[str, Dict[str, float]]
    pairs: List[Dict[str, Any]]  # 显著的相关对


@dataclass
class ANOVAResult:
    """方差分析结果"""
    factor: str
    target: str
    f_statistic: float
    p_value: float
    significant: bool
    group_stats: List[Dict[str, Any]]
    eta_squared: Optional[float] = None
    effect_size: Optional[str] = None


@dataclass
class OutlierResult:
    """异常值检测结果"""
    method: str
    column: str
    total: int
    outlier_count: int
    outlier_rate: float
    outliers: List[Dict[str, Any]]  # 前50个异常值
    bounds: Dict[str, float]


class AdvancedAnalytics:
    """高级统计分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ------------------------------------------------------------------
    # 1. 描述统计（扩展版）
    # ------------------------------------------------------------------
    def descriptive_stats(self, columns: Optional[List[str]] = None) -> List[DescriptiveResult]:
        """计算扩展描述统计"""
        cols = columns or self.numeric_cols
        results = []
        log_info(f"开始描述统计: 列数={len(cols)}", category="AdvancedAnalytics")
        for col in cols:
            if col not in self.df.columns:
                continue
            s = self.df[col].dropna()
            if len(s) == 0 or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            mean = float(s.mean())
            std = float(s.std())
            n = len(s)
            min_val = float(s.min())
            max_val = float(s.max())
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            # 95% 置信区间
            sem = stats.sem(s) if n > 1 else 0
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 and sem > 0 else (mean, mean)
            # 分布形态判断
            skew = float(s.skew()) if n > 2 else None
            kurt = float(s.kurtosis()) if n > 3 else None
            dist_parts = []
            if skew is not None:
                if skew > 1:
                    dist_parts.append("右偏")
                elif skew < -1:
                    dist_parts.append("左偏")
                else:
                    dist_parts.append("近似对称")
            if kurt is not None:
                if kurt > 3:
                    dist_parts.append("尖峰")
                elif kurt < -3:
                    dist_parts.append("平峰")
            distribution = "、".join(dist_parts) if dist_parts else None

            results.append(DescriptiveResult(
                column=col,
                dtype=str(self.df[col].dtype),
                count=int(n),
                missing=int(self.df[col].isnull().sum()),
                missing_rate=round(float(self.df[col].isnull().sum() / len(self.df)), 4),
                mean=round(mean, 6) if pd.notna(mean) else None,
                std=round(std, 6) if pd.notna(std) else None,
                min=round(min_val, 6),
                max=round(max_val, 6),
                median=round(float(s.median()), 6),
                q1=round(q1, 6),
                q3=round(q3, 6),
                skewness=round(skew, 4) if skew is not None else None,
                kurtosis=round(kurt, 4) if kurt is not None else None,
                ci_lower=round(float(ci[0]), 6) if ci[0] is not None else None,
                ci_upper=round(float(ci[1]), 6) if ci[1] is not None else None,
                cv=round(std / abs(mean), 4) if mean and mean != 0 else None,
                range_val=round(max_val - min_val, 4),
                iqr=round(q3 - q1, 4),
                distribution=distribution,
            ))
        log_info(f"描述统计完成: 有效列数={len(results)}", category="AdvancedAnalytics")
        return results

    # ------------------------------------------------------------------
    # 2. 主成分分析 (PCA)
    # ------------------------------------------------------------------
    def pca_analysis(self, n_components: Optional[int] = None) -> PCAResult:
        """PCA 主成分分析"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        if len(self.numeric_cols) < 2:
            raise ValueError("PCA 需要至少 2 个数值列")

        log_info(f"开始PCA分析: 请求组件数={n_components}", category="AdvancedAnalytics")
        df_num = self.df[self.numeric_cols].dropna()
        if len(df_num) < 2:
            raise ValueError("有效样本数不足，无法进行 PCA")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_num)

        original_features = len(self.numeric_cols)
        max_components = min(original_features, len(df_num) - 1)
        n_comp = n_components or min(5, max_components)
        n_comp = min(n_comp, max_components)

        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)

        evr = pca.explained_variance_ratio_.tolist()
        cumsum = np.cumsum(evr).tolist()

        # 主成分载荷（特征权重）
        components = []
        for i, comp in enumerate(pca.components_):
            weights = {name: round(float(w), 4) for name, w in zip(self.numeric_cols, comp)}
            # 按绝对值排序取Top贡献特征
            top_features = dict(sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:8])
            components.append({
                "pc": f"PC{i + 1}",
                "variance_ratio": round(float(evr[i]), 4),
                "features": top_features,
            })

        # 样本得分（前3个主成分，最多50个样本）
        scores = None
        if len(df_num) > 0:
            scores_3d = pca.transform(X_scaled)[:, :min(3, n_comp)].tolist()
            scores = scores_3d[:50]  # 限制返回数量

        # 生成解释文字
        total_var = cumsum[-1] if cumsum else 0
        interpretation = f"前{n_comp}个主成分累计解释{total_var:.1%}的方差"

        log_info(f"PCA完成: 组件数={n_comp}, 累计方差={total_var:.4f}", category="AdvancedAnalytics")
        return PCAResult(
            n_components=n_comp,
            explained_variance_ratio=[round(x, 4) for x in evr],
            cumulative_variance=[round(x, 4) for x in cumsum],
            components=components,
            scores=scores,
            feature_names=self.numeric_cols,
            original_features=original_features,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # 2.5 因子分析 (Factor Analysis)
    # ------------------------------------------------------------------
    def _kmo_test(self, corr_matrix: np.ndarray) -> Tuple[float, bool]:
        """计算KMO检验值（手动实现）"""
        n = corr_matrix.shape[0]
        if n < 3:
            return None, False
        # 反像相关矩阵
        inv_corr = np.linalg.inv(corr_matrix)
        partial_corr = np.zeros_like(inv_corr)
        for i in range(n):
            for j in range(n):
                if i != j:
                    partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
        np.fill_diagonal(partial_corr, 0)
        # KMO = sum(r_ij^2) / (sum(r_ij^2) + sum(u_ij^2))
        sum_r2 = np.sum(corr_matrix ** 2) - n  # 排除对角线
        sum_u2 = np.sum(partial_corr ** 2)
        if sum_r2 + sum_u2 == 0:
            return 0.0, False
        kmo = sum_r2 / (sum_r2 + sum_u2)
        return round(float(kmo), 4), bool(kmo >= 0.5)

    def _bartlett_test(self, corr_matrix: np.ndarray, n_samples: int) -> Tuple[float, float, bool]:
        """Bartlett球形检验"""
        n = corr_matrix.shape[0]
        if n < 2 or n_samples <= n:
            return None, None, False
        # 计算行列式
        det_r = np.linalg.det(corr_matrix)
        if det_r <= 0:
            det_r = 1e-10
        chi2 = -(n_samples - 1 - (2 * n + 5) / 6) * np.log(det_r)
        df = n * (n - 1) / 2
        p_value = 1 - stats.chi2.cdf(chi2, df) if chi2 > 0 else 1.0
        return round(float(chi2), 4), round(float(p_value), 6), bool(p_value < 0.05)

    def _varimax_rotation(self, loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Varimax因子旋转"""
        p, k = loadings.shape
        if k < 2:
            return loadings
        # 归一化
        h2 = np.sum(loadings ** 2, axis=1)
        normalization = np.sqrt(h2)
        normalization[normalization == 0] = 1
        L = loadings / normalization[:, np.newaxis]

        T = np.eye(k)
        for _ in range(max_iter):
            L2 = L ** 2
            L3 = L2 * L
            N = np.sum(L2, axis=0)
            A = L3 - L * N / p
            U, _, Vt = np.linalg.svd(A.T @ L)
            T_new = U @ Vt
            if np.max(np.abs(T_new - T)) < tol:
                break
            T = T_new
            L = loadings @ T

        return L

    def factor_analysis(self, n_factors: Optional[int] = None, rotation: str = "varimax") -> FactorAnalysisResult:
        """
        因子分析
        
        Args:
            n_factors: 因子数量，None则自动选择（特征值>1的因子数）
            rotation: 旋转方法，支持 'varimax' 和 'none'
        """
        log_info(f"开始因子分析: n_factors={n_factors}, rotation={rotation}", category="AdvancedAnalytics")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import FactorAnalysis as SklearnFA

        if len(self.numeric_cols) < 3:
            raise ValueError("因子分析需要至少 3 个数值列")

        df_num = self.df[self.numeric_cols].dropna()
        n_samples = len(df_num)
        n_features = len(self.numeric_cols)

        if n_samples < n_features + 10:
            raise ValueError(f"样本数({n_samples})不足，建议至少{n_features + 10}行")

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_num)

        # 相关性矩阵（用于检验）
        corr_matrix = np.corrcoef(X_scaled.T)

        # KMO和Bartlett检验
        kmo, kmo_ok = self._kmo_test(corr_matrix)
        bartlett_chi2, bartlett_p, bartlett_ok = self._bartlett_test(corr_matrix, n_samples)

        # 先用PCA估计因子数（特征值>1规则）
        from sklearn.decomposition import PCA
        pca_temp = PCA(n_components=n_features)
        pca_temp.fit(X_scaled)
        eigenvalues = pca_temp.explained_variance_
        suggested_factors = int(np.sum(eigenvalues > 1))
        suggested_factors = max(1, min(suggested_factors, n_features - 1))

        n_fact = n_factors or suggested_factors
        n_fact = max(1, min(n_fact, n_features - 1, n_samples - 1))

        # 因子提取
        fa = SklearnFA(n_components=n_fact, random_state=42, max_iter=1000)
        fa.fit(X_scaled)
        loadings_raw = fa.components_.T  # (n_features, n_factors)

        # 计算方差（基于因子载荷平方和）
        loadings_squared = loadings_raw ** 2
        sum_sq = np.sum(loadings_squared, axis=0)
        total_variance = np.sum(sum_sq)
        variance_ratio = (sum_sq / total_variance).tolist() if total_variance > 0 else [1.0 / n_fact] * n_fact

        # 旋转
        rotated = False
        if rotation == "varimax" and n_fact >= 2:
            loadings_rotated = self._varimax_rotation(loadings_raw)
            rotated = True
        else:
            loadings_rotated = loadings_raw

        # 因子载荷矩阵
        loadings_list = []
        for i, feat in enumerate(self.numeric_cols):
            row = {"feature": feat}
            for j in range(n_fact):
                row[f"F{j + 1}"] = round(float(loadings_rotated[i, j]), 4)
            # 找出最大载荷
            abs_loads = [abs(loadings_rotated[i, j]) for j in range(n_fact)]
            max_idx = int(np.argmax(abs_loads))
            row["dominant_factor"] = f"F{max_idx + 1}"
            row["dominant_loading"] = round(float(loadings_rotated[i, max_idx]), 4)
            loadings_list.append(row)

        # 共同度（每行载荷平方和）
        communalities = {}
        for i, feat in enumerate(self.numeric_cols):
            h2 = float(np.sum(loadings_rotated[i, :] ** 2))
            communalities[feat] = round(min(h2, 1.0), 4)

        # 方差解释
        variance_explained = []
        for j in range(n_fact):
            ss = float(np.sum(loadings_rotated[:, j] ** 2))
            variance_explained.append({
                "factor": f"F{j + 1}",
                "sum_of_squares": round(ss, 4),
                "proportion": round(float(variance_ratio[j]), 4),
                "cumulative": round(float(np.sum(variance_ratio[:j + 1])), 4),
            })

        # 因子得分（使用回归法估计）
        scores = None
        if n_samples > 0 and n_samples <= 200:
            try:
                # 简化得分估计: Z * R^-1 * L
                inv_corr = np.linalg.pinv(corr_matrix)
                weights = inv_corr @ loadings_rotated
                factor_scores = X_scaled @ weights
                scores = factor_scores[:50, :min(3, n_fact)].tolist()
            except Exception:
                # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
                pass

        # 生成适用性判断文字
        interpretation_parts = []
        if kmo is not None:
            if kmo >= 0.9:
                interpretation_parts.append(f"KMO={kmo}，非常适合因子分析")
            elif kmo >= 0.8:
                interpretation_parts.append(f"KMO={kmo}，适合因子分析")
            elif kmo >= 0.7:
                interpretation_parts.append(f"KMO={kmo}，一般适合因子分析")
            elif kmo >= 0.6:
                interpretation_parts.append(f"KMO={kmo}，不太适合因子分析")
            else:
                interpretation_parts.append(f"KMO={kmo}，不适合因子分析")
        if bartlett_p is not None:
            if bartlett_ok:
                interpretation_parts.append("Bartlett球形检验显著，变量间存在相关性")
            else:
                interpretation_parts.append("Bartlett球形检验不显著，变量间相关性弱")
        interpretation = "；".join(interpretation_parts)

        log_info(f"因子分析完成: 因子数={n_fact}, KMO={kmo}, Bartlett显著={bartlett_ok}", category="AdvancedAnalytics")
        return FactorAnalysisResult(
            n_factors=n_fact,
            n_samples=n_samples,
            n_features=n_features,
            method="FactorAnalysis",
            rotation=rotation if rotated else None,
            kmo=kmo,
            kmo_acceptable=kmo_ok if kmo is not None else False,
            bartlett_chi2=bartlett_chi2,
            bartlett_pvalue=bartlett_p,
            bartlett_significant=bool(bartlett_ok) if bartlett_p is not None else False,
            loadings=loadings_list,
            communalities=communalities,
            variance_explained=variance_explained,
            rotated=rotated,
            scores=scores,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # 3. 相关性分析
    # ------------------------------------------------------------------
    def correlation_analysis(self, method: str = "pearson", threshold: float = 0.5) -> CorrelationResult:
        """相关性矩阵分析"""
        if len(self.numeric_cols) < 2:
            raise ValueError("相关性分析需要至少 2 个数值列")

        log_info(f"开始相关性分析: method={method}, threshold={threshold}", category="AdvancedAnalytics")
        df_num = self.df[self.numeric_cols]
        corr = df_num.corr(method=method)

        # 构建矩阵字典
        matrix = {}
        for col in corr.columns:
            matrix[col] = {row: round(float(corr.loc[row, col]), 4) for row in corr.index}

        # 提取显著相关对（排除自身，取绝对值大于阈值）
        pairs = []
        seen = set()
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if i >= j:
                    continue
                val = float(corr.iloc[i, j])
                if abs(val) >= threshold:
                    pair_key = tuple(sorted([col1, col2]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        # 计算p值
                        try:
                            clean = df_num[[col1, col2]].dropna()
                            if len(clean) > 2:
                                _, pval = stats.pearsonr(clean[col1], clean[col2])
                            else:
                                pval = None
                        except Exception:
                            # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
                            pval = None
                        # 显著性标记
                        sig = ""
                        if pval is not None:
                            if pval < 0.001:
                                sig = "***"
                            elif pval < 0.01:
                                sig = "**"
                            elif pval < 0.05:
                                sig = "*"
                        pairs.append({
                            "col1": col1,
                            "col2": col2,
                            "correlation": round(val, 4),
                            "abs_correlation": round(abs(val), 4),
                            "p_value": round(float(pval), 6) if pval is not None else None,
                            "significance": sig,
                            "strength": "强" if abs(val) >= 0.8 else ("中" if abs(val) >= 0.5 else "弱"),
                        })

        # 按绝对值排序
        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        log_info(f"相关性分析完成: 发现 {len(pairs)} 对显著相关", category="AdvancedAnalytics")
        return CorrelationResult(
            method=method,
            matrix=matrix,
            pairs=pairs[:20],  # 限制返回数量
        )

    # ------------------------------------------------------------------
    # 4. 方差分析 (ANOVA)
    # ------------------------------------------------------------------
    def anova_analysis(self, factor: str, target: str) -> ANOVAResult:
        """单因素方差分析"""
        if factor not in self.df.columns:
            raise ValueError(f"因子列 '{factor}' 不存在")
        if target not in self.df.columns:
            raise ValueError(f"目标列 '{target}' 不存在")
        if not pd.api.types.is_numeric_dtype(self.df[target]):
            raise ValueError(f"目标列 '{target}' 必须是数值类型")

        log_info(f"开始方差分析: factor={factor}, target={target}", category="AdvancedAnalytics")
        groups = []
        group_stats = []
        for category, group_df in self.df.groupby(factor, observed=False):
            values = group_df[target].dropna()
            if len(values) > 0:
                groups.append(values)
                group_stats.append({
                    "group": str(category),
                    "count": int(len(values)),
                    "mean": round(float(values.mean()), 4),
                    "std": round(float(values.std()), 4),
                })

        if len(groups) < 2:
            raise ValueError("因子至少需要 2 个分组")

        f_stat, p_value = stats.f_oneway(*groups)

        # 计算 eta squared 效应量
        # eta_sq = SS_between / SS_total
        all_values = self.df[target].dropna()
        grand_mean = all_values.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum((x - grand_mean) ** 2 for x in all_values)
        eta_sq = round(float(ss_between / ss_total), 4) if ss_total > 0 else 0

        # 效应量解释
        if eta_sq < 0.01:
            effect_size = "可忽略"
        elif eta_sq < 0.06:
            effect_size = "小"
        elif eta_sq < 0.14:
            effect_size = "中"
        else:
            effect_size = "大"

        log_info(f"方差分析完成: F={f_stat:.4f}, p={p_value:.6f}, eta²={eta_sq}", category="AdvancedAnalytics")
        return ANOVAResult(
            factor=factor,
            target=target,
            f_statistic=round(float(f_stat), 4) if pd.notna(f_stat) else 0,
            p_value=round(float(p_value), 6) if pd.notna(p_value) else 1,
            significant=bool(p_value < 0.05) if pd.notna(p_value) else False,
            group_stats=group_stats,
            eta_squared=eta_sq,
            effect_size=effect_size,
        )

    # ------------------------------------------------------------------
    # 5. 卡方检验
    # ------------------------------------------------------------------
    def chi2_analysis(self, col1: str, col2: str) -> Dict[str, Any]:
        """两个分类变量的卡方独立性检验"""
        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise ValueError("指定的列不存在")

        log_info(f"开始卡方检验: {col1} x {col2}", category="AdvancedAnalytics")
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        # Cramer's V 效应量
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * min(contingency.shape) - 1)) if n > 0 and min(contingency.shape) > 1 else 0
        cramers_v = float(cramers_v)

        # 效应量解释
        if cramers_v < 0.1:
            effect = "可忽略"
        elif cramers_v < 0.3:
            effect = "小"
        elif cramers_v < 0.5:
            effect = "中"
        else:
            effect = "大"

        # 限制列联表大小，避免无意义的大矩阵拖垮前端
        max_table_dim = 30
        table_warning = None
        if contingency.shape[0] > max_table_dim and contingency.shape[1] > max_table_dim:
            table_warning = f"列联表维度({contingency.shape[0]}×{contingency.shape[1]})过大，仅展示前{max_table_dim}×{max_table_dim}"
            contingency_display = contingency.iloc[:max_table_dim, :max_table_dim]
        else:
            contingency_display = contingency

        log_info(f"卡方检验完成: chi2={chi2:.4f}, p={p:.6f}, CramersV={cramers_v:.4f}", category="AdvancedAnalytics")
        return {
            "col1": col1,
            "col2": col2,
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p), 6),
            "dof": int(dof),
            "significant": bool(p < 0.05),
            "cramers_v": round(cramers_v, 4),
            "effect_size": effect,
            "contingency_table": contingency_display.to_dict(),
            "table_shape": list(contingency.shape),
            "table_warning": table_warning,
        }

    # ------------------------------------------------------------------
    # 6. 异常值检测
    # ------------------------------------------------------------------
    def outlier_detection(self, column: str, method: str = "iqr") -> OutlierResult:
        """异常值检测（IQR或Z-Score方法）"""
        if column not in self.df.columns:
            raise ValueError(f"列 '{column}' 不存在")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"列 '{column}' 必须是数值类型")

        log_info(f"开始异常值检测: 列={column}, 方法={method}", category="AdvancedAnalytics")
        s = self.df[column].dropna()
        n = len(s)
        outliers = []
        bounds = {}

        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            bounds = {"lower": float(lower), "upper": float(upper), "q1": float(q1), "q3": float(q3)}
            mask = (s < lower) | (s > upper)
        elif method == "zscore":
            mean = s.mean()
            std = s.std()
            if std == 0:
                return OutlierResult(
                    method="zscore", column=column, total=n,
                    outlier_count=0, outlier_rate=0.0,
                    outliers=[], bounds={"mean": float(mean), "std": 0, "threshold": 3}
                )
            z_scores = np.abs((s - mean) / std)
            bounds = {"mean": float(mean), "std": float(std), "threshold": 3}
            mask = z_scores > 3
        else:
            raise ValueError("method 必须是 'iqr' 或 'zscore'")

        outlier_series = s[mask]
        outlier_indices = outlier_series.index.tolist()
        for idx, val in zip(outlier_indices[:50], outlier_series.values[:50]):
            outliers.append({"index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "value": float(val)})

        outlier_count = int(mask.sum())
        outlier_rate = round(outlier_count / n, 4) if n > 0 else 0
        log_info(f"异常值检测完成: 列={column}, 异常值={outlier_count}/{n}, 比率={outlier_rate}", category="AdvancedAnalytics")
        return OutlierResult(
            method=method,
            column=column,
            total=n,
            outlier_count=outlier_count,
            outlier_rate=outlier_rate,
            outliers=outliers,
            bounds=bounds,
        )

    # ------------------------------------------------------------------
    # 7. 综合摘要
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """返回数据集的综合统计摘要"""
        log_info("开始计算数据集综合摘要", category="AdvancedAnalytics")
        desc = self.df.describe().T
        desc_dict = {}
        for col in desc.index:
            row = desc.loc[col]
            desc_dict[col] = {
                k: round(float(v), 4) if pd.notna(v) and not isinstance(v, (pd.Timestamp, datetime.datetime)) else (v.isoformat() if isinstance(v, (pd.Timestamp, datetime.datetime)) else (None if pd.isna(v) else v))
                for k, v in row.items()
            }

        # 数据质量指标
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = int(self.df.isnull().sum().sum())
        missing_rate = round(total_missing / total_cells, 4) if total_cells > 0 else 0
        duplicate_rows = int(self.df.duplicated().sum())

        # 列类型统计
        type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "other": 0}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                type_counts["numeric"] += 1
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                type_counts["datetime"] += 1
            elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                type_counts["categorical"] += 1
            else:
                type_counts["other"] += 1

        # 分布概况
        distributions = {}
        for col in self.df.columns:
            if col in self.numeric_cols:
                s = self.df[col].dropna()
                distributions[col] = {
                    "type": "numeric",
                    "unique_count": int(s.nunique()),
                    "zero_count": int((s == 0).sum()),
                    "negative_count": int((s < 0).sum()),
                }
            else:
                vc = self.df[col].value_counts().head(5)
                distributions[col] = {
                    "type": "categorical",
                    "unique_count": int(self.df[col].nunique()),
                    "top_values": [{"value": str(v), "count": int(c)} for v, c in vc.items()],
                }

        log_info(f"综合摘要完成: 形状={self.df.shape}, 缺失率={missing_rate}", category="AdvancedAnalytics")
        return {
            "shape": list(self.df.shape),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "description": desc_dict,
            "dtypes": {col: str(self.df[col].dtype) for col in self.df.columns},
            "quality": {
                "total_cells": total_cells,
                "total_missing": total_missing,
                "missing_rate": missing_rate,
                "duplicate_rows": duplicate_rows,
                "duplicate_rate": round(duplicate_rows / self.df.shape[0], 4) if self.df.shape[0] > 0 else 0,
                "column_type_summary": type_counts,
            },
            "distributions": distributions,
        }
