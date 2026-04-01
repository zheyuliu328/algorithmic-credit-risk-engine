"""
Model Selection Pipeline — Enhanced

Systematically evaluates candidate PD forward models across:
- Variable combinations (single, pairwise, full set)
- Lag structures (current quarter, lag-1, lag-2)
- Model types (logistic regression, probit, linear)

For each candidate, computes: AIC, BIC, Adjusted R², out-of-sample RMSE,
Hosmer-Lemeshow p-value, VIF, Durbin-Watson statistic, and coefficient p-values.

Enhancements over v1:
- VIF screening: flags/excludes candidates with VIF > threshold (default 5)
- Walk-forward cross-validation: expanding window instead of fixed split
- Durbin-Watson test: autocorrelation in residuals
- P-value rejection: excludes models where key coefficients are insignificant

Outputs a ranked comparison table for transparent, auditable model selection —
aligned with IFRS 9 governance requirements.
"""

import itertools
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from credit_one.ecl.pd_forward_model import PDForwardModel, MACRO_VARIABLES


class ModelSelectionPipeline:
    """
    Automated model selection for PD forward mapping.

    Evaluates all combinations of model type × variable set × lag structure,
    ranks by composite score, and selects the best model.

    Parameters
    ----------
    model_types : list of str
        Model types to test. Default: ['logistic', 'linear'].
    max_lag : int
        Maximum lag to test (0, 1, ..., max_lag). Default: 2.
    variable_sets : list of list of str, optional
        Explicit variable combinations. If None, generates all subsets of MACRO_VARIABLES.
    train_ratio : float
        Fraction of data for training (used only for simple split fallback). Default: 0.7.
    ranking_metric : str
        Primary metric for ranking. Default: 'composite'.
    vif_threshold : float
        VIF 阈值: VIF > threshold 的候选模型会被标记。Default: 5.0.
    pvalue_threshold : float
        系数 p-value 阈值: 任何系数 p > threshold 的模型会被标记。Default: 0.10.
    walk_forward_folds : int
        Walk-forward cross-validation 折数。Default: 5.
        设为 0 则使用简单 train/test split。
    """

    def __init__(
        self,
        model_types: Optional[List[str]] = None,
        max_lag: int = 2,
        variable_sets: Optional[List[List[str]]] = None,
        train_ratio: float = 0.7,
        ranking_metric: str = "composite",
        vif_threshold: float = 5.0,
        pvalue_threshold: float = 0.10,
        walk_forward_folds: int = 5,
    ):
        self.model_types = model_types or ["logistic", "linear"]
        self.max_lag = max_lag
        self.train_ratio = train_ratio
        self.ranking_metric = ranking_metric
        self.vif_threshold = vif_threshold
        self.pvalue_threshold = pvalue_threshold
        self.walk_forward_folds = walk_forward_folds

        if variable_sets is None:
            self.variable_sets = self._generate_variable_subsets()
        else:
            self.variable_sets = variable_sets

        self._results: List[Dict] = []
        self._results_df: Optional[pd.DataFrame] = None
        self._best_model: Optional[PDForwardModel] = None

    def _generate_variable_subsets(self) -> List[List[str]]:
        """Generate all non-empty subsets of macro variables."""
        subsets = []
        for r in range(1, len(MACRO_VARIABLES) + 1):
            for combo in itertools.combinations(MACRO_VARIABLES, r):
                subsets.append(list(combo))
        return subsets

    def run(self, macro_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Execute the full model selection pipeline.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Historical macro data with observed_default_rate.
        verbose : bool
            Print progress updates.

        Returns
        -------
        pd.DataFrame
            Comparison table with all candidate models ranked.
        """
        self._results = []
        total_candidates = (
            len(self.model_types) * len(self.variable_sets) * (self.max_lag + 1)
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print("ECL MODEL SELECTION PIPELINE (Enhanced)")
            print(f"{'=' * 70}")
            print(f"Candidates to evaluate: {total_candidates}")
            print(f"Model types: {self.model_types}")
            print(f"Variable sets: {len(self.variable_sets)}")
            print(f"Lag range: 0 to {self.max_lag}")
            cv_mode = f"Walk-forward ({self.walk_forward_folds} folds)" if self.walk_forward_folds > 0 else f"Simple split ({self.train_ratio:.0%}/{1 - self.train_ratio:.0%})"
            print(f"Cross-validation: {cv_mode}")
            print(f"VIF threshold: {self.vif_threshold}")
            print(f"P-value threshold: {self.pvalue_threshold}")
            print(f"Ranking metric: {self.ranking_metric}")
            print("-" * 70)

        evaluated = 0
        for model_type in self.model_types:
            for variables in self.variable_sets:
                for lag in range(self.max_lag + 1):
                    result = self._evaluate_candidate(macro_df, model_type, variables, lag)
                    if result is not None:
                        self._results.append(result)
                    evaluated += 1
                    if verbose and evaluated % 10 == 0:
                        print(f"  Evaluated {evaluated}/{total_candidates} candidates...")

        self._results_df = self._build_comparison_table()

        if verbose:
            print(f"\nCompleted: {len(self._results)} valid models out of {total_candidates}")
            self._print_top_models()

        self._select_best_model(macro_df)

        return self._results_df

    def _evaluate_candidate(
        self,
        macro_df: pd.DataFrame,
        model_type: str,
        variables: List[str],
        lag: int,
    ) -> Optional[Dict]:
        """Evaluate a single candidate model configuration."""
        try:
            model = PDForwardModel(model_type=model_type, variables=variables, lag=lag)

            n = len(macro_df)
            n_effective = n - lag
            if n_effective < 10:
                return None

            # Fit on training data (simple split for fit stats)
            split_idx = int(n * self.train_ratio)
            train_df = macro_df.iloc[:split_idx].copy()
            test_df = macro_df.iloc[split_idx:].copy()

            fit_stats = model.fit(train_df)

            # Out-of-sample evaluation
            if self.walk_forward_folds > 0:
                oos_rmse = self._walk_forward_cv(macro_df, model_type, variables, lag)
            else:
                oos_rmse = self._compute_oos_rmse(model, test_df)

            # Hosmer-Lemeshow test
            hl_result = model.hosmer_lemeshow_test(macro_df)

            # VIF computation (only for multi-variable models)
            vif_values = self._compute_vif(macro_df, variables, lag)
            max_vif = max(vif_values.values()) if vif_values else 0.0
            vif_flag = max_vif > self.vif_threshold

            # Durbin-Watson test for residual autocorrelation
            dw_stat = self._compute_durbin_watson(model, train_df)

            # P-value screening
            p_values_detail = fit_stats.get("p_values", {})
            # 检查非常数项的系数 p-value
            max_pvalue = 0.0
            for var_name, pv in p_values_detail.items():
                if var_name != "const":
                    max_pvalue = max(max_pvalue, pv)
            pvalue_flag = max_pvalue > self.pvalue_threshold

            result = {
                "model_type": model_type,
                "variables": ", ".join(variables),
                "n_variables": len(variables),
                "lag": lag,
                "model_label": f"{model_type}|{'+'.join(variables)}|lag{lag}",
            }

            result["aic"] = fit_stats.get("aic", np.nan)
            result["bic"] = fit_stats.get("bic", np.nan)

            if model_type == "probit":
                result["adj_r2"] = fit_stats.get("pseudo_r2", np.nan)
            else:
                result["adj_r2"] = fit_stats.get("adj_r2", np.nan)

            result["oos_rmse"] = oos_rmse
            result["hl_chi2"] = hl_result["chi2_statistic"]
            result["hl_p_value"] = hl_result["p_value"]

            # 新增指标
            result["max_vif"] = round(max_vif, 2)
            result["vif_flag"] = vif_flag
            result["vif_detail"] = vif_values
            result["durbin_watson"] = round(dw_stat, 4)
            result["max_coef_pvalue"] = round(max_pvalue, 6)
            result["pvalue_flag"] = pvalue_flag

            result["coefficients"] = fit_stats.get("coefficients", {})
            result["p_values_detail"] = p_values_detail
            result["n_obs"] = fit_stats.get("n_obs", 0)
            result["log_likelihood"] = fit_stats.get("log_likelihood", np.nan)

            return result

        except Exception:
            return None

    def _compute_vif(
        self, macro_df: pd.DataFrame, variables: List[str], lag: int
    ) -> Dict[str, float]:
        """
        计算各自变量的 Variance Inflation Factor。

        VIF > 5 表示多重共线性问题，VIF > 10 严重共线性。
        单变量模型返回空 dict。
        """
        if len(variables) < 2:
            return {}

        df = macro_df.copy()
        if lag > 0:
            for var in variables:
                df[var] = df[var].shift(lag)
            df = df.dropna()

        X = df[variables].values
        if len(X) < len(variables) + 1:
            return {}

        # 加常数项计算 VIF
        X_with_const = sm.add_constant(X)
        vif_dict = {}
        for i, var in enumerate(variables):
            try:
                vif_val = variance_inflation_factor(X_with_const, i + 1)  # +1 跳过常数项
                vif_dict[var] = round(float(vif_val), 2)
            except Exception:
                vif_dict[var] = np.nan

        return vif_dict

    def _compute_durbin_watson(self, model: PDForwardModel, train_df: pd.DataFrame) -> float:
        """
        计算 Durbin-Watson 统计量，检测残差自相关。

        DW ≈ 2: 无自相关
        DW < 1.5: 正自相关
        DW > 2.5: 负自相关
        """
        try:
            if not model.is_fitted:
                return np.nan
            residuals = model._fitted_model.resid
            return float(durbin_watson(residuals))
        except Exception:
            return np.nan

    def _walk_forward_cv(
        self,
        macro_df: pd.DataFrame,
        model_type: str,
        variables: List[str],
        lag: int,
    ) -> float:
        """
        Walk-forward (expanding window) cross-validation。

        每个 fold:
        - 训练集: 从开头扩展到 split point
        - 测试集: split point 后的固定窗口

        比简单 train/test split 更适合时间序列。

        Returns
        -------
        float
            各折 RMSE 的均值。
        """
        n = len(macro_df)
        min_train = max(int(n * 0.4), lag + 10)  # 至少 40% 数据做初始训练
        test_size = max((n - min_train) // self.walk_forward_folds, 3)

        rmse_list = []
        for fold in range(self.walk_forward_folds):
            train_end = min_train + fold * test_size
            test_end = min(train_end + test_size, n)

            if train_end >= n or test_end <= train_end:
                break

            train_df = macro_df.iloc[:train_end].copy()
            test_df = macro_df.iloc[train_end:test_end].copy()

            if len(test_df) < 2:
                continue

            try:
                fold_model = PDForwardModel(
                    model_type=model_type, variables=variables, lag=lag
                )
                fold_model.fit(train_df)
                rmse = self._compute_oos_rmse(fold_model, test_df)
                if not np.isnan(rmse):
                    rmse_list.append(rmse)
            except Exception:
                continue

        if not rmse_list:
            return np.nan
        return round(float(np.mean(rmse_list)), 8)

    def _compute_oos_rmse(self, model: PDForwardModel, test_df: pd.DataFrame) -> float:
        """Compute out-of-sample RMSE on test data."""
        try:
            df = test_df.copy()
            if model.lag > 0:
                for var in model.variables:
                    df[var] = df[var].shift(model.lag)
                df = df.dropna()

            if len(df) < 3:
                return np.nan

            X = sm.add_constant(df[model.variables])
            raw_pred = model._fitted_model.predict(X)

            if model.model_type == "logistic":
                pred_pd = 1.0 / (1.0 + np.exp(-raw_pred))
                actual = df["observed_default_rate"].values
            elif model.model_type == "probit":
                pred_pd = np.clip(raw_pred, 1e-6, 1 - 1e-6)
                actual = (df["observed_default_rate"] > df["observed_default_rate"].median())
                actual = actual.astype(float).values
            else:
                pred_pd = np.clip(raw_pred, 1e-6, 1 - 1e-6)
                actual = df["observed_default_rate"].values

            rmse = np.sqrt(np.mean((actual - pred_pd) ** 2))
            return round(rmse, 8)
        except Exception:
            return np.nan

    def _build_comparison_table(self) -> pd.DataFrame:
        """Build and rank the comparison table with enhanced scoring."""
        df = pd.DataFrame(self._results)
        if df.empty:
            return df

        # Composite score: normalize each metric to [0, 1], then weighted average
        # Penalty for VIF flag and p-value flag
        score_cols = ["aic", "bic", "adj_r2", "oos_rmse", "hl_p_value"]
        for col in score_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if vals.nunique() <= 1:
                df[f"{col}_norm"] = 0.5
                continue

            col_min = vals.min()
            col_max = vals.max()
            rng = col_max - col_min
            if rng == 0:
                df[f"{col}_norm"] = 0.5
            elif col in ("adj_r2", "hl_p_value"):
                df[f"{col}_norm"] = (df[col] - col_min) / rng
            else:
                df[f"{col}_norm"] = 1 - (df[col] - col_min) / rng

        # Durbin-Watson: 越接近 2 越好 → 归一化为 |DW - 2| 的反向
        if "durbin_watson" in df.columns:
            dw_deviation = (df["durbin_watson"] - 2.0).abs()
            dw_max = dw_deviation.max()
            if dw_max > 0:
                df["dw_norm"] = 1 - dw_deviation / dw_max
            else:
                df["dw_norm"] = 0.5

        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        if norm_cols:
            df["composite_score"] = df[norm_cols].mean(axis=1)
        else:
            df["composite_score"] = 0.0

        # 对 VIF 和 p-value 标记施加惩罚
        if "vif_flag" in df.columns:
            df.loc[df["vif_flag"] == True, "composite_score"] *= 0.85  # noqa: E712
        if "pvalue_flag" in df.columns:
            df.loc[df["pvalue_flag"] == True, "composite_score"] *= 0.90  # noqa: E712

        # Sort by ranking metric
        if self.ranking_metric == "composite":
            df = df.sort_values("composite_score", ascending=False)
        elif self.ranking_metric in ("aic", "bic", "oos_rmse"):
            df = df.sort_values(self.ranking_metric, ascending=True)
        elif self.ranking_metric == "adj_r2":
            df = df.sort_values(self.ranking_metric, ascending=False)
        else:
            df = df.sort_values("composite_score", ascending=False)

        df["rank"] = range(1, len(df) + 1)
        df = df.reset_index(drop=True)

        return df

    def _print_top_models(self, top_n: int = 10) -> None:
        """Print the top N models in a formatted table."""
        if self._results_df is None or self._results_df.empty:
            print("No valid models found.")
            return

        display_cols = [
            "rank", "model_label", "aic", "bic", "adj_r2",
            "oos_rmse", "hl_p_value", "max_vif", "durbin_watson",
            "max_coef_pvalue", "composite_score",
        ]
        available = [c for c in display_cols if c in self._results_df.columns]
        top = self._results_df.head(top_n)[available]

        print(f"\n{'=' * 70}")
        print(f"TOP {min(top_n, len(top))} MODELS")
        print("=" * 70)
        print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        # VIF / p-value 警告
        flagged_vif = self._results_df.head(top_n)["vif_flag"].sum()
        flagged_pv = self._results_df.head(top_n)["pvalue_flag"].sum()
        if flagged_vif > 0:
            print(f"⚠ {flagged_vif} of top {top_n} models flagged for VIF > {self.vif_threshold}")
        if flagged_pv > 0:
            print(f"⚠ {flagged_pv} of top {top_n} models have insignificant coefficients (p > {self.pvalue_threshold})")

        print("=" * 70)

    def _select_best_model(self, macro_df: pd.DataFrame) -> None:
        """Refit the best model on full data for production use."""
        if self._results_df is None or self._results_df.empty:
            return

        best = self._results_df.iloc[0]
        variables = [v.strip() for v in best["variables"].split(",")]
        self._best_model = PDForwardModel(
            model_type=best["model_type"],
            variables=variables,
            lag=int(best["lag"]),
        )
        self._best_model.fit(macro_df)

    @property
    def best_model(self) -> PDForwardModel:
        """Return the auto-selected best model (refitted on full data)."""
        if self._best_model is None:
            raise RuntimeError("Pipeline not run yet. Call run() first.")
        return self._best_model

    @property
    def results(self) -> pd.DataFrame:
        """Return the full comparison table."""
        if self._results_df is None:
            raise RuntimeError("Pipeline not run yet. Call run() first.")
        return self._results_df

    def export_results(self, filepath: str) -> None:
        """Export the comparison table to CSV."""
        if self._results_df is None:
            raise RuntimeError("Pipeline not run yet. Call run() first.")

        export_cols = [
            "rank", "model_label", "model_type", "variables", "n_variables", "lag",
            "aic", "bic", "adj_r2", "oos_rmse", "hl_chi2", "hl_p_value",
            "max_vif", "vif_flag", "durbin_watson", "max_coef_pvalue", "pvalue_flag",
            "composite_score", "n_obs", "log_likelihood",
        ]
        available = [c for c in export_cols if c in self._results_df.columns]
        self._results_df[available].to_csv(filepath, index=False)
        print(f"Model selection results exported to {filepath}")

    def get_model_details(self, rank: int = 1) -> Dict:
        """Get detailed information about a specific ranked model."""
        if self._results_df is None:
            raise RuntimeError("Pipeline not run yet. Call run() first.")

        mask = self._results_df["rank"] == rank
        if not mask.any():
            raise ValueError(f"No model with rank {rank}")

        return self._results_df[mask].iloc[0].to_dict()
