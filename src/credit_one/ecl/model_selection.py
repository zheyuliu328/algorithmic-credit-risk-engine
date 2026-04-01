"""
Model Selection Pipeline — Core Component

Systematically evaluates candidate PD forward models across:
- Variable combinations (single, pairwise, full set)
- Lag structures (current quarter, lag-1, lag-2)
- Model types (logistic regression, probit, linear)

For each candidate, computes: AIC, BIC, Adjusted R², out-of-sample RMSE,
and Hosmer-Lemeshow p-value. Outputs a ranked comparison table for
transparent, auditable model selection — aligned with IFRS 9 governance.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

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
        Fraction of data for training in train/test split. Default: 0.7.
    ranking_metric : str
        Primary metric for ranking. One of 'aic', 'bic', 'adj_r2', 'oos_rmse', 'composite'.
        Default: 'composite'.
    """

    def __init__(
        self,
        model_types: Optional[List[str]] = None,
        max_lag: int = 2,
        variable_sets: Optional[List[List[str]]] = None,
        train_ratio: float = 0.7,
        ranking_metric: str = "composite",
    ):
        self.model_types = model_types or ["logistic", "linear"]
        self.max_lag = max_lag
        self.train_ratio = train_ratio
        self.ranking_metric = ranking_metric

        # Generate all non-empty subsets of MACRO_VARIABLES if not specified
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
            print("ECL MODEL SELECTION PIPELINE")
            print(f"{'=' * 70}")
            print(f"Candidates to evaluate: {total_candidates}")
            print(f"Model types: {self.model_types}")
            print(f"Variable sets: {len(self.variable_sets)}")
            print(f"Lag range: 0 to {self.max_lag}")
            print(f"Train/Test split: {self.train_ratio:.0%} / {1 - self.train_ratio:.0%}")
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

        # Auto-select best model
        self._select_best_model(macro_df)

        return self._results_df

    def _evaluate_candidate(
        self,
        macro_df: pd.DataFrame,
        model_type: str,
        variables: List[str],
        lag: int,
    ) -> Optional[Dict]:
        """
        Evaluate a single candidate model configuration.

        Returns None if the model fails to fit (e.g., insufficient data after lagging).
        """
        try:
            model = PDForwardModel(model_type=model_type, variables=variables, lag=lag)

            # Train/test split (temporal: first N% train, rest test)
            n = len(macro_df)
            n_effective = n - lag  # usable rows after lagging
            if n_effective < 10:
                return None

            split_idx = int(n * self.train_ratio)
            train_df = macro_df.iloc[:split_idx].copy()
            test_df = macro_df.iloc[split_idx:].copy()

            # Fit on training data
            fit_stats = model.fit(train_df)

            # Out-of-sample prediction and RMSE
            oos_rmse = self._compute_oos_rmse(model, test_df)

            # Hosmer-Lemeshow test
            hl_result = model.hosmer_lemeshow_test(macro_df)

            # Build result record
            result = {
                "model_type": model_type,
                "variables": ", ".join(variables),
                "n_variables": len(variables),
                "lag": lag,
                "model_label": f"{model_type}|{'+'.join(variables)}|lag{lag}",
            }

            # Information criteria (lower is better)
            result["aic"] = fit_stats.get("aic", np.nan)
            result["bic"] = fit_stats.get("bic", np.nan)

            # Goodness of fit (higher is better)
            if model_type == "probit":
                result["adj_r2"] = fit_stats.get("pseudo_r2", np.nan)
            else:
                result["adj_r2"] = fit_stats.get("adj_r2", np.nan)

            # Out-of-sample performance (lower is better)
            result["oos_rmse"] = oos_rmse

            # Hosmer-Lemeshow (higher p-value = better calibration)
            result["hl_chi2"] = hl_result["chi2_statistic"]
            result["hl_p_value"] = hl_result["p_value"]

            # Coefficient details
            result["coefficients"] = fit_stats.get("coefficients", {})
            result["p_values_detail"] = fit_stats.get("p_values", {})
            result["n_obs"] = fit_stats.get("n_obs", 0)
            result["log_likelihood"] = fit_stats.get("log_likelihood", np.nan)

            return result

        except Exception:
            # Skip models that fail to converge or have numerical issues
            return None

    def _compute_oos_rmse(self, model: PDForwardModel, test_df: pd.DataFrame) -> float:
        """Compute out-of-sample RMSE on test data."""
        try:
            import statsmodels.api as sm

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
        """Build and rank the comparison table."""
        df = pd.DataFrame(self._results)
        if df.empty:
            return df

        # Compute composite score (normalize each metric to [0, 1], then average)
        # Lower is better for: AIC, BIC, OOS_RMSE
        # Higher is better for: adj_R2, HL p-value
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
                # Higher is better → normalize directly
                df[f"{col}_norm"] = (df[col] - col_min) / rng
            else:
                # Lower is better → invert
                df[f"{col}_norm"] = 1 - (df[col] - col_min) / rng

        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        if norm_cols:
            df["composite_score"] = df[norm_cols].mean(axis=1)
        else:
            df["composite_score"] = 0.0

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
            "oos_rmse", "hl_p_value", "composite_score",
        ]
        available = [c for c in display_cols if c in self._results_df.columns]
        top = self._results_df.head(top_n)[available]

        print(f"\n{'=' * 70}")
        print(f"TOP {min(top_n, len(top))} MODELS")
        print("=" * 70)
        print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
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
        """
        Export the comparison table to CSV.

        Parameters
        ----------
        filepath : str
            Output CSV path.
        """
        if self._results_df is None:
            raise RuntimeError("Pipeline not run yet. Call run() first.")

        # Drop internal columns before export
        export_cols = [
            "rank", "model_label", "model_type", "variables", "n_variables", "lag",
            "aic", "bic", "adj_r2", "oos_rmse", "hl_chi2", "hl_p_value",
            "composite_score", "n_obs", "log_likelihood",
        ]
        available = [c for c in export_cols if c in self._results_df.columns]
        self._results_df[available].to_csv(filepath, index=False)
        print(f"Model selection results exported to {filepath}")

    def get_model_details(self, rank: int = 1) -> Dict:
        """
        Get detailed information about a specific ranked model.

        Parameters
        ----------
        rank : int
            Rank of the model (1 = best).

        Returns
        -------
        dict
            Full model details including coefficients and p-values.
        """
        if self._results_df is None:
            raise RuntimeError("Pipeline not run yet. Call run() first.")

        mask = self._results_df["rank"] == rank
        if not mask.any():
            raise ValueError(f"No model with rank {rank}")

        return self._results_df[mask].iloc[0].to_dict()
