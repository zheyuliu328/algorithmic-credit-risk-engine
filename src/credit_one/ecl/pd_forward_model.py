"""
PD Forward Mapping Model

Maps through-the-cycle PD to forward-looking PD using macro-economic variables.
Supports logistic regression, probit regression, and OLS models with configurable
variable sets and lag structures.

Core formula (logistic):
    logit(PD_t) = β₀ + β₁·GDP_t + β₂·Unemployment_t + β₃·Interest_Rate_t

The fitted model converts a base PD estimate into a conditional PD given
a specific macro-economic scenario.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit, Probit

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# All available macro variable names
MACRO_VARIABLES = ["gdp_growth", "unemployment_rate", "interest_rate"]


class PDForwardModel:
    """
    Regression model mapping macro variables to default rates.

    Parameters
    ----------
    model_type : str
        One of 'logistic', 'probit', 'linear'.
    variables : list of str
        Macro variable names to include (subset of MACRO_VARIABLES).
    lag : int
        Number of quarters to lag the macro variables (0 = current quarter).
    """

    def __init__(
        self,
        model_type: str = "logistic",
        variables: Optional[List[str]] = None,
        lag: int = 0,
    ):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for PDForwardModel. pip install statsmodels")

        if model_type not in ("logistic", "probit", "linear"):
            raise ValueError(f"model_type must be 'logistic', 'probit', or 'linear', got {model_type}")

        self.model_type = model_type
        self.variables = variables or list(MACRO_VARIABLES)
        self.lag = lag
        self._fitted_model = None
        self._fit_summary: Optional[Dict] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted_model is not None

    def _prepare_data(self, macro_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X (macro regressors) and y (default rate) with optional lag.

        Returns
        -------
        X : pd.DataFrame
            Macro variable columns with constant, lagged if specified.
        y : pd.Series
            Observed default rate (logit-transformed for logistic/probit).
        """
        df = macro_df.copy()

        # Apply lag
        if self.lag > 0:
            for var in self.variables:
                df[var] = df[var].shift(self.lag)
            df = df.dropna().reset_index(drop=True)

        X = sm.add_constant(df[self.variables])
        y = df["observed_default_rate"]
        return X, y

    def fit(self, macro_df: pd.DataFrame) -> Dict:
        """
        Fit the forward PD model on historical macro data.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Must contain columns for each variable in self.variables plus
            'observed_default_rate'.

        Returns
        -------
        dict
            Fit statistics: coefficients, aic, bic, adj_r2, log_likelihood.
        """
        X, y = self._prepare_data(macro_df)

        if self.model_type == "logistic":
            # Transform y to logit scale, then OLS (two-step approach for simplicity)
            y_logit = np.log(np.clip(y, 1e-6, 1 - 1e-6) / (1 - np.clip(y, 1e-6, 1 - 1e-6)))
            model = sm.OLS(y_logit, X)
            self._fitted_model = model.fit()
        elif self.model_type == "probit":
            # Binary threshold approach: convert default rate to binary outcome for probit
            # Use median as threshold for binary classification
            y_binary = (y > y.median()).astype(int)
            model = Probit(y_binary, X)
            self._fitted_model = model.fit(disp=0)
        else:  # linear
            model = sm.OLS(y, X)
            self._fitted_model = model.fit()

        self._fit_summary = self._extract_fit_stats(X, y)
        return self._fit_summary

    def _extract_fit_stats(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Extract model fit statistics for comparison."""
        result = self._fitted_model
        n = len(y)
        k = len(self.variables) + 1  # +1 for constant

        stats: Dict = {
            "model_type": self.model_type,
            "variables": list(self.variables),
            "lag": self.lag,
            "n_obs": n,
            "coefficients": dict(zip(result.params.index, np.round(result.params.values, 6))),
            "p_values": dict(zip(result.pvalues.index, np.round(result.pvalues.values, 6))),
        }

        if self.model_type in ("logistic", "linear"):
            stats["aic"] = round(result.aic, 4)
            stats["bic"] = round(result.bic, 4)
            stats["adj_r2"] = round(result.rsquared_adj, 6)
            stats["r2"] = round(result.rsquared, 6)
            stats["log_likelihood"] = round(result.llf, 4)
        elif self.model_type == "probit":
            stats["aic"] = round(result.aic, 4)
            stats["bic"] = round(result.bic, 4)
            stats["pseudo_r2"] = round(result.prsquared, 6)
            stats["log_likelihood"] = round(result.llf, 4)

        return stats

    def predict(self, macro_scenario: pd.DataFrame, base_pd: Optional[float] = None) -> np.ndarray:
        """
        Predict forward-looking PD given macro scenario forecasts.

        Parameters
        ----------
        macro_scenario : pd.DataFrame
            Future macro variable values (one row per projection quarter).
        base_pd : float, optional
            Through-the-cycle PD from CreditOne scoring pipeline.
            If provided, used as anchor for the level shift.

        Returns
        -------
        np.ndarray
            Forward PD for each quarter in the scenario.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = sm.add_constant(macro_scenario[self.variables])
        raw_pred = self._fitted_model.predict(X)

        if self.model_type == "logistic":
            # raw_pred is on logit scale, convert back to probability
            forward_pd = 1.0 / (1.0 + np.exp(-raw_pred))
        elif self.model_type == "probit":
            # Probit predict already returns probability
            forward_pd = np.clip(raw_pred, 1e-6, 1 - 1e-6)
        else:
            # Linear: clip to valid probability range
            forward_pd = np.clip(raw_pred, 1e-6, 1 - 1e-6)

        # Anchor to base PD if provided: shift the level while preserving term structure shape
        if base_pd is not None:
            mean_pred = forward_pd.mean()
            if mean_pred > 0:
                scale_factor = base_pd / mean_pred
                forward_pd = np.clip(forward_pd * scale_factor, 1e-6, 1 - 1e-6)

        return np.round(forward_pd, 8)

    def hosmer_lemeshow_test(self, macro_df: pd.DataFrame, n_groups: int = 10) -> Dict:
        """
        Hosmer-Lemeshow goodness-of-fit test.

        Groups predictions into deciles and tests observed vs expected default rates.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Historical data with observed_default_rate.
        n_groups : int
            Number of groups for the test.

        Returns
        -------
        dict
            chi2_statistic, p_value, degrees_of_freedom.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X, y = self._prepare_data(macro_df)
        raw_pred = self._fitted_model.predict(X)

        if self.model_type == "logistic":
            pred_pd = 1.0 / (1.0 + np.exp(-raw_pred))
        elif self.model_type == "probit":
            pred_pd = np.clip(raw_pred, 1e-6, 1 - 1e-6)
        else:
            pred_pd = np.clip(raw_pred, 1e-6, 1 - 1e-6)

        # Sort by predicted and group
        order = np.argsort(pred_pd)
        y_sorted = np.array(y)[order]
        pred_sorted = np.array(pred_pd)[order]

        actual_groups = min(n_groups, len(y_sorted))
        groups = np.array_split(np.arange(len(y_sorted)), actual_groups)

        chi2 = 0.0
        for grp in groups:
            obs = y_sorted[grp].mean()
            exp = pred_sorted[grp].mean()
            n_g = len(grp)
            if exp > 0 and exp < 1:
                chi2 += n_g * (obs - exp) ** 2 / (exp * (1 - exp))

        dof = max(actual_groups - 2, 1)
        p_value = 1 - scipy_stats.chi2.cdf(chi2, dof)

        return {
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": dof,
            "n_groups": actual_groups,
        }

    def get_summary(self) -> str:
        """Return statsmodels summary as string."""
        if not self.is_fitted:
            return "Model not fitted."
        return str(self._fitted_model.summary())
