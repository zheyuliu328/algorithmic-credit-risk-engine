"""
Probability-Weighted ECL Calculator

Combines scenario-specific forward PDs with scenario weights, LGD, and EAD
to produce IFRS 9 Expected Credit Loss estimates.

Outputs:
- 12-month ECL (Stage 1): PD over next 12 months × LGD × EAD
- Lifetime ECL (Stage 2): Cumulative PD over remaining life × LGD × EAD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from credit_one.ecl.pd_forward_model import PDForwardModel
from credit_one.ecl.scenario_engine import ScenarioEngine


class ECLCalculator:
    """
    IFRS 9 ECL calculator using probability-weighted multi-scenario forward PDs.

    Parameters
    ----------
    lgd : float
        Loss Given Default (fraction). Default: 0.45 (consistent with CreditOne pipeline).
    ead : float
        Exposure at Default (monetary). Default: 100000.0 (nominal portfolio exposure).
    discount_rate : float
        Annual discount rate for present value calculation. Default: 0.05.
    """

    def __init__(
        self,
        lgd: float = 0.45,
        ead: float = 100000.0,
        discount_rate: float = 0.05,
    ):
        self.lgd = lgd
        self.ead = ead
        self.discount_rate = discount_rate

    def compute_scenario_pds(
        self,
        model: PDForwardModel,
        scenario_engine: ScenarioEngine,
        base_pd: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Project forward PDs for each scenario using the fitted model.

        Parameters
        ----------
        model : PDForwardModel
            Fitted PD forward model.
        scenario_engine : ScenarioEngine
            Engine with generated scenarios.
        base_pd : float, optional
            Through-the-cycle PD from CreditOne scoring.

        Returns
        -------
        dict
            Mapping of scenario name to array of quarterly forward PDs.
        """
        scenario_pds = {}
        for name, scenario_df in scenario_engine.scenarios.items():
            pds = model.predict(scenario_df, base_pd=base_pd)
            scenario_pds[name] = pds
        return scenario_pds

    def compute_weighted_pd(
        self,
        scenario_pds: Dict[str, np.ndarray],
        scenario_engine: ScenarioEngine,
    ) -> np.ndarray:
        """
        Compute probability-weighted PD term structure.

        Parameters
        ----------
        scenario_pds : dict
            Scenario name to PD array.
        scenario_engine : ScenarioEngine
            For scenario weights.

        Returns
        -------
        np.ndarray
            Weighted PD for each quarter.
        """
        weighted_pd = None
        for name, pds in scenario_pds.items():
            weight = scenario_engine.get_weight(name)
            contribution = pds * weight
            if weighted_pd is None:
                weighted_pd = contribution
            else:
                weighted_pd = weighted_pd + contribution

        return weighted_pd

    def _quarterly_discount_factors(self, n_quarters: int) -> np.ndarray:
        """Compute quarterly discount factors from annual rate."""
        quarterly_rate = (1 + self.discount_rate) ** 0.25 - 1
        return np.array([(1 + quarterly_rate) ** (-t) for t in range(1, n_quarters + 1)])

    def _cumulative_pd(self, marginal_pds: np.ndarray) -> np.ndarray:
        """
        Convert marginal (quarterly) PDs to cumulative PDs.
        Cumulative PD_T = 1 - Π(1 - PD_t) for t = 1..T
        """
        survival = np.cumprod(1 - np.asarray(marginal_pds, dtype=float))
        return 1 - survival

    def compute_ecl(
        self,
        scenario_pds: Dict[str, np.ndarray],
        scenario_engine: ScenarioEngine,
        lgd: Optional[float] = None,
        ead: Optional[float] = None,
    ) -> Dict:
        """
        Compute 12-month and lifetime ECL.

        Parameters
        ----------
        scenario_pds : dict
            Scenario name to array of quarterly forward PDs.
        scenario_engine : ScenarioEngine
            For scenario weights.
        lgd : float, optional
            Override default LGD.
        ead : float, optional
            Override default EAD.

        Returns
        -------
        dict
            ecl_12m: Stage 1 ECL (12-month)
            ecl_lifetime: Stage 2 ECL (full horizon)
            weighted_pd_term_structure: array of weighted quarterly PDs
            cumulative_pd: array of cumulative PDs
            scenario_contributions: per-scenario ECL breakdown
        """
        lgd = lgd or self.lgd
        ead = ead or self.ead

        weighted_pd = self.compute_weighted_pd(scenario_pds, scenario_engine)
        n_quarters = len(weighted_pd)
        discount_factors = self._quarterly_discount_factors(n_quarters)
        cum_pd = self._cumulative_pd(weighted_pd)

        # 12-month ECL (Stage 1): sum of discounted marginal PDs over first 4 quarters
        n_12m = min(4, n_quarters)
        ecl_12m = np.sum(weighted_pd[:n_12m] * discount_factors[:n_12m]) * lgd * ead

        # Lifetime ECL (Stage 2): sum of discounted marginal PDs over full horizon
        ecl_lifetime = np.sum(weighted_pd * discount_factors) * lgd * ead

        # Per-scenario contribution breakdown
        contributions = {}
        for name, pds in scenario_pds.items():
            weight = scenario_engine.get_weight(name)
            pds_arr = np.asarray(pds, dtype=float)
            scenario_cum_pd = self._cumulative_pd(pds_arr)
            scenario_ecl_lt = np.sum(pds_arr * discount_factors) * lgd * ead * weight
            scenario_ecl_12m = np.sum(pds_arr[:n_12m] * discount_factors[:n_12m]) * lgd * ead * weight
            contributions[name] = {
                "weight": weight,
                "ecl_12m_contribution": round(scenario_ecl_12m, 2),
                "ecl_lifetime_contribution": round(scenario_ecl_lt, 2),
                "terminal_cumulative_pd": round(float(scenario_cum_pd[-1]), 6),
            }

        return {
            "ecl_12m": round(ecl_12m, 2),
            "ecl_lifetime": round(ecl_lifetime, 2),
            "lgd": lgd,
            "ead": ead,
            "discount_rate": self.discount_rate,
            "weighted_pd_term_structure": np.round(weighted_pd, 8),
            "cumulative_pd": np.round(cum_pd, 8),
            "scenario_contributions": contributions,
            "horizon_quarters": n_quarters,
        }

    def sensitivity_analysis(
        self,
        scenario_pds: Dict[str, np.ndarray],
        scenario_engine: ScenarioEngine,
        weight_shifts: Optional[List[Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Analyze how ECL changes when scenario weights shift.

        Parameters
        ----------
        scenario_pds : dict
            Scenario PDs.
        scenario_engine : ScenarioEngine
            Base scenario engine (for structure).
        weight_shifts : list of dict, optional
            Alternative weight configurations to test.
            Defaults to 5 standard configurations.

        Returns
        -------
        pd.DataFrame
            ECL under each weight configuration.
        """
        if weight_shifts is None:
            weight_shifts = [
                {"base": 0.50, "downside": 0.30, "upside": 0.20},  # Standard
                {"base": 0.40, "downside": 0.40, "upside": 0.20},  # Pessimistic
                {"base": 0.60, "downside": 0.20, "upside": 0.20},  # Neutral
                {"base": 0.33, "downside": 0.34, "upside": 0.33},  # Equal
                {"base": 0.30, "downside": 0.50, "upside": 0.20},  # Stressed
            ]

        rows = []
        for i, weights in enumerate(weight_shifts):
            # Temporarily update weights
            original_weights = dict(scenario_engine.scenario_weights)
            scenario_engine.update_weights(weights)

            ecl_result = self.compute_ecl(scenario_pds, scenario_engine)

            rows.append({
                "config": f"Config {i + 1}",
                "base_weight": weights.get("base", 0),
                "downside_weight": weights.get("downside", 0),
                "upside_weight": weights.get("upside", 0),
                "ecl_12m": ecl_result["ecl_12m"],
                "ecl_lifetime": ecl_result["ecl_lifetime"],
            })

            # Restore original weights
            scenario_engine.update_weights(original_weights)

        return pd.DataFrame(rows)
