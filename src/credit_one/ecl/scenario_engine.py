"""
Multi-Scenario Projection Engine

Generates Base, Downside, and Upside macro-economic scenarios for forward PD projection.
Each scenario is a quarterly time series of macro forecasts over a configurable horizon.
Scenario weights are applied for probability-weighted ECL calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# Default scenario weights (IFRS 9 standard practice)
DEFAULT_SCENARIO_WEIGHTS = {
    "base": 0.50,
    "downside": 0.30,
    "upside": 0.20,
}

# Default macro forecast assumptions by scenario
DEFAULT_SCENARIO_ASSUMPTIONS = {
    "base": {
        "gdp_growth": 2.0,
        "unemployment_rate": 5.0,
        "interest_rate": 3.5,
    },
    "downside": {
        "gdp_growth": -1.5,
        "unemployment_rate": 8.0,
        "interest_rate": 1.5,
    },
    "upside": {
        "gdp_growth": 3.5,
        "unemployment_rate": 3.8,
        "interest_rate": 4.5,
    },
}


class ScenarioEngine:
    """
    Generates and manages macro-economic scenarios for IFRS 9 forward-looking ECL.

    Parameters
    ----------
    horizon_quarters : int
        Number of quarters to project (e.g., 12 = 3 years).
    scenario_weights : dict, optional
        Mapping of scenario name to probability weight. Must sum to 1.0.
    scenario_assumptions : dict, optional
        Mapping of scenario name to dict of macro variable terminal values.
    """

    def __init__(
        self,
        horizon_quarters: int = 12,
        scenario_weights: Optional[Dict[str, float]] = None,
        scenario_assumptions: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.horizon_quarters = horizon_quarters
        self.scenario_weights = scenario_weights or dict(DEFAULT_SCENARIO_WEIGHTS)
        self.scenario_assumptions = scenario_assumptions or {
            k: dict(v) for k, v in DEFAULT_SCENARIO_ASSUMPTIONS.items()
        }
        self._validate_weights()
        self._scenarios: Dict[str, pd.DataFrame] = {}

    def _validate_weights(self) -> None:
        """Ensure scenario weights sum to approximately 1.0."""
        total = sum(self.scenario_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scenario weights must sum to 1.0, got {total}. "
                f"Weights: {self.scenario_weights}"
            )

    def generate_scenarios(
        self,
        current_macro: Optional[Dict[str, float]] = None,
        start_date: str = "2025-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate quarterly macro forecasts for each scenario.

        The path from current values to terminal values follows an exponential
        convergence pattern — fast initial adjustment, then gradual settling.

        Parameters
        ----------
        current_macro : dict, optional
            Current quarter macro values. Defaults to neutral starting point.
        start_date : str
            Start date for projection.

        Returns
        -------
        dict
            Mapping of scenario name to DataFrame with quarterly macro forecasts.
        """
        if current_macro is None:
            current_macro = {
                "gdp_growth": 2.0,
                "unemployment_rate": 4.5,
                "interest_rate": 3.0,
            }

        dates = pd.date_range(start=start_date, periods=self.horizon_quarters, freq="QS")

        for scenario_name, terminal_values in self.scenario_assumptions.items():
            rows = []
            for t in range(self.horizon_quarters):
                # Exponential convergence: λ controls speed
                lam = 0.3
                weight = 1 - np.exp(-lam * (t + 1))
                row = {"date": dates[t]}
                for var in terminal_values:
                    current = current_macro.get(var, 0.0)
                    terminal = terminal_values[var]
                    row[var] = round(current + weight * (terminal - current), 4)
                rows.append(row)

            self._scenarios[scenario_name] = pd.DataFrame(rows)

        return dict(self._scenarios)

    @property
    def scenarios(self) -> Dict[str, pd.DataFrame]:
        if not self._scenarios:
            raise ValueError("No scenarios generated. Call generate_scenarios() first.")
        return self._scenarios

    def get_scenario(self, name: str) -> pd.DataFrame:
        """Get a single scenario DataFrame by name."""
        if name not in self._scenarios:
            available = list(self._scenarios.keys())
            raise KeyError(f"Scenario '{name}' not found. Available: {available}")
        return self._scenarios[name]

    def get_weight(self, name: str) -> float:
        """Get probability weight for a scenario."""
        if name not in self.scenario_weights:
            raise KeyError(f"No weight defined for scenario '{name}'")
        return self.scenario_weights[name]

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update scenario weights with validation.

        Parameters
        ----------
        new_weights : dict
            New scenario weights. Must sum to 1.0.
        """
        self.scenario_weights = dict(new_weights)
        self._validate_weights()

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table showing terminal values and weights per scenario.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for name, assumptions in self.scenario_assumptions.items():
            row = {"scenario": name, "weight": self.scenario_weights.get(name, 0.0)}
            row.update(assumptions)
            rows.append(row)
        return pd.DataFrame(rows)
