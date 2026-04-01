"""
Macro-Economic Data Layer

Provides synthetic and file-based macro-economic time series for PD forward modeling.
Variables: GDP growth rate, unemployment rate, interest rate (quarterly, 2010-2024).

Designed for easy swap-in of real macro data sources (FRED, Bloomberg, central bank APIs).
"""

import numpy as np
import pandas as pd
from typing import Optional


class MacroDataGenerator:
    """
    Generates realistic synthetic macro-economic quarterly time series.

    The generator produces correlated macro variables that mimic business cycle dynamics:
    - GDP growth follows a mean-reverting process with recession episodes
    - Unemployment is inversely correlated with GDP (Okun's law approximation)
    - Interest rates follow a Taylor rule-like response to GDP and inflation proxy
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        start_year: int = 2010,
        end_year: int = 2024,
        include_recession: bool = True,
    ) -> pd.DataFrame:
        """
        Generate quarterly macro-economic time series.

        Parameters
        ----------
        start_year : int
            First year of the series.
        end_year : int
            Last year of the series (inclusive).
        include_recession : bool
            If True, inject a recession episode (2020 Q1-Q3) to add realism.

        Returns
        -------
        pd.DataFrame
            Columns: date, gdp_growth, unemployment_rate, interest_rate, observed_default_rate.
            Indexed by quarterly date.
        """
        quarters = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq="QS",
        )
        n = len(quarters)

        # GDP growth: mean-reverting AR(1) around 2.0% annualized
        gdp = np.zeros(n)
        gdp[0] = 2.0
        for t in range(1, n):
            gdp[t] = 0.7 * gdp[t - 1] + 0.3 * 2.0 + self.rng.normal(0, 0.8)

        # Unemployment: inverse Okun's law relationship with GDP
        unemployment = 5.5 - 0.4 * gdp + self.rng.normal(0, 0.3, n)
        unemployment = np.clip(unemployment, 3.0, 12.0)

        # Interest rate: simplified Taylor rule
        interest = 2.0 + 0.5 * gdp + 0.3 * (unemployment - 5.0) + self.rng.normal(0, 0.2, n)
        interest = np.clip(interest, 0.25, 8.0)

        # Inject recession shock (COVID-like, 2020 Q1-Q3)
        if include_recession:
            recession_mask = (quarters >= "2020-01-01") & (quarters < "2020-10-01")
            recession_idx = np.where(recession_mask)[0]
            if len(recession_idx) > 0:
                gdp[recession_idx] = np.array([-5.0, -9.0, -2.5])[: len(recession_idx)]
                unemployment[recession_idx] = np.array([6.0, 10.5, 8.8])[: len(recession_idx)]
                interest[recession_idx] = np.array([1.0, 0.25, 0.25])[: len(recession_idx)]

        # Observed default rate: logistic function of macro conditions
        # Higher default when GDP is low, unemployment is high
        logit_dr = -3.5 - 0.15 * gdp + 0.20 * unemployment + 0.05 * interest
        logit_dr += self.rng.normal(0, 0.1, n)
        observed_default_rate = 1.0 / (1.0 + np.exp(-logit_dr))

        df = pd.DataFrame(
            {
                "date": quarters,
                "gdp_growth": np.round(gdp, 4),
                "unemployment_rate": np.round(unemployment, 4),
                "interest_rate": np.round(interest, 4),
                "observed_default_rate": np.round(observed_default_rate, 6),
            }
        )
        return df


class MacroDataLoader:
    """
    Loads macro-economic data from CSV or DataFrame.
    Validates required columns and date parsing.
    """

    REQUIRED_COLUMNS = ["date", "gdp_growth", "unemployment_rate", "interest_rate"]

    def __init__(self) -> None:
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("No data loaded. Call load_csv() or load_dataframe() first.")
        return self._data

    def load_csv(self, filepath: str, date_column: str = "date") -> pd.DataFrame:
        """
        Load macro data from CSV file.

        Parameters
        ----------
        filepath : str
            Path to CSV file.
        date_column : str
            Name of the date column.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_csv(filepath, parse_dates=[date_column])
        return self.load_dataframe(df)

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and validate a DataFrame of macro data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: date, gdp_growth, unemployment_rate, interest_rate.

        Returns
        -------
        pd.DataFrame
        """
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._data = df
        return df

    def generate_synthetic(self, **kwargs: int) -> pd.DataFrame:
        """Convenience method: generate and load synthetic data in one call."""
        gen = MacroDataGenerator(seed=kwargs.pop("seed", 42))
        df = gen.generate(**kwargs)
        return self.load_dataframe(df)
