"""
Macro-Economic Data Layer

Provides real and synthetic macro-economic time series for PD forward modeling.

Data sources (priority order):
1. FRED API via fredapi — live US quarterly data (requires API key)
2. Bundled CSV — data/us_macro_quarterly.csv (2000Q1-2024Q4, no API key needed)
3. CSV file loader — user-supplied file
4. Synthetic generator — for unit tests only

Variables: GDP growth rate, unemployment rate, interest rate, observed default rate.
All series aligned to quarterly frequency.

FRED series IDs:
- GDP growth:       A191RL1Q225SBEA (Real GDP, annualized quarterly)
- Unemployment:     UNRATE (monthly → quarterly average)
- Interest rate:    FEDFUNDS (monthly → quarterly average)
- Default rate:     DRCCLACBS (credit card delinquency, quarterly)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# 项目 data 目录下的打包历史数据
_BUNDLED_CSV = Path(__file__).resolve().parents[3] / "data" / "us_macro_quarterly.csv"

# FRED series 配置
FRED_SERIES = {
    "gdp_growth": "A191RL1Q225SBEA",
    "unemployment_rate": "UNRATE",
    "interest_rate": "FEDFUNDS",
    "observed_default_rate": "DRCCLACBS",
}


class FREDDataLoader:
    """
    从 FRED API 拉取实际美国宏观经济季度数据。

    需要:
    - pip install fredapi
    - 设置环境变量 FRED_API_KEY 或传入 api_key 参数
      (免费申请: https://fred.stlouisfed.org/docs/api/api_key.html)

    Parameters
    ----------
    api_key : str, optional
        FRED API key. 若为 None，从 FRED_API_KEY 环境变量读取。
    start : str
        起始日期，默认 '2000-01-01'。
    end : str, optional
        截止日期，默认为最新可用。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        start: str = "2000-01-01",
        end: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.start = start
        self.end = end

        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY env var or pass api_key param. "
                "Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("fredapi required: pip install fredapi")

        self._fred = Fred(api_key=self.api_key)

    def load(self) -> pd.DataFrame:
        """
        拉取并对齐所有 FRED 序列为季度频率。

        Returns
        -------
        pd.DataFrame
            Columns: date, gdp_growth, unemployment_rate, interest_rate, observed_default_rate.
        """
        series_data = {}
        for col_name, series_id in FRED_SERIES.items():
            raw = self._fred.get_series(
                series_id,
                observation_start=self.start,
                observation_end=self.end,
            )
            series_data[col_name] = raw

        # GDP growth 已经是季度频率，其他需要 resample
        gdp = series_data["gdp_growth"].dropna()
        gdp.index = gdp.index.to_period("Q").to_timestamp("Q")

        # 月度 → 季度平均
        unemp = series_data["unemployment_rate"].resample("QS").mean().dropna()
        ffr = series_data["interest_rate"].resample("QS").mean().dropna()

        # 违约率已经是季度
        delinq_raw = series_data["observed_default_rate"].dropna()
        delinq = delinq_raw.resample("QS").last().dropna()
        # DRCCLACBS 是百分比，转为小数
        delinq = delinq / 100.0

        # 对齐到共同日期范围
        df = pd.DataFrame({
            "gdp_growth": gdp,
            "unemployment_rate": unemp,
            "interest_rate": ffr,
            "observed_default_rate": delinq,
        })
        df = df.dropna()
        df = df.reset_index()
        df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])

        return df


class MacroDataGenerator:
    """
    合成宏观经济数据生成器 — 仅用于单元测试。

    生成相关的宏观变量，模拟经济周期:
    - GDP 增长率: 均值回归 AR(1) + 衰退冲击
    - 失业率: Okun 定律近似（与 GDP 反向）
    - 利率: 简化 Taylor 规则
    - 违约率: 宏观条件的 logistic 函数
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        start_year: int = 2010,
        end_year: int = 2024,
        include_recession: bool = True,
    ) -> pd.DataFrame:
        """生成季度宏观经济时间序列（仅供测试）。"""
        quarters = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq="QS",
        )
        n = len(quarters)

        gdp = np.zeros(n)
        gdp[0] = 2.0
        for t in range(1, n):
            gdp[t] = 0.7 * gdp[t - 1] + 0.3 * 2.0 + self.rng.normal(0, 0.8)

        unemployment = 5.5 - 0.4 * gdp + self.rng.normal(0, 0.3, n)
        unemployment = np.clip(unemployment, 3.0, 12.0)

        interest = 2.0 + 0.5 * gdp + 0.3 * (unemployment - 5.0) + self.rng.normal(0, 0.2, n)
        interest = np.clip(interest, 0.25, 8.0)

        if include_recession:
            recession_mask = (quarters >= "2020-01-01") & (quarters < "2020-10-01")
            recession_idx = np.where(recession_mask)[0]
            if len(recession_idx) > 0:
                gdp[recession_idx] = np.array([-5.0, -9.0, -2.5])[: len(recession_idx)]
                unemployment[recession_idx] = np.array([6.0, 10.5, 8.8])[: len(recession_idx)]
                interest[recession_idx] = np.array([1.0, 0.25, 0.25])[: len(recession_idx)]

        logit_dr = -3.5 - 0.15 * gdp + 0.20 * unemployment + 0.05 * interest
        logit_dr += self.rng.normal(0, 0.1, n)
        observed_default_rate = 1.0 / (1.0 + np.exp(-logit_dr))

        df = pd.DataFrame({
            "date": quarters,
            "gdp_growth": np.round(gdp, 4),
            "unemployment_rate": np.round(unemployment, 4),
            "interest_rate": np.round(interest, 4),
            "observed_default_rate": np.round(observed_default_rate, 6),
        })
        return df


class MacroDataLoader:
    """
    加载宏观经济数据，支持多种来源:
    1. load_bundled() — 使用打包的美国历史数据（默认推荐）
    2. load_fred() — 从 FRED API 拉取最新数据
    3. load_csv() — 从用户指定的 CSV 文件加载
    4. load_dataframe() — 从 DataFrame 加载
    5. generate_synthetic() — 生成合成数据（仅供测试）
    """

    REQUIRED_COLUMNS = ["date", "gdp_growth", "unemployment_rate", "interest_rate"]

    def __init__(self) -> None:
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("No data loaded. Call load_bundled(), load_csv(), etc. first.")
        return self._data

    def load_bundled(self) -> pd.DataFrame:
        """
        加载打包的美国宏观季度数据 (2000Q1-2024Q4, 100 quarters)。

        数据来源: FRED (A191RL1Q225SBEA, UNRATE, FEDFUNDS, DRCCLACBS)。
        这是生产环境的默认数据源 — 无需 API key。

        Returns
        -------
        pd.DataFrame
        """
        if not _BUNDLED_CSV.exists():
            raise FileNotFoundError(
                f"Bundled macro data not found at {_BUNDLED_CSV}. "
                "Run: python scripts/generate_macro_data.py"
            )
        return self.load_csv(str(_BUNDLED_CSV))

    def load_fred(self, api_key: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        从 FRED API 拉取最新数据。

        Parameters
        ----------
        api_key : str, optional
            FRED API key (或设置 FRED_API_KEY 环境变量)。

        Returns
        -------
        pd.DataFrame
        """
        loader = FREDDataLoader(api_key=api_key, **kwargs)
        df = loader.load()
        return self.load_dataframe(df)

    def load_csv(self, filepath: str, date_column: str = "date") -> pd.DataFrame:
        """从 CSV 文件加载宏观数据。"""
        df = pd.read_csv(filepath, parse_dates=[date_column])
        return self.load_dataframe(df)

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载并验证 DataFrame。"""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._data = df
        return df

    def generate_synthetic(self, **kwargs) -> pd.DataFrame:
        """生成合成数据（仅供测试）。生产环境请用 load_bundled()。"""
        gen = MacroDataGenerator(seed=kwargs.pop("seed", 42))
        df = gen.generate(**kwargs)
        return self.load_dataframe(df)
