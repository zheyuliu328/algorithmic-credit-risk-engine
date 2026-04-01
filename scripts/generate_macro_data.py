"""
Generate realistic US macro-economic quarterly data (2000Q1-2024Q4).

Based on actual US economic history patterns from FRED:
- A191RL1Q225SBEA (Real GDP growth, annualized)
- UNRATE (Unemployment rate)
- FEDFUNDS (Federal Funds Rate)
- DRCCLACBS (Delinquency rate on credit card loans)

Values are approximate and calibrated to match real FRED data trends.
This file serves as a bundled fallback when FRED API is unavailable.
"""

import numpy as np
import pandas as pd

# fmt: off
# Approximate quarterly US macro data based on FRED historical patterns
# Each row: (year, quarter, gdp_growth_annualized, unemployment_rate, fed_funds_rate, delinquency_rate)
RAW_DATA = [
    # 2000: Strong economy, dot-com peak
    (2000, 1,  1.2, 4.0, 5.68, 4.10),
    (2000, 2,  7.8, 3.9, 6.27, 4.28),
    (2000, 3,  0.5, 4.0, 6.52, 4.40),
    (2000, 4,  2.3, 3.9, 6.47, 4.55),
    # 2001: Dot-com recession
    (2001, 1, -1.3, 4.2, 5.59, 4.72),
    (2001, 2,  2.5, 4.4, 4.33, 4.95),
    (2001, 3, -1.6, 4.8, 3.50, 5.18),
    (2001, 4,  1.1, 5.5, 2.13, 5.48),
    # 2002: Slow recovery
    (2002, 1,  3.5, 5.7, 1.73, 5.52),
    (2002, 2,  2.2, 5.8, 1.75, 5.38),
    (2002, 3,  2.4, 5.7, 1.74, 5.20),
    (2002, 4,  0.2, 5.9, 1.44, 5.15),
    # 2003: Expansion begins
    (2003, 1,  2.1, 5.9, 1.25, 5.08),
    (2003, 2,  3.8, 6.1, 1.24, 4.95),
    (2003, 3,  6.9, 6.1, 1.02, 4.80),
    (2003, 4,  4.8, 5.8, 1.00, 4.68),
    # 2004: Solid growth, rate hikes begin
    (2004, 1,  2.3, 5.7, 1.00, 4.52),
    (2004, 2,  3.0, 5.6, 1.01, 4.38),
    (2004, 3,  3.7, 5.4, 1.43, 4.25),
    (2004, 4,  3.5, 5.4, 1.95, 4.18),
    # 2005: Continued expansion, housing boom
    (2005, 1,  4.3, 5.3, 2.47, 4.12),
    (2005, 2,  2.1, 5.1, 2.94, 4.05),
    (2005, 3,  3.4, 5.0, 3.46, 4.00),
    (2005, 4,  2.3, 5.0, 3.98, 3.98),
    # 2006: Peak, housing cools
    (2006, 1,  5.4, 4.7, 4.46, 4.02),
    (2006, 2,  1.2, 4.6, 4.91, 4.08),
    (2006, 3,  0.4, 4.6, 5.25, 4.15),
    (2006, 4,  3.5, 4.4, 5.25, 4.22),
    # 2007: Pre-crisis slowdown
    (2007, 1,  0.2, 4.5, 5.26, 4.35),
    (2007, 2,  3.1, 4.5, 5.25, 4.48),
    (2007, 3,  2.7, 4.7, 5.07, 4.62),
    (2007, 4,  1.4, 4.8, 4.50, 4.85),
    # 2008: Global Financial Crisis
    (2008, 1, -2.3, 5.0, 3.18, 5.12),
    (2008, 2,  2.1, 5.3, 2.09, 5.45),
    (2008, 3, -2.1, 6.0, 1.94, 5.82),
    (2008, 4, -8.4, 6.9, 0.51, 6.28),
    # 2009: GFC trough
    (2009, 1, -4.4, 8.3, 0.18, 6.50),
    (2009, 2, -0.6, 9.3, 0.18, 6.75),
    (2009, 3,  1.5, 9.6, 0.16, 6.82),
    (2009, 4,  4.5, 9.9, 0.12, 6.90),
    # 2010: Early recovery
    (2010, 1,  1.5, 9.8, 0.13, 6.72),
    (2010, 2,  3.7, 9.6, 0.19, 6.48),
    (2010, 3,  2.7, 9.5, 0.19, 6.15),
    (2010, 4,  2.5, 9.4, 0.19, 5.80),
    # 2011: Moderate growth, debt ceiling crisis
    (2011, 1, -1.5, 9.0, 0.16, 5.42),
    (2011, 2,  2.9, 9.1, 0.09, 5.15),
    (2011, 3,  0.8, 9.0, 0.08, 4.90),
    (2011, 4,  4.7, 8.6, 0.07, 4.72),
    # 2012: Steady but slow
    (2012, 1,  3.2, 8.3, 0.10, 4.52),
    (2012, 2,  1.7, 8.2, 0.16, 4.35),
    (2012, 3,  0.5, 7.8, 0.14, 4.18),
    (2012, 4,  0.5, 7.8, 0.16, 4.08),
    # 2013: Gradual improvement
    (2013, 1,  3.6, 7.7, 0.14, 3.95),
    (2013, 2,  0.5, 7.5, 0.12, 3.82),
    (2013, 3,  3.2, 7.2, 0.08, 3.68),
    (2013, 4,  3.2, 6.9, 0.09, 3.55),
    # 2014: Solid expansion
    (2014, 1, -1.5, 6.7, 0.07, 3.42),
    (2014, 2,  5.5, 6.2, 0.09, 3.30),
    (2014, 3,  4.9, 6.1, 0.09, 3.22),
    (2014, 4,  2.3, 5.7, 0.10, 3.15),
    # 2015: Rate normalization begins
    (2015, 1,  3.2, 5.5, 0.11, 3.10),
    (2015, 2,  2.7, 5.4, 0.12, 3.05),
    (2015, 3,  1.3, 5.1, 0.14, 3.00),
    (2015, 4,  0.5, 5.0, 0.24, 2.95),
    # 2016: Stable growth
    (2016, 1,  1.3, 4.9, 0.36, 2.88),
    (2016, 2,  2.7, 4.9, 0.39, 2.82),
    (2016, 3,  2.2, 4.9, 0.40, 2.78),
    (2016, 4,  2.0, 4.7, 0.54, 2.72),
    # 2017: Tax reform era
    (2017, 1,  2.2, 4.7, 0.70, 2.65),
    (2017, 2,  2.2, 4.3, 0.95, 2.60),
    (2017, 3,  2.9, 4.3, 1.15, 2.55),
    (2017, 4,  3.5, 4.1, 1.20, 2.48),
    # 2018: Strong growth + rate hikes
    (2018, 1,  3.8, 4.1, 1.45, 2.42),
    (2018, 2,  2.9, 3.9, 1.74, 2.38),
    (2018, 3,  2.1, 3.7, 1.91, 2.35),
    (2018, 4,  1.3, 3.8, 2.22, 2.42),
    # 2019: Late cycle, rate cuts
    (2019, 1,  2.7, 3.8, 2.40, 2.48),
    (2019, 2,  3.2, 3.6, 2.38, 2.52),
    (2019, 3,  2.8, 3.6, 2.04, 2.58),
    (2019, 4,  2.4, 3.5, 1.64, 2.62),
    # 2020: COVID-19 pandemic
    (2020, 1, -5.3, 3.8, 1.08, 2.68),
    (2020, 2, -28.0, 13.0, 0.06, 3.15),
    (2020, 3,  35.3, 8.8, 0.09, 3.45),
    (2020, 4,  4.5, 6.7, 0.09, 3.20),
    # 2021: Recovery + stimulus
    (2021, 1,  6.3, 6.2, 0.08, 2.85),
    (2021, 2,  7.0, 5.9, 0.06, 2.55),
    (2021, 3,  2.7, 5.1, 0.08, 2.35),
    (2021, 4,  7.0, 4.2, 0.08, 2.15),
    # 2022: Inflation + aggressive rate hikes
    (2022, 1, -1.6, 3.8, 0.20, 2.05),
    (2022, 2, -0.6, 3.6, 0.76, 2.10),
    (2022, 3,  3.2, 3.5, 2.19, 2.18),
    (2022, 4,  2.6, 3.5, 3.65, 2.28),
    # 2023: Resilient economy, high rates
    (2023, 1,  2.2, 3.5, 4.57, 2.42),
    (2023, 2,  2.1, 3.5, 5.08, 2.55),
    (2023, 3,  4.9, 3.7, 5.33, 2.68),
    (2023, 4,  3.4, 3.7, 5.33, 2.82),
    # 2024: Soft landing path
    (2024, 1,  1.4, 3.8, 5.33, 2.95),
    (2024, 2,  3.0, 4.0, 5.33, 3.05),
    (2024, 3,  2.8, 4.2, 5.12, 3.12),
    (2024, 4,  2.3, 4.1, 4.58, 3.18),
]
# fmt: on


def main():
    rows = []
    for year, quarter, gdp, unemp, ffr, delinq in RAW_DATA:
        month = (quarter - 1) * 3 + 1
        date = f"{year}-{month:02d}-01"
        rows.append({
            "date": date,
            "gdp_growth": gdp,
            "unemployment_rate": unemp,
            "interest_rate": ffr,
            "observed_default_rate": round(delinq / 100, 6),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    out_path = "data/us_macro_quarterly.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} quarters to {out_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSummary statistics:")
    print(df.describe().round(4))


if __name__ == "__main__":
    main()
