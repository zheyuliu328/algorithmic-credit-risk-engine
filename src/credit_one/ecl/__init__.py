"""
ECL PD Forward Model Selection Module (Enhanced)

Implements IFRS 9 forward-looking ECL calculation with:
- Real macro data from FRED API or bundled US quarterly data (2000-2024)
- PD forward mapping via regression models (logistic, probit, linear)
- VAR-based Monte Carlo scenario engine (5 percentile scenarios)
- Enhanced model selection: VIF, walk-forward CV, Durbin-Watson, p-value screening
- Probability-weighted ECL calculation (Stage 1 / Stage 2)

This module extends CreditOne's existing PD scoring pipeline by converting
through-the-cycle PD estimates into forward-looking, scenario-weighted ECL.
"""

from credit_one.ecl.macro_data import MacroDataGenerator, MacroDataLoader, FREDDataLoader
from credit_one.ecl.pd_forward_model import PDForwardModel
from credit_one.ecl.scenario_engine import ScenarioEngine
from credit_one.ecl.model_selection import ModelSelectionPipeline
from credit_one.ecl.ecl_calculator import ECLCalculator
from credit_one.ecl.visualization import ECLVisualizer
from credit_one.ecl.runner import ECLRunner

__all__ = [
    "MacroDataGenerator",
    "MacroDataLoader",
    "FREDDataLoader",
    "PDForwardModel",
    "ScenarioEngine",
    "ModelSelectionPipeline",
    "ECLCalculator",
    "ECLVisualizer",
    "ECLRunner",
]
