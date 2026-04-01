"""
ECL PD Forward Model Selection Module

Implements IFRS 9 forward-looking ECL calculation with:
- Macro-economic data ingestion and management
- PD forward mapping via regression models (logistic, probit, linear)
- Multi-scenario projection engine (Base / Downside / Upside)
- Automated model selection pipeline with AIC, BIC, R², RMSE, H-L test
- Probability-weighted ECL calculation (Stage 1 / Stage 2)

This module extends CreditOne's existing PD scoring pipeline by converting
through-the-cycle PD estimates into forward-looking, scenario-weighted ECL.
"""

from credit_one.ecl.macro_data import MacroDataGenerator, MacroDataLoader
from credit_one.ecl.pd_forward_model import PDForwardModel
from credit_one.ecl.scenario_engine import ScenarioEngine
from credit_one.ecl.model_selection import ModelSelectionPipeline
from credit_one.ecl.ecl_calculator import ECLCalculator
from credit_one.ecl.visualization import ECLVisualizer
from credit_one.ecl.runner import ECLRunner

__all__ = [
    "MacroDataGenerator",
    "MacroDataLoader",
    "PDForwardModel",
    "ScenarioEngine",
    "ModelSelectionPipeline",
    "ECLCalculator",
    "ECLVisualizer",
    "ECLRunner",
]
