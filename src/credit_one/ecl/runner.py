"""
ECL Pipeline Runner — Integration with CreditOne

Orchestrates the full ECL forward model workflow:
1. Load real macro data (bundled US quarterly or FRED API)
2. Run enhanced model selection pipeline (VIF, walk-forward CV, DW)
3. Fit VAR model and generate Monte Carlo scenarios (5 percentile-based)
4. Compute scenario-specific forward PDs
5. Calculate probability-weighted ECL
6. Produce visualizations (including fan charts) and export results

Designed to accept base PD from CreditOne's existing XGBoost/Scorecard pipeline.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional

from credit_one.ecl.macro_data import MacroDataLoader
from credit_one.ecl.model_selection import ModelSelectionPipeline
from credit_one.ecl.scenario_engine import ScenarioEngine
from credit_one.ecl.ecl_calculator import ECLCalculator
from credit_one.ecl.visualization import ECLVisualizer


class ECLRunner:
    """
    End-to-end orchestrator for the ECL PD Forward Model module.

    Parameters
    ----------
    base_pd : float
        Through-the-cycle PD from CreditOne scoring pipeline.
    lgd : float
        Loss Given Default assumption. Default: 0.45.
    ead : float
        Exposure at Default. Default: 100000.0.
    output_dir : str
        Directory for artifacts. Default: 'artifacts'.
    """

    def __init__(
        self,
        base_pd: float = 0.05,
        lgd: float = 0.45,
        ead: float = 100000.0,
        output_dir: str = "artifacts",
    ):
        self.base_pd = base_pd
        self.lgd = lgd
        self.ead = ead
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.macro_loader = MacroDataLoader()
        self.model_selector: Optional[ModelSelectionPipeline] = None
        self.scenario_engine: Optional[ScenarioEngine] = None
        self.ecl_calculator: Optional[ECLCalculator] = None
        self.visualizer = ECLVisualizer(output_dir=output_dir)

    def run(
        self,
        macro_df: Optional[pd.DataFrame] = None,
        scenario_weights: Optional[Dict[str, float]] = None,
        horizon_quarters: int = 12,
        n_simulations: int = 2000,
        model_types: Optional[list] = None,
        max_lag: int = 2,
        ranking_metric: str = "composite",
        verbose: bool = True,
    ) -> Dict:
        """
        Execute the full ECL pipeline.

        Parameters
        ----------
        macro_df : pd.DataFrame, optional
            Historical macro data. If None, loads bundled US data.
        scenario_weights : dict, optional
            Override default scenario weights (5 scenarios).
        horizon_quarters : int
            Projection horizon in quarters.
        n_simulations : int
            Number of Monte Carlo paths for VAR simulation.
        model_types : list, optional
            Model types for selection. Default: ['logistic', 'linear'].
        max_lag : int
            Maximum lag for model selection.
        ranking_metric : str
            Metric for model ranking.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Complete pipeline results.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("ECL PD FORWARD MODEL PIPELINE (Enhanced)")
            print("=" * 70)
            print(f"Base PD (from CreditOne): {self.base_pd:.4%}")
            print(f"LGD: {self.lgd:.0%}  |  EAD: ${self.ead:,.0f}")
            print(f"Horizon: {horizon_quarters} quarters ({horizon_quarters / 4:.1f} years)")
            print(f"Monte Carlo simulations: {n_simulations}")

        # Step 1: Macro data — default to bundled real US data
        if verbose:
            print(f"\n--- Step 1: Macro-Economic Data ---")
        if macro_df is None:
            macro_df = self.macro_loader.load_bundled()
            if verbose:
                print(f"Loaded bundled US macro data: {len(macro_df)} quarters")
                d0 = macro_df['date'].iloc[0]
                d1 = macro_df['date'].iloc[-1]
                print(f"  Date range: {d0.strftime('%Y-%m')} to {d1.strftime('%Y-%m')}")
                print(f"  GDP growth: [{macro_df['gdp_growth'].min():.1f}%, {macro_df['gdp_growth'].max():.1f}%]")
                print(f"  Unemployment: [{macro_df['unemployment_rate'].min():.1f}%, {macro_df['unemployment_rate'].max():.1f}%]")
                print(f"  Default rate: [{macro_df['observed_default_rate'].min():.4%}, {macro_df['observed_default_rate'].max():.4%}]")
        else:
            macro_df = self.macro_loader.load_dataframe(macro_df)
            if verbose:
                print(f"Loaded user-supplied macro data: {len(macro_df)} quarters")

        # Step 2: Model selection (with VIF, walk-forward CV, DW)
        if verbose:
            print(f"\n--- Step 2: Model Selection (Enhanced) ---")
        self.model_selector = ModelSelectionPipeline(
            model_types=model_types or ["logistic", "linear"],
            max_lag=max_lag,
            ranking_metric=ranking_metric,
        )
        selection_results = self.model_selector.run(macro_df, verbose=verbose)
        best_model = self.model_selector.best_model

        if verbose:
            best_info = self.model_selector.get_model_details(rank=1)
            print(f"\nBest model: {best_info['model_label']}")
            print(f"  AIC: {best_info['aic']:.4f}")
            print(f"  Adj R²: {best_info['adj_r2']:.6f}")
            print(f"  OOS RMSE: {best_info['oos_rmse']:.8f}")
            print(f"  Max VIF: {best_info.get('max_vif', 'N/A')}")
            print(f"  Durbin-Watson: {best_info.get('durbin_watson', 'N/A')}")
            if best_info.get('vif_flag'):
                print(f"  ⚠ VIF flag: multicollinearity detected")
            if best_info.get('pvalue_flag'):
                print(f"  ⚠ P-value flag: insignificant coefficients")

        # Step 3: VAR-based scenario generation with Monte Carlo
        if verbose:
            print(f"\n--- Step 3: VAR Scenario Generation ---")
        self.scenario_engine = ScenarioEngine(
            horizon_quarters=horizon_quarters,
            n_simulations=n_simulations,
            scenario_weights=scenario_weights,
        )

        scenarios = self.scenario_engine.generate_scenarios(macro_df=macro_df)

        if verbose:
            var_diag = self.scenario_engine.var_diagnostics()
            print(f"  VAR order: {var_diag['var_order']} (AIC: {var_diag['aic']})")
            print(f"  Simulated {n_simulations} paths")
            print(f"  Generated {len(scenarios)} scenarios:")
            print(self.scenario_engine.summary().to_string(index=False))

        # Step 4: Forward PD projection
        if verbose:
            print(f"\n--- Step 4: Forward PD Projection ---")
        self.ecl_calculator = ECLCalculator(
            lgd=self.lgd, ead=self.ead,
        )
        scenario_pds = self.ecl_calculator.compute_scenario_pds(
            model=best_model,
            scenario_engine=self.scenario_engine,
            base_pd=self.base_pd,
        )

        if verbose:
            for name, pds in scenario_pds.items():
                print(f"  {name:>18}: PD range [{pds.min():.4%}, {pds.max():.4%}]")

        # Step 5: ECL calculation
        if verbose:
            print(f"\n--- Step 5: ECL Calculation ---")
        ecl_result = self.ecl_calculator.compute_ecl(
            scenario_pds=scenario_pds,
            scenario_engine=self.scenario_engine,
        )

        if verbose:
            print(f"  12-Month ECL (Stage 1): ${ecl_result['ecl_12m']:,.2f}")
            print(f"  Lifetime ECL (Stage 2): ${ecl_result['ecl_lifetime']:,.2f}")
            for name, contrib in ecl_result["scenario_contributions"].items():
                print(
                    f"    {name:>18}: "
                    f"12m=${contrib['ecl_12m_contribution']:,.2f}  "
                    f"LT=${contrib['ecl_lifetime_contribution']:,.2f}  "
                    f"(weight={contrib['weight']:.1%})"
                )

        # Step 6: Sensitivity analysis
        if verbose:
            print(f"\n--- Step 6: Sensitivity Analysis ---")

        # 为 5 场景生成合理的权重配置
        weight_shifts = [
            {"severe_downside": 0.125, "downside": 0.225, "base": 0.300, "upside": 0.225, "severe_upside": 0.125},
            {"severe_downside": 0.200, "downside": 0.300, "base": 0.250, "upside": 0.150, "severe_upside": 0.100},
            {"severe_downside": 0.050, "downside": 0.150, "base": 0.400, "upside": 0.250, "severe_upside": 0.150},
            {"severe_downside": 0.200, "downside": 0.200, "base": 0.200, "upside": 0.200, "severe_upside": 0.200},
            {"severe_downside": 0.300, "downside": 0.300, "base": 0.200, "upside": 0.100, "severe_upside": 0.100},
        ]
        sensitivity_df = self.ecl_calculator.sensitivity_analysis(
            scenario_pds=scenario_pds,
            scenario_engine=self.scenario_engine,
            weight_shifts=weight_shifts,
        )
        if verbose:
            print(sensitivity_df.to_string(index=False))

        # Step 7: Visualizations
        if verbose:
            print(f"\n--- Step 7: Generating Visualizations ---")
        weighted_pd = self.ecl_calculator.compute_weighted_pd(
            scenario_pds, self.scenario_engine,
        )

        self.visualizer.plot_model_selection(selection_results, save=True)
        self.visualizer.plot_pd_term_structure(
            scenario_pds, weighted_pd, horizon_quarters, save=True,
        )
        self.visualizer.plot_ecl_waterfall(ecl_result, save=True)
        self.visualizer.plot_sensitivity(sensitivity_df, save=True)

        # Fan chart (new)
        fan_data = self.scenario_engine.fan_chart_data()
        self.visualizer.plot_fan_chart(fan_data, save=True)

        # Export
        csv_path = f"{self.output_dir}/ecl_model_selection_results.csv"
        self.model_selector.export_results(csv_path)

        summary_path = f"{self.output_dir}/ecl_summary.csv"
        self._export_ecl_summary(ecl_result, sensitivity_df, summary_path)

        if verbose:
            print(f"\n{'=' * 70}")
            print("ECL PIPELINE COMPLETE")
            print(f"{'=' * 70}")
            print(f"Artifacts saved to: {self.output_dir}/")

        return {
            "ecl_result": ecl_result,
            "model_selection": selection_results,
            "best_model_label": self.model_selector.get_model_details(rank=1)["model_label"],
            "scenario_pds": scenario_pds,
            "weighted_pd": weighted_pd,
            "sensitivity": sensitivity_df,
            "macro_data": macro_df,
            "var_diagnostics": self.scenario_engine.var_diagnostics(),
            "fan_chart_data": fan_data,
        }

    def _export_ecl_summary(
        self,
        ecl_result: Dict,
        sensitivity_df: pd.DataFrame,
        filepath: str,
    ) -> None:
        """Export ECL summary to CSV."""
        rows = [
            {"metric": "Base PD (TTC)", "value": f"{self.base_pd:.4%}"},
            {"metric": "LGD", "value": f"{self.lgd:.0%}"},
            {"metric": "EAD", "value": f"${self.ead:,.0f}"},
            {"metric": "12-Month ECL (Stage 1)", "value": f"${ecl_result['ecl_12m']:,.2f}"},
            {"metric": "Lifetime ECL (Stage 2)", "value": f"${ecl_result['ecl_lifetime']:,.2f}"},
            {"metric": "Horizon (quarters)", "value": str(ecl_result["horizon_quarters"])},
        ]
        for name, contrib in ecl_result["scenario_contributions"].items():
            rows.append({
                "metric": f"{name} - 12m contribution",
                "value": f"${contrib['ecl_12m_contribution']:,.2f}",
            })
            rows.append({
                "metric": f"{name} - lifetime contribution",
                "value": f"${contrib['ecl_lifetime_contribution']:,.2f}",
            })
            rows.append({
                "metric": f"{name} - terminal cum PD",
                "value": f"{contrib['terminal_cumulative_pd']:.4%}",
            })

        pd.DataFrame(rows).to_csv(filepath, index=False)
        print(f"ECL summary exported to {filepath}")
