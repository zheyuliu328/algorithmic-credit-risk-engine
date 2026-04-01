"""
ECL Visualization Module

Generates publication-quality charts for:
- Model selection comparison table
- PD term structure across scenarios
- ECL waterfall / bar chart by scenario contribution
- Sensitivity analysis heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Optional


# Consistent color palette
COLORS = {
    "base": "#2196F3",
    "downside": "#F44336",
    "upside": "#4CAF50",
    "weighted": "#FF9800",
}


class ECLVisualizer:
    """
    Visualization suite for ECL forward model outputs.

    Parameters
    ----------
    output_dir : str
        Directory to save chart files.
    figsize : tuple
        Default figure size (width, height).
    """

    def __init__(self, output_dir: str = "artifacts", figsize: tuple = (12, 6)):
        self.output_dir = output_dir
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_model_selection(
        self,
        results_df: pd.DataFrame,
        top_n: int = 15,
        save: bool = True,
    ) -> plt.Figure:
        """
        Horizontal bar chart comparing top candidate models by composite score.

        Parameters
        ----------
        results_df : pd.DataFrame
            Output from ModelSelectionPipeline.run().
        top_n : int
            Number of top models to display.
        save : bool
            Save to file.

        Returns
        -------
        plt.Figure
        """
        top = results_df.head(top_n).copy()
        top = top.sort_values("composite_score", ascending=True)  # For horizontal bar

        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, top_n * 0.4)))

        # Panel 1: Composite score
        ax = axes[0]
        bars = ax.barh(top["model_label"], top["composite_score"], color="#2196F3", alpha=0.8)
        ax.set_xlabel("Composite Score (higher = better)")
        ax.set_title("Composite Score")
        ax.axvline(x=top["composite_score"].max(), color="red", linestyle="--", alpha=0.5)

        # Panel 2: AIC comparison
        ax = axes[1]
        ax.barh(top["model_label"], top["aic"], color="#FF9800", alpha=0.8)
        ax.set_xlabel("AIC (lower = better)")
        ax.set_title("Akaike Information Criterion")

        # Panel 3: OOS RMSE
        ax = axes[2]
        ax.barh(top["model_label"], top["oos_rmse"], color="#4CAF50", alpha=0.8)
        ax.set_xlabel("Out-of-Sample RMSE (lower = better)")
        ax.set_title("Out-of-Sample RMSE")

        fig.suptitle(f"Model Selection Results — Top {top_n} Candidates", fontsize=14, y=1.02)
        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_model_selection.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_pd_term_structure(
        self,
        scenario_pds: Dict[str, np.ndarray],
        weighted_pd: np.ndarray,
        horizon_quarters: int,
        save: bool = True,
    ) -> plt.Figure:
        """
        Line chart of PD term structure across scenarios.

        Parameters
        ----------
        scenario_pds : dict
            Scenario name to PD array.
        weighted_pd : np.ndarray
            Probability-weighted PD.
        horizon_quarters : int
            Number of projection quarters.
        save : bool
            Save to file.

        Returns
        -------
        plt.Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        quarters = np.arange(1, horizon_quarters + 1)

        # Left: Marginal PD per quarter
        for name, pds in scenario_pds.items():
            color = COLORS.get(name, "#999999")
            ax1.plot(quarters, pds * 100, marker="o", markersize=4, label=name.title(),
                     color=color, linewidth=2)
        ax1.plot(quarters, weighted_pd * 100, marker="s", markersize=5, label="Weighted",
                 color=COLORS["weighted"], linewidth=2.5, linestyle="--")
        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Marginal PD (%)")
        ax1.set_title("Quarterly Marginal PD by Scenario")
        ax1.legend()
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Right: Cumulative PD
        for name, pds in scenario_pds.items():
            cum_pd = 1 - np.cumprod(1 - pds)
            color = COLORS.get(name, "#999999")
            ax2.plot(quarters, cum_pd * 100, marker="o", markersize=4, label=name.title(),
                     color=color, linewidth=2)
        cum_weighted = 1 - np.cumprod(1 - weighted_pd)
        ax2.plot(quarters, cum_weighted * 100, marker="s", markersize=5, label="Weighted",
                 color=COLORS["weighted"], linewidth=2.5, linestyle="--")
        ax2.set_xlabel("Quarter")
        ax2.set_ylabel("Cumulative PD (%)")
        ax2.set_title("Cumulative PD by Scenario")
        ax2.legend()
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        fig.suptitle("PD Term Structure — Forward-Looking Projection", fontsize=14)
        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_pd_term_structure.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_ecl_waterfall(
        self,
        ecl_result: Dict,
        save: bool = True,
    ) -> plt.Figure:
        """
        Bar chart showing ECL contribution by scenario.

        Parameters
        ----------
        ecl_result : dict
            Output from ECLCalculator.compute_ecl().
        save : bool
            Save to file.

        Returns
        -------
        plt.Figure
        """
        contributions = ecl_result["scenario_contributions"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Left: 12-month ECL breakdown
        names = list(contributions.keys())
        ecl_12m_vals = [contributions[n]["ecl_12m_contribution"] for n in names]
        colors = [COLORS.get(n, "#999999") for n in names]

        bars = ax1.bar([n.title() for n in names], ecl_12m_vals, color=colors, alpha=0.85)
        ax1.bar(["Total"], [ecl_result["ecl_12m"]], color=COLORS["weighted"], alpha=0.85)

        ax1.set_ylabel("ECL Amount ($)")
        ax1.set_title("12-Month ECL (Stage 1)")
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f"${height:,.0f}", ha="center", va="bottom", fontsize=9)

        # Right: Lifetime ECL breakdown
        ecl_lt_vals = [contributions[n]["ecl_lifetime_contribution"] for n in names]
        bars = ax2.bar([n.title() for n in names], ecl_lt_vals, color=colors, alpha=0.85)
        ax2.bar(["Total"], [ecl_result["ecl_lifetime"]], color=COLORS["weighted"], alpha=0.85)

        ax2.set_ylabel("ECL Amount ($)")
        ax2.set_title("Lifetime ECL (Stage 2)")
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f"${height:,.0f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle(
            f"ECL Scenario Contribution (LGD={ecl_result['lgd']:.0%}, "
            f"EAD=${ecl_result['ead']:,.0f})",
            fontsize=14,
        )
        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_waterfall.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """
        Grouped bar chart showing ECL under different weight configurations.

        Parameters
        ----------
        sensitivity_df : pd.DataFrame
            Output from ECLCalculator.sensitivity_analysis().
        save : bool
            Save to file.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(sensitivity_df))
        width = 0.35

        bars1 = ax.bar(x - width / 2, sensitivity_df["ecl_12m"], width,
                        label="12-Month ECL", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width / 2, sensitivity_df["ecl_lifetime"], width,
                        label="Lifetime ECL", color="#F44336", alpha=0.85)

        # X-axis labels: show weight config
        labels = []
        for _, row in sensitivity_df.iterrows():
            labels.append(
                f"B:{row['base_weight']:.0%}\n"
                f"D:{row['downside_weight']:.0%}\n"
                f"U:{row['upside_weight']:.0%}"
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("ECL Amount ($)")
        ax.set_title("ECL Sensitivity to Scenario Weight Changes")
        ax.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f"${height:,.0f}", ha="center", va="bottom", fontsize=7)

        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_sensitivity.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig
