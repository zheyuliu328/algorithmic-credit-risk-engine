"""
ECL Visualization Module

Generates publication-quality charts for:
- Model selection comparison table
- PD term structure across scenarios (5 scenarios)
- ECL waterfall / bar chart by scenario contribution
- Sensitivity analysis grouped bar chart
- Fan chart showing Monte Carlo simulation distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Optional


# 5 场景调色板
COLORS = {
    "severe_downside": "#B71C1C",  # 深红
    "downside": "#F44336",          # 红
    "base": "#2196F3",              # 蓝
    "upside": "#4CAF50",            # 绿
    "severe_upside": "#1B5E20",     # 深绿
    "weighted": "#FF9800",          # 橙
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

    def __init__(self, output_dir: str = "artifacts", figsize: tuple = (14, 6)):
        self.output_dir = output_dir
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-whitegrid")

    def _get_color(self, name: str) -> str:
        """根据场景名称获取颜色，未知名称返回灰色。"""
        return COLORS.get(name, "#999999")

    def plot_model_selection(
        self,
        results_df: pd.DataFrame,
        top_n: int = 15,
        save: bool = True,
    ) -> plt.Figure:
        """Horizontal bar chart comparing top candidate models."""
        top = results_df.head(top_n).copy()
        top = top.sort_values("composite_score", ascending=True)

        n_panels = 3
        has_vif = "max_vif" in top.columns
        has_dw = "durbin_watson" in top.columns
        if has_vif:
            n_panels += 1
        if has_dw:
            n_panels += 1

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, max(6, top_n * 0.4)))
        if n_panels == 1:
            axes = [axes]

        idx = 0
        # Panel 1: Composite score
        ax = axes[idx]
        ax.barh(top["model_label"], top["composite_score"], color="#2196F3", alpha=0.8)
        ax.set_xlabel("Composite Score (higher = better)")
        ax.set_title("Composite Score")
        ax.axvline(x=top["composite_score"].max(), color="red", linestyle="--", alpha=0.5)
        idx += 1

        # Panel 2: AIC
        ax = axes[idx]
        ax.barh(top["model_label"], top["aic"], color="#FF9800", alpha=0.8)
        ax.set_xlabel("AIC (lower = better)")
        ax.set_title("Akaike Information Criterion")
        idx += 1

        # Panel 3: OOS RMSE
        ax = axes[idx]
        ax.barh(top["model_label"], top["oos_rmse"], color="#4CAF50", alpha=0.8)
        ax.set_xlabel("OOS RMSE (lower = better)")
        ax.set_title("Out-of-Sample RMSE")
        idx += 1

        # Panel 4: Max VIF (if available)
        if has_vif:
            ax = axes[idx]
            colors_vif = ["#F44336" if v > 5 else "#4CAF50" for v in top["max_vif"]]
            ax.barh(top["model_label"], top["max_vif"], color=colors_vif, alpha=0.8)
            ax.axvline(x=5.0, color="red", linestyle="--", alpha=0.7, label="VIF=5")
            ax.set_xlabel("Max VIF")
            ax.set_title("Multicollinearity (VIF)")
            ax.legend()
            idx += 1

        # Panel 5: Durbin-Watson (if available)
        if has_dw:
            ax = axes[idx]
            colors_dw = [
                "#F44336" if dw < 1.5 or dw > 2.5 else "#4CAF50"
                for dw in top["durbin_watson"]
            ]
            ax.barh(top["model_label"], top["durbin_watson"], color=colors_dw, alpha=0.8)
            ax.axvline(x=2.0, color="blue", linestyle="--", alpha=0.5, label="DW=2 (ideal)")
            ax.set_xlabel("Durbin-Watson")
            ax.set_title("Residual Autocorrelation")
            ax.legend()

        fig.suptitle(f"Model Selection Results — Top {top_n} Candidates", fontsize=14, y=1.02)
        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_model_selection.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        plt.close(fig)
        return fig

    def plot_pd_term_structure(
        self,
        scenario_pds: Dict[str, np.ndarray],
        weighted_pd: np.ndarray,
        horizon_quarters: int,
        save: bool = True,
    ) -> plt.Figure:
        """Line chart of PD term structure across 5 scenarios."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        quarters = np.arange(1, horizon_quarters + 1)

        # Left: Marginal PD
        for name, pds in scenario_pds.items():
            color = self._get_color(name)
            ax1.plot(quarters, pds * 100, marker="o", markersize=3, label=name.replace("_", " ").title(),
                     color=color, linewidth=1.5)
        ax1.plot(quarters, weighted_pd * 100, marker="s", markersize=4, label="Weighted",
                 color=COLORS["weighted"], linewidth=2.5, linestyle="--")
        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Marginal PD (%)")
        ax1.set_title("Quarterly Marginal PD by Scenario")
        ax1.legend(fontsize=8)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # Right: Cumulative PD
        for name, pds in scenario_pds.items():
            cum_pd = 1 - np.cumprod(1 - pds)
            color = self._get_color(name)
            ax2.plot(quarters, cum_pd * 100, marker="o", markersize=3, label=name.replace("_", " ").title(),
                     color=color, linewidth=1.5)
        cum_weighted = 1 - np.cumprod(1 - weighted_pd)
        ax2.plot(quarters, cum_weighted * 100, marker="s", markersize=4, label="Weighted",
                 color=COLORS["weighted"], linewidth=2.5, linestyle="--")
        ax2.set_xlabel("Quarter")
        ax2.set_ylabel("Cumulative PD (%)")
        ax2.set_title("Cumulative PD by Scenario")
        ax2.legend(fontsize=8)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        fig.suptitle("PD Term Structure — Forward-Looking Projection", fontsize=14)
        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_pd_term_structure.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        plt.close(fig)
        return fig

    def plot_ecl_waterfall(
        self,
        ecl_result: Dict,
        save: bool = True,
    ) -> plt.Figure:
        """Bar chart showing ECL contribution by scenario."""
        contributions = ecl_result["scenario_contributions"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        names = list(contributions.keys())
        display_names = [n.replace("_", " ").title() for n in names]
        colors = [self._get_color(n) for n in names]

        # Left: 12-month ECL
        ecl_12m_vals = [contributions[n]["ecl_12m_contribution"] for n in names]
        bars = ax1.bar(display_names, ecl_12m_vals, color=colors, alpha=0.85)
        ax1.bar(["Total"], [ecl_result["ecl_12m"]], color=COLORS["weighted"], alpha=0.85)
        ax1.set_ylabel("ECL Amount ($)")
        ax1.set_title("12-Month ECL (Stage 1)")
        ax1.tick_params(axis="x", rotation=30, labelsize=8)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f"${height:,.0f}", ha="center", va="bottom", fontsize=7)

        # Right: Lifetime ECL
        ecl_lt_vals = [contributions[n]["ecl_lifetime_contribution"] for n in names]
        bars = ax2.bar(display_names, ecl_lt_vals, color=colors, alpha=0.85)
        ax2.bar(["Total"], [ecl_result["ecl_lifetime"]], color=COLORS["weighted"], alpha=0.85)
        ax2.set_ylabel("ECL Amount ($)")
        ax2.set_title("Lifetime ECL (Stage 2)")
        ax2.tick_params(axis="x", rotation=30, labelsize=8)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f"${height:,.0f}", ha="center", va="bottom", fontsize=7)

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

        plt.close(fig)
        return fig

    def plot_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """Grouped bar chart showing ECL under different weight configurations."""
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sensitivity_df))
        width = 0.35

        bars1 = ax.bar(x - width / 2, sensitivity_df["ecl_12m"], width,
                        label="12-Month ECL", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width / 2, sensitivity_df["ecl_lifetime"], width,
                        label="Lifetime ECL", color="#F44336", alpha=0.85)

        # X-axis labels: show weight config
        labels = []
        weight_cols = [c for c in sensitivity_df.columns if c.endswith("_weight")]
        for _, row in sensitivity_df.iterrows():
            parts = []
            for wc in weight_cols:
                short_name = wc.replace("_weight", "")[:4].upper()
                parts.append(f"{short_name}:{row[wc]:.0%}")
            labels.append("\n".join(parts))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("ECL Amount ($)")
        ax.set_title("ECL Sensitivity to Scenario Weight Changes")
        ax.legend()

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

        plt.close(fig)
        return fig

    def plot_fan_chart(
        self,
        fan_data: Dict[str, pd.DataFrame],
        save: bool = True,
    ) -> plt.Figure:
        """
        Fan chart showing Monte Carlo simulation distribution vs scenarios.

        每个宏观变量一个子图，显示 5-95 百分位的扇形分布。

        Parameters
        ----------
        fan_data : dict
            变量名 → DataFrame (date, p5, p10, p25, p50, p75, p90, p95)
        """
        n_vars = len(fan_data)
        fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
        if n_vars == 1:
            axes = [axes]

        var_labels = {
            "gdp_growth": "GDP Growth (%)",
            "unemployment_rate": "Unemployment Rate (%)",
            "interest_rate": "Interest Rate (%)",
        }

        for ax, (var_name, df) in zip(axes, fan_data.items()):
            quarters = np.arange(1, len(df) + 1)

            # 90% band (p5 - p95)
            ax.fill_between(quarters, df["p5"], df["p95"], alpha=0.15, color="#2196F3",
                           label="5-95th pct")
            # 80% band (p10 - p90)
            ax.fill_between(quarters, df["p10"], df["p90"], alpha=0.25, color="#2196F3",
                           label="10-90th pct")
            # 50% band (p25 - p75)
            ax.fill_between(quarters, df["p25"], df["p75"], alpha=0.4, color="#2196F3",
                           label="25-75th pct")
            # Median
            ax.plot(quarters, df["p50"], color="#1565C0", linewidth=2, label="Median")

            ax.set_xlabel("Quarter")
            ax.set_ylabel(var_labels.get(var_name, var_name))
            ax.set_title(var_labels.get(var_name, var_name))
            ax.legend(fontsize=7, loc="upper right")

        fig.suptitle("Monte Carlo Scenario Fan Chart — VAR Simulation Distribution", fontsize=13)
        fig.tight_layout()

        if save:
            path = f"{self.output_dir}/ecl_fan_chart.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        plt.close(fig)
        return fig
