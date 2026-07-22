"""Chart generation for the Mutual Fund Portfolio project.

The :class:`PortfolioVisualizer` renders the six professional charts required
by the case study using **matplotlib only** and saves them as high-resolution
PNG files in the ``charts`` directory.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend suitable for batch rendering.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils import CHARTS_DIR, ensure_directories  # noqa: E402

_DPI: int = 150


class PortfolioVisualizer:
    """Render and persist the portfolio charts.

    Attributes:
        merged: Cleaned, merged transaction-level dataframe.
        nav_history: Cleaned NAV-history dataframe.
        logger: Logger used for execution tracing.
        charts_dir: Directory where PNG charts are written.
    """

    def __init__(
        self,
        merged: pd.DataFrame,
        nav_history: pd.DataFrame,
        logger: logging.Logger,
        charts_dir: str = CHARTS_DIR,
    ) -> None:
        """Initialise the visualizer.

        Args:
            merged: Cleaned, merged transaction-level dataframe.
            nav_history: Cleaned NAV-history dataframe.
            logger: Configured logger instance.
            charts_dir: Directory where charts are saved.
        """
        self.merged = merged
        self.nav_history = nav_history
        self.logger = logger
        self.charts_dir = charts_dir
        ensure_directories(self.charts_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_all_charts(
        self, analysis: Dict[str, Any], metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate every required chart.

        Args:
            analysis: Dictionary returned by the investor analysis.
            metrics: Dictionary returned by the financial-metrics computation.

        Returns:
            A list of file paths for the charts that were created.
        """
        self.logger.info("Generating charts")
        paths = [
            self.portfolio_allocation_pie(metrics["fund_allocation_pct"]),
            self.fund_investment_bar(),
            self.monthly_investment_trend(),
            self.category_returns_bar(),
            self.nav_movement_line(),
            self.top_investors_barh(analysis["top_20_investors"]),
        ]
        self.logger.info("Generated %d charts in %s", len(paths), self.charts_dir)
        return paths

    # ------------------------------------------------------------------ #
    # Individual charts
    # ------------------------------------------------------------------ #
    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save a figure with tight layout and high resolution.

        Args:
            fig: The matplotlib figure to save.
            filename: Target file name inside the charts directory.

        Returns:
            The absolute path to the saved chart.
        """
        fig.tight_layout()
        path = os.path.join(self.charts_dir, filename)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved chart: %s", path)
        return path

    def portfolio_allocation_pie(self, fund_allocation: pd.DataFrame) -> str:
        """Portfolio allocation pie chart (top 10 funds, rest grouped).

        Args:
            fund_allocation: Dataframe with ``FundName`` and ``Percentage``.

        Returns:
            The path to the saved chart.
        """
        top = fund_allocation.head(10).copy()
        others = fund_allocation.iloc[10:]["Percentage"].sum()
        labels = top["FundName"].tolist()
        sizes = top["Percentage"].tolist()
        if others > 0:
            labels.append("Others")
            sizes.append(float(others))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 8},
        )
        ax.set_title("Portfolio Allocation by Fund", fontsize=14, fontweight="bold")
        ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.axis("equal")
        return self._save(fig, "portfolio_allocation_pie.png")

    def fund_investment_bar(self) -> str:
        """Fund-wise investment bar chart (top 15 funds).

        Returns:
            The path to the saved chart.
        """
        grouped = (
            self.merged.groupby("FundName")["Amount"].sum().nlargest(15).sort_values()
        )
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(grouped.index, grouped.to_numpy(), color="#2a7de1")
        ax.set_title("Top 15 Funds by Total Investment", fontsize=14, fontweight="bold")
        ax.set_xlabel("Fund Name", fontsize=11)
        ax.set_ylabel("Total Investment (\u20b9)", fontsize=11)
        ax.tick_params(axis="x", rotation=75, labelsize=8)
        ax.legend(["Total Investment"], loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        return self._save(fig, "fund_investment_bar.png")

    def monthly_investment_trend(self) -> str:
        """Monthly investment trend line chart.

        Returns:
            The path to the saved chart.
        """
        monthly = (
            self.merged.set_index("TransactionDate")
            .resample("ME")["Amount"]
            .sum()
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            monthly.index,
            monthly.to_numpy(),
            marker="o",
            color="#e1642a",
            label="Monthly Investment",
        )
        ax.set_title("Monthly Investment Trend", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Investment Amount (\u20b9)", fontsize=11)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        return self._save(fig, "monthly_investment_trend.png")

    def category_returns_bar(self) -> str:
        """Category-wise returns bar chart.

        Returns:
            The path to the saved chart.
        """
        # Return proxy: current value versus net invested per category.
        grouped = self.merged.groupby("Category").agg(
            NetInvested=("SignedAmount", "sum"),
            CurrentValue=("CurrentValue", "sum"),
        )
        grouped["ReturnPct"] = (
            (grouped["CurrentValue"] - grouped["NetInvested"])
            / grouped["NetInvested"].replace(0, float("nan"))
            * 100.0
        )
        grouped = grouped["ReturnPct"].fillna(0.0).sort_values()

        colors = ["#c0392b" if v < 0 else "#27ae60" for v in grouped.to_numpy()]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(grouped.index, grouped.to_numpy(), color=colors)
        ax.set_title("Category-wise Returns", fontsize=14, fontweight="bold")
        ax.set_xlabel("Category", fontsize=11)
        ax.set_ylabel("Return (%)", fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.legend(["Return %"], loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        return self._save(fig, "category_returns_bar.png")

    def nav_movement_line(self) -> str:
        """NAV movement line chart for the five most-traded funds.

        Returns:
            The path to the saved chart.
        """
        top_funds = (
            self.merged.groupby("FundID")["TransactionID"].count().nlargest(5).index
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        for fund_id in top_funds:
            series = (
                self.nav_history[self.nav_history["FundID"] == fund_id]
                .sort_values("Date")
            )
            if series.empty:
                continue
            ax.plot(series["Date"], series["NAV"], label=str(fund_id))
        ax.set_title("NAV Movement (Top 5 Traded Funds)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("NAV (\u20b9)", fontsize=11)
        ax.legend(loc="upper left", title="Fund ID")
        ax.grid(True, linestyle="--", alpha=0.4)
        return self._save(fig, "nav_movement_line.png")

    def top_investors_barh(self, top_investors: pd.DataFrame) -> str:
        """Top 10 investors horizontal bar chart.

        Args:
            top_investors: Dataframe holding the top investors by investment.

        Returns:
            The path to the saved chart.
        """
        top_10 = top_investors.nlargest(10, "CurrentValue").sort_values(
            "CurrentValue"
        )
        labels = (
            top_10["InvestorName"].astype(str)
            + " ("
            + top_10["InvestorID"].astype(str)
            + ")"
        )
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(labels, top_10["CurrentValue"].to_numpy(), color="#6c3483")
        ax.set_title(
            "Top 10 Investors by Portfolio Value", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Portfolio Value (\u20b9)", fontsize=11)
        ax.set_ylabel("Investor", fontsize=11)
        ax.legend(["Portfolio Value"], loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        return self._save(fig, "top_investors_barh.png")
