"""Report generation for the Mutual Fund Portfolio project.

The :class:`ReportGenerator` exports the four required CSV reports into the
``reports`` directory:

* ``portfolio_summary.csv``
* ``fund_analysis.csv``
* ``investor_analysis.csv``
* ``financial_metrics.csv``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import pandas as pd

from utils import REPORTS_DIR, ensure_directories, get_timestamp


class ReportGenerator:
    """Persist analysis results as CSV reports.

    Attributes:
        logger: Logger used for execution tracing.
        reports_dir: Directory where CSV reports are written.
    """

    def __init__(
        self, logger: logging.Logger, reports_dir: str = REPORTS_DIR
    ) -> None:
        """Initialise the report generator.

        Args:
            logger: Configured logger instance.
            reports_dir: Directory where reports are saved.
        """
        self.logger = logger
        self.reports_dir = reports_dir
        ensure_directories(self.reports_dir)

    def _write(self, frame: pd.DataFrame, filename: str) -> str:
        """Write a dataframe to CSV and return its path.

        Args:
            frame: The dataframe to persist.
            filename: The target file name inside the reports directory.

        Returns:
            The absolute path to the written CSV file.
        """
        path = os.path.join(self.reports_dir, filename)
        frame.to_csv(path, index=False)
        self.logger.info("Wrote report: %s (%d rows)", path, len(frame))
        return path

    def export_all(
        self,
        investor_analysis: Dict[str, Any],
        fund_analysis: Dict[str, Any],
        metrics: Dict[str, Any],
        statistics: Dict[str, float],
    ) -> List[str]:
        """Export all four CSV reports.

        Args:
            investor_analysis: Result of the investor analysis.
            fund_analysis: Result of the fund analysis.
            metrics: Result of the financial-metrics computation.
            statistics: Result of the NumPy statistics computation.

        Returns:
            A list of the report file paths that were written.
        """
        self.logger.info("Exporting CSV reports")
        paths = [
            self._export_portfolio_summary(metrics, statistics, investor_analysis, fund_analysis),
            self._export_fund_analysis(fund_analysis),
            self._export_investor_analysis(investor_analysis),
            self._export_financial_metrics(metrics, statistics),
        ]
        self.logger.info("Exported %d reports", len(paths))
        return paths

    def _export_portfolio_summary(
        self,
        metrics: Dict[str, Any],
        statistics: Dict[str, float],
        investor_analysis: Dict[str, Any],
        fund_analysis: Dict[str, Any],
    ) -> str:
        """Export the high-level portfolio summary report."""
        rows = [
            ("Generated At", get_timestamp()),
            ("Total Investors", investor_analysis["summary"].shape[0]),
            ("Total Funds", fund_analysis["summary"].shape[0]),
            ("Total Portfolio Value", round(metrics["total_portfolio_value"], 2)),
            ("Total Invested", round(metrics["total_invested"], 2)),
            ("Absolute Return", round(metrics["absolute_return"], 2)),
            ("Portfolio Return %", round(metrics["portfolio_return_pct"], 2)),
            ("Annualized Return %", round(metrics["annualized_return_pct"], 2)),
            ("CAGR %", round(metrics["cagr_pct"], 2)),
            ("Diversification Score", round(metrics["diversification_score"], 2)),
            (
                "Average Holding Period (days)",
                round(metrics["average_holding_period_days"], 2),
            ),
            ("Expense Ratio Impact", round(metrics["expense_ratio_impact"], 2)),
            ("Sharpe Ratio", round(metrics["sharpe_ratio"], 4)),
            ("Mean Investment Amount", round(statistics["mean_investment_amount"], 2)),
            ("Median Investor Income", round(statistics["median_investor_income"], 2)),
            ("Std Deviation of NAV", round(statistics["std_nav"], 4)),
            ("Average Daily NAV", round(statistics["average_daily_nav"], 4)),
        ]
        frame = pd.DataFrame(rows, columns=["Metric", "Value"])
        return self._write(frame, "portfolio_summary.csv")

    def _export_fund_analysis(self, fund_analysis: Dict[str, Any]) -> str:
        """Export the per-fund analysis report."""
        frame = fund_analysis["summary"].copy()
        numeric_cols = [
            "ExpenseRatio",
            "TotalInvestment",
            "NetInvested",
            "CurrentValue",
            "AUM",
            "ReturnPct",
        ]
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = frame[col].round(2)
        return self._write(frame, "fund_analysis.csv")

    def _export_investor_analysis(self, investor_analysis: Dict[str, Any]) -> str:
        """Export the per-investor analysis report."""
        frame = investor_analysis["summary"].copy()
        numeric_cols = [
            "AnnualIncome",
            "TotalInvestment",
            "NetInvested",
            "CurrentValue",
            "ProfitLoss",
        ]
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = frame[col].round(2)
        return self._write(frame, "investor_analysis.csv")

    def _export_financial_metrics(
        self, metrics: Dict[str, Any], statistics: Dict[str, float]
    ) -> str:
        """Export the detailed financial-metrics report."""
        scalar_rows = [
            ("Total Portfolio Value", round(metrics["total_portfolio_value"], 2)),
            ("Total Invested", round(metrics["total_invested"], 2)),
            ("Absolute Return", round(metrics["absolute_return"], 2)),
            ("Portfolio Return %", round(metrics["portfolio_return_pct"], 2)),
            ("Annualized Return %", round(metrics["annualized_return_pct"], 2)),
            ("CAGR %", round(metrics["cagr_pct"], 2)),
            ("Diversification Score", round(metrics["diversification_score"], 2)),
            (
                "Average Holding Period (days)",
                round(metrics["average_holding_period_days"], 2),
            ),
            ("Expense Ratio Impact", round(metrics["expense_ratio_impact"], 2)),
            ("Sharpe Ratio", round(metrics["sharpe_ratio"], 4)),
            ("90th Percentile Fund Returns", round(statistics["p90_fund_returns"], 4)),
            ("95th Percentile Fund Returns", round(statistics["p95_fund_returns"], 4)),
            ("Average Daily NAV", round(statistics["average_daily_nav"], 4)),
            (
                "Corr(Income, Investment)",
                round(statistics["corr_income_investment"], 4),
            ),
            ("Corr(Income, Avg NAV)", round(statistics["corr_income_nav"], 4)),
            (
                "Corr(Investment, Avg NAV)",
                round(statistics["corr_investment_nav"], 4),
            ),
        ]
        scalar_frame = pd.DataFrame(scalar_rows, columns=["Metric", "Value"])

        category_frame = metrics["category_investment_pct"].copy()
        category_frame.insert(0, "Section", "Category Investment %")
        category_frame = category_frame.rename(
            columns={"Category": "Metric", "Percentage": "Value"}
        )[["Section", "Metric", "Value"]]
        category_frame["Value"] = category_frame["Value"].round(2)

        scalar_frame.insert(0, "Section", "Portfolio Metric")
        combined = pd.concat([scalar_frame, category_frame], ignore_index=True)
        return self._write(combined, "financial_metrics.csv")
