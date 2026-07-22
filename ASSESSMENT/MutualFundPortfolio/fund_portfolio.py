"""The central :class:`FundPortfolio` orchestration class.

This module ties together data loading, cleaning, analytics, visualisation and
reporting behind a single, cohesive object-oriented interface, exactly as
required by the case study.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics import PortfolioAnalytics
from data_loader import DataLoader
from report_generator import ReportGenerator
from utils import (
    CHARTS_DIR,
    DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    ensure_directories,
    format_currency,
    setup_logger,
)
from visualizations import PortfolioVisualizer


class FundPortfolio:
    """End-to-end mutual-fund portfolio performance and risk analyser.

    The class exposes discrete methods for each stage of the pipeline so that
    callers can run them individually or through :meth:`run_full_pipeline`.

    Attributes:
        logger: Configured logger instance.
        loader: The data loader/cleaner.
        merged: Cleaned, merged transaction-level dataframe.
        analytics: The analytics engine (created after data is prepared).
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the portfolio analyser and required directories.

        Args:
            data_dir: Directory containing the raw CSV files.
            logger: Optional pre-configured logger; a new one is created when
                omitted.
        """
        ensure_directories(DATA_DIR, REPORTS_DIR, CHARTS_DIR, LOGS_DIR)
        self.logger = logger or setup_logger()
        self.loader = DataLoader(self.logger, data_dir=data_dir)
        self.merged: pd.DataFrame = pd.DataFrame()
        self.analytics: Optional[PortfolioAnalytics] = None

        self._statistics: Dict[str, float] = {}
        self._investor_analysis: Dict[str, Any] = {}
        self._fund_analysis: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def load_data(self) -> None:
        """Load the raw CSV datasets."""
        self.loader.load_data()

    def clean_data(self) -> None:
        """Clean the loaded datasets (duplicates, missing values, outliers)."""
        self.loader.clean_data()

    def merge_data(self) -> pd.DataFrame:
        """Merge the cleaned datasets into a single dataframe.

        Returns:
            The cleaned, merged dataframe.
        """
        self.merged = self.loader.merge_data()
        self.analytics = PortfolioAnalytics(
            merged=self.merged,
            investors=self.loader.investors,
            funds=self.loader.funds,
            nav_history=self.loader.nav_history,
            logger=self.logger,
        )
        return self.merged

    def _require_analytics(self) -> PortfolioAnalytics:
        """Return the analytics engine, raising if data is not prepared.

        Returns:
            The initialised :class:`PortfolioAnalytics` instance.

        Raises:
            RuntimeError: If :meth:`merge_data` has not been called yet.
        """
        if self.analytics is None:
            raise RuntimeError(
                "Data has not been prepared. Call load_data(), clean_data() "
                "and merge_data() before running analysis."
            )
        return self.analytics

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def compute_statistics(self) -> Dict[str, float]:
        """Compute the NumPy statistics.

        Returns:
            The statistics dictionary.
        """
        self._statistics = self._require_analytics().numpy_statistics()
        return self._statistics

    def portfolio_analysis(self) -> Dict[str, Any]:
        """Run investor and fund analysis and financial-metric computation.

        Returns:
            A dictionary combining investor analysis, fund analysis and
            financial metrics.
        """
        self.investor_analysis()
        self.fund_analysis()
        self.financial_metrics()
        return {
            "investor_analysis": self._investor_analysis,
            "fund_analysis": self._fund_analysis,
            "financial_metrics": self._metrics,
        }

    def investor_analysis(self) -> Dict[str, Any]:
        """Run the investor-level analysis.

        Returns:
            The investor-analysis dictionary.
        """
        self._investor_analysis = self._require_analytics().investor_analysis()
        return self._investor_analysis

    def fund_analysis(self) -> Dict[str, Any]:
        """Run the fund-level analysis.

        Returns:
            The fund-analysis dictionary.
        """
        self._fund_analysis = self._require_analytics().fund_analysis()
        return self._fund_analysis

    def financial_metrics(self) -> Dict[str, Any]:
        """Compute the portfolio-level financial metrics.

        Returns:
            The financial-metrics dictionary.
        """
        self._metrics = self._require_analytics().financial_metrics()
        return self._metrics

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def generate_charts(self) -> List[str]:
        """Generate and save all charts.

        Returns:
            A list of saved chart file paths.
        """
        visualizer = PortfolioVisualizer(
            merged=self.merged,
            nav_history=self.loader.nav_history,
            logger=self.logger,
        )
        return visualizer.generate_all_charts(self._investor_analysis, self._metrics)

    def export_reports(self) -> List[str]:
        """Export all CSV reports.

        Returns:
            A list of written report file paths.
        """
        generator = ReportGenerator(self.logger)
        return generator.export_all(
            investor_analysis=self._investor_analysis,
            fund_analysis=self._fund_analysis,
            metrics=self._metrics,
            statistics=self._statistics,
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def print_summary(self) -> None:
        """Print a human-readable summary of the analysis to the console."""
        metrics = self._metrics
        stats = self._statistics
        fund = self._fund_analysis
        investor = self._investor_analysis

        line = "=" * 70
        print(line)
        print("        MUTUAL FUND PORTFOLIO - PERFORMANCE & RISK SUMMARY")
        print(line)
        print(f"Transactions analysed      : {len(self.merged):,}")
        print(f"Total investors            : {investor['summary'].shape[0]:,}")
        print(f"Total funds                : {fund['summary'].shape[0]:,}")
        print(line)
        print(
            f"Total portfolio value      : "
            f"{format_currency(metrics['total_portfolio_value'])}"
        )
        print(
            f"Total invested             : "
            f"{format_currency(metrics['total_invested'])}"
        )
        print(
            f"Absolute return            : "
            f"{format_currency(metrics['absolute_return'])}"
        )
        print(f"Portfolio return           : {metrics['portfolio_return_pct']:.2f}%")
        print(f"Annualized return          : {metrics['annualized_return_pct']:.2f}%")
        print(f"CAGR                       : {metrics['cagr_pct']:.2f}%")
        print(f"Diversification score      : {metrics['diversification_score']:.2f}")
        print(
            f"Avg holding period (days)  : "
            f"{metrics['average_holding_period_days']:.1f}"
        )
        print(
            f"Expense ratio impact       : "
            f"{format_currency(metrics['expense_ratio_impact'])}"
        )
        print(f"Sharpe ratio (simplified)  : {metrics['sharpe_ratio']:.4f}")
        print(line)
        print("NumPy statistics")
        print(
            f"  Mean investment amount   : "
            f"{format_currency(stats['mean_investment_amount'])}"
        )
        print(
            f"  Median investor income   : "
            f"{format_currency(stats['median_investor_income'])}"
        )
        print(f"  Std deviation of NAV     : {stats['std_nav']:.4f}")
        print(
            f"  Average daily NAV        : "
            f"{format_currency(stats['average_daily_nav'])}"
        )
        print(f"  90th pct fund returns    : {stats['p90_fund_returns']:.2f}%")
        print(f"  95th pct fund returns    : {stats['p95_fund_returns']:.2f}%")
        print(
            f"  Corr(income, investment) : {stats['corr_income_investment']:.4f}"
        )
        print(line)
        print("Fund highlights")
        print(
            f"  Best performing fund     : {fund['best_fund']['FundName']} "
            f"({fund['best_fund']['ReturnPct']:.2f}%)"
        )
        print(
            f"  Worst performing fund    : {fund['worst_fund']['FundName']} "
            f"({fund['worst_fund']['ReturnPct']:.2f}%)"
        )
        print(
            f"  Highest expense ratio    : "
            f"{fund['highest_expense_ratio']['FundName']} "
            f"({fund['highest_expense_ratio']['ExpenseRatio']:.2f}%)"
        )
        print(
            f"  Highest AUM fund         : {fund['highest_aum']['FundName']} "
            f"({format_currency(fund['highest_aum']['AUM'])})"
        )
        print(
            f"  Most popular fund        : "
            f"{fund['most_popular_fund']['FundName']} "
            f"({int(fund['most_popular_fund']['InvestorCount'])} investors)"
        )
        print(line)
        print("Investor highlights")
        top = investor["top_20_investors"].iloc[0]
        print(
            f"  Top investor             : {top['InvestorName']} "
            f"({format_currency(top['CurrentValue'])} portfolio value)"
        )
        print(f"  Investors > 10 Lakhs     : {len(investor['large_investors']):,}")
        print(f"  High-risk investors      : {len(investor['high_risk_investors']):,}")
        print(f"  Active (>10 txns)        : {len(investor['active_investors']):,}")
        print(line)

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def run_full_pipeline(self) -> None:
        """Execute the complete analysis pipeline end to end."""
        self.logger.info("Starting full portfolio pipeline")
        self.load_data()
        self.clean_data()
        self.merge_data()
        self.compute_statistics()
        self.portfolio_analysis()
        self.generate_charts()
        self.export_reports()
        self.print_summary()
        self.logger.info("Full portfolio pipeline completed successfully")
