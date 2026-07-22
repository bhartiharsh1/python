"""Analytics engine for the Mutual Fund Portfolio project.

The :class:`PortfolioAnalytics` class computes the NumPy statistics, investor
analysis, fund analysis and portfolio-level financial metrics required by the
case study.  All heavy lifting operates on the cleaned, merged dataframe
produced by :class:`data_loader.DataLoader`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    percentage,
    safe_divide,
)

# Business thresholds expressed in rupees.
TEN_LAKHS: float = 1_000_000.0
FIFTEEN_LAKHS: float = 1_500_000.0


class PortfolioAnalytics:
    """Compute statistics, investor/fund analysis and financial metrics.

    Attributes:
        merged: The cleaned, merged transaction-level dataframe.
        investors: The cleaned investors dataframe.
        funds: The cleaned funds dataframe.
        nav_history: The cleaned NAV-history dataframe.
        logger: Logger used for execution tracing.
    """

    def __init__(
        self,
        merged: pd.DataFrame,
        investors: pd.DataFrame,
        funds: pd.DataFrame,
        nav_history: pd.DataFrame,
        logger: logging.Logger,
    ) -> None:
        """Initialise the analytics engine.

        Args:
            merged: Cleaned, merged transaction-level dataframe.
            investors: Cleaned investors dataframe.
            funds: Cleaned funds dataframe.
            nav_history: Cleaned NAV-history dataframe.
            logger: Configured logger instance.
        """
        self.merged = merged
        self.investors = investors
        self.funds = funds
        self.nav_history = nav_history
        self.logger = logger

    # ------------------------------------------------------------------ #
    # NumPy statistics
    # ------------------------------------------------------------------ #
    def numpy_statistics(self) -> Dict[str, float]:
        """Compute the core NumPy statistics.

        Returns:
            A dictionary containing mean investment amount, median investor
            income, NAV standard deviation, 90th/95th percentile fund returns
            and the correlation between annual income, investment amount and
            average daily NAV.
        """
        self.logger.info("Computing NumPy statistics")
        amounts = self.merged["Amount"].to_numpy(dtype=float)
        incomes = self.investors["AnnualIncome"].to_numpy(dtype=float)
        navs = self.nav_history["NAV"].to_numpy(dtype=float)

        fund_returns = self._fund_returns_series().to_numpy(dtype=float)
        fund_returns = fund_returns[~np.isnan(fund_returns)]

        stats: Dict[str, float] = {
            "mean_investment_amount": float(np.mean(amounts)),
            "median_investor_income": float(np.median(incomes)),
            "std_nav": float(np.std(navs)),
            "average_daily_nav": float(np.mean(navs)),
            "p90_fund_returns": float(np.percentile(fund_returns, 90))
            if fund_returns.size
            else 0.0,
            "p95_fund_returns": float(np.percentile(fund_returns, 95))
            if fund_returns.size
            else 0.0,
        }

        correlations = self._income_investment_nav_correlation()
        stats.update(correlations)
        self.logger.info("NumPy statistics computed: %s", stats)
        return stats

    def _income_investment_nav_correlation(self) -> Dict[str, float]:
        """Correlate annual income, investment amount and average daily NAV.

        Returns:
            A dictionary of pairwise Pearson correlation coefficients.
        """
        agg = (
            self.merged.groupby("InvestorID")
            .agg(
                Investment=("Amount", "sum"),
                AnnualIncome=("AnnualIncome", "first"),
                AvgDailyNAV=("AvgDailyNAV", "mean"),
            )
            .dropna()
        )
        if len(agg) < 2:
            return {
                "corr_income_investment": 0.0,
                "corr_income_nav": 0.0,
                "corr_investment_nav": 0.0,
            }
        matrix = np.corrcoef(
            [
                agg["AnnualIncome"].to_numpy(dtype=float),
                agg["Investment"].to_numpy(dtype=float),
                agg["AvgDailyNAV"].to_numpy(dtype=float),
            ]
        )
        return {
            "corr_income_investment": float(matrix[0, 1]),
            "corr_income_nav": float(matrix[0, 2]),
            "corr_investment_nav": float(matrix[1, 2]),
        }

    def _fund_returns_series(self) -> pd.Series:
        """Compute the total return of every fund over the NAV history window.

        Returns:
            A series indexed by ``FundID`` holding the percentage return.
        """
        ordered = self.nav_history.sort_values(["FundID", "Date"])
        grouped = ordered.groupby("FundID")["NAV"]
        first = grouped.first()
        last = grouped.last()
        returns = (last - first) / first * 100.0
        return returns

    # ------------------------------------------------------------------ #
    # Investor analysis
    # ------------------------------------------------------------------ #
    def _investor_summary(self) -> pd.DataFrame:
        """Build a per-investor summary dataframe.

        Returns:
            A dataframe indexed by ``InvestorID`` with aggregated metrics.
        """
        summary = self.merged.groupby("InvestorID").agg(
            InvestorName=("InvestorName", "first"),
            City=("City", "first"),
            AnnualIncome=("AnnualIncome", "first"),
            RiskProfile=("RiskProfile", "first"),
            TotalInvestment=("Amount", "sum"),
            NetInvested=("SignedAmount", "sum"),
            TransactionCount=("TransactionID", "count"),
            CurrentValue=("CurrentValue", "sum"),
        )
        summary["ProfitLoss"] = summary["CurrentValue"] - summary["NetInvested"]
        return summary.reset_index()

    def investor_analysis(self) -> Dict[str, Any]:
        """Perform the investor-level analysis required by the case study.

        Returns:
            A dictionary with the top investors, large investors, high-risk
            investors, active investors and per-investor profit/loss.
        """
        self.logger.info("Performing investor analysis")
        summary = self._investor_summary()

        # Top 20 investors ranked by current portfolio value.
        top_20 = summary.nlargest(20, "CurrentValue")

        large_investors = summary[summary["TotalInvestment"] > TEN_LAKHS]

        high_risk = summary[
            (summary["RiskProfile"] == "High")
            | (summary["TransactionCount"] > 10)
            | (summary["AnnualIncome"] > FIFTEEN_LAKHS)
        ]

        active_investors = summary[summary["TransactionCount"] > 10]

        profit_loss = summary[
            ["InvestorID", "InvestorName", "NetInvested", "CurrentValue", "ProfitLoss"]
        ].sort_values("ProfitLoss", ascending=False)

        self.logger.info(
            "Investor analysis complete: %d large, %d high-risk, %d active",
            len(large_investors),
            len(high_risk),
            len(active_investors),
        )
        return {
            "summary": summary,
            "top_20_investors": top_20,
            "large_investors": large_investors,
            "high_risk_investors": high_risk,
            "active_investors": active_investors,
            "profit_loss": profit_loss,
        }

    # ------------------------------------------------------------------ #
    # Fund analysis
    # ------------------------------------------------------------------ #
    def _fund_summary(self) -> pd.DataFrame:
        """Build a per-fund summary dataframe.

        Returns:
            A dataframe with aggregated per-fund metrics and returns.
        """
        summary = self.merged.groupby("FundID").agg(
            FundName=("FundName", "first"),
            Category=("Category", "first"),
            ExpenseRatio=("ExpenseRatio", "first"),
            TotalInvestment=("Amount", "sum"),
            NetInvested=("SignedAmount", "sum"),
            InvestorCount=("InvestorID", "nunique"),
            TransactionCount=("TransactionID", "count"),
            CurrentValue=("CurrentValue", "sum"),
        )
        fund_returns = self._fund_returns_series().rename("ReturnPct")
        summary = summary.join(fund_returns, how="left")
        summary["AUM"] = summary["CurrentValue"]
        return summary.reset_index()

    def fund_analysis(self) -> Dict[str, Any]:
        """Perform the fund-level analysis required by the case study.

        Returns:
            A dictionary describing best/worst funds, expense-ratio and AUM
            leaders and the most popular fund.
        """
        self.logger.info("Performing fund analysis")
        summary = self._fund_summary()

        best_fund = summary.loc[summary["ReturnPct"].idxmax()]
        worst_fund = summary.loc[summary["ReturnPct"].idxmin()]
        highest_expense = summary.loc[summary["ExpenseRatio"].idxmax()]
        highest_aum = summary.loc[summary["AUM"].idxmax()]
        most_popular = summary.loc[summary["InvestorCount"].idxmax()]

        self.logger.info(
            "Fund analysis complete. Best fund: %s, Worst fund: %s",
            best_fund["FundName"],
            worst_fund["FundName"],
        )
        return {
            "summary": summary,
            "best_fund": best_fund,
            "worst_fund": worst_fund,
            "highest_expense_ratio": highest_expense,
            "highest_aum": highest_aum,
            "most_popular_fund": most_popular,
        }

    # ------------------------------------------------------------------ #
    # Financial metrics
    # ------------------------------------------------------------------ #
    def financial_metrics(self) -> Dict[str, Any]:
        """Compute the portfolio-level financial metrics.

        Returns:
            A dictionary of scalar metrics plus category/fund allocation and
            investor-wise profit/loss breakdowns.
        """
        self.logger.info("Computing financial metrics")
        total_invested = float(self.merged["SignedAmount"].sum())
        total_value = float(self.merged["CurrentValue"].sum())
        absolute_return = total_value - total_invested
        portfolio_return_pct = percentage(absolute_return, total_invested)

        avg_holding_days = self._average_holding_period()
        years = safe_divide(avg_holding_days, 365.0, default=1.0) or 1.0

        cagr = self._cagr(total_invested, total_value, years)
        annualized_return = self._annualized_return(portfolio_return_pct, years)
        diversification = self._diversification_score()
        expense_impact = self._expense_ratio_impact()
        sharpe = self._sharpe_ratio()

        category_pct = self._category_investment_percentage()
        fund_allocation = self._fund_allocation_percentage()
        investor_pl = self._investor_wise_profit_loss()

        metrics: Dict[str, Any] = {
            "total_portfolio_value": total_value,
            "total_invested": total_invested,
            "absolute_return": absolute_return,
            "portfolio_return_pct": portfolio_return_pct,
            "annualized_return_pct": annualized_return,
            "cagr_pct": cagr,
            "diversification_score": diversification,
            "average_holding_period_days": avg_holding_days,
            "expense_ratio_impact": expense_impact,
            "sharpe_ratio": sharpe,
            "category_investment_pct": category_pct,
            "fund_allocation_pct": fund_allocation,
            "investor_profit_loss": investor_pl,
        }
        self.logger.info(
            "Financial metrics computed. Portfolio value: %.2f, Return: %.2f%%",
            total_value,
            portfolio_return_pct,
        )
        return metrics

    def _average_holding_period(self) -> float:
        """Compute the average holding period in days across transactions.

        Returns:
            The mean number of days between each transaction and the latest
            NAV date in the history.
        """
        latest_date = self.nav_history["Date"].max()
        deltas = (latest_date - self.merged["TransactionDate"]).dt.days
        deltas = deltas[deltas >= 0]
        if deltas.empty:
            return 0.0
        return float(deltas.mean())

    @staticmethod
    def _cagr(invested: float, value: float, years: float) -> float:
        """Compute the compound annual growth rate as a percentage.

        Args:
            invested: Net amount invested (cost basis).
            value: Current portfolio value.
            years: Investment horizon in years.

        Returns:
            The CAGR expressed as a percentage.
        """
        if invested <= 0 or value <= 0 or years <= 0:
            return 0.0
        return (float((value / invested) ** (1.0 / years)) - 1.0) * 100.0

    @staticmethod
    def _annualized_return(total_return_pct: float, years: float) -> float:
        """Annualise a total return percentage.

        Args:
            total_return_pct: The cumulative return percentage.
            years: Investment horizon in years.

        Returns:
            The annualised return expressed as a percentage.
        """
        if years <= 0:
            return total_return_pct
        growth = 1.0 + total_return_pct / 100.0
        if growth <= 0:
            return -100.0
        return (float(growth ** (1.0 / years)) - 1.0) * 100.0

    def _diversification_score(self) -> float:
        """Compute a diversification score based on the Herfindahl index.

        Returns:
            A score between 0 and 100, where higher means more diversified.
        """
        fund_value = self.merged.groupby("FundID")["Amount"].sum()
        total = float(fund_value.sum())
        if total <= 0:
            return 0.0
        weights = fund_value.to_numpy(dtype=float) / total
        herfindahl = float(np.sum(weights ** 2))
        return (1.0 - herfindahl) * 100.0

    def _expense_ratio_impact(self) -> float:
        """Estimate the annual cost of expense ratios across the portfolio.

        Returns:
            The weighted expense-ratio cost in rupees.
        """
        fund_value = self.merged.groupby("FundID").agg(
            Value=("Amount", "sum"),
            ExpenseRatio=("ExpenseRatio", "first"),
        )
        impact = (fund_value["Value"] * fund_value["ExpenseRatio"] / 100.0).sum()
        return float(impact)

    def _sharpe_ratio(self) -> float:
        """Compute a simplified Sharpe ratio from daily portfolio-proxy returns.

        The proxy uses the mean daily NAV returns across all funds as the
        portfolio return stream.

        Returns:
            The annualised simplified Sharpe ratio.
        """
        ordered = self.nav_history.sort_values(["FundID", "Date"])
        daily_returns = ordered.groupby("FundID")["NAV"].pct_change()
        daily_returns = daily_returns.dropna()
        if daily_returns.empty:
            return 0.0
        mean_daily = float(daily_returns.mean())
        std_daily = float(daily_returns.std())
        if std_daily == 0:
            return 0.0
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        sharpe_daily = (mean_daily - daily_rf) / std_daily
        return float(sharpe_daily * np.sqrt(TRADING_DAYS_PER_YEAR))

    def _category_investment_percentage(self) -> pd.DataFrame:
        """Compute investment percentage per fund category.

        Returns:
            A dataframe with columns ``Category``, ``Investment`` and
            ``Percentage``.
        """
        grouped = self.merged.groupby("Category")["Amount"].sum()
        total = float(grouped.sum())
        result = grouped.reset_index().rename(columns={"Amount": "Investment"})
        result["Percentage"] = result["Investment"].apply(
            lambda value: percentage(value, total)
        )
        return result.sort_values("Percentage", ascending=False).reset_index(drop=True)

    def _fund_allocation_percentage(self) -> pd.DataFrame:
        """Compute allocation percentage per fund.

        Returns:
            A dataframe with columns ``FundID``, ``FundName``, ``Investment``
            and ``Percentage``.
        """
        grouped = self.merged.groupby(["FundID", "FundName"])["Amount"].sum()
        total = float(grouped.sum())
        result = grouped.reset_index().rename(columns={"Amount": "Investment"})
        result["Percentage"] = result["Investment"].apply(
            lambda value: percentage(value, total)
        )
        return result.sort_values("Percentage", ascending=False).reset_index(drop=True)

    def _investor_wise_profit_loss(self) -> pd.DataFrame:
        """Compute profit/loss for every investor.

        Returns:
            A dataframe of per-investor net invested amount, current value and
            profit/loss.
        """
        summary = self._investor_summary()
        return summary[
            ["InvestorID", "InvestorName", "NetInvested", "CurrentValue", "ProfitLoss"]
        ].sort_values("ProfitLoss", ascending=False).reset_index(drop=True)
