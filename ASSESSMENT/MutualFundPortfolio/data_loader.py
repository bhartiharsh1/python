"""Data loading, cleaning and merging for the Mutual Fund Portfolio project.

The :class:`DataLoader` class reads the four raw CSV files, removes duplicate
transactions, imputes missing values according to the business rules, removes
statistical outliers and finally merges everything into a single cleaned
dataframe used by the rest of the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from utils import DATA_DIR


class DataLoadError(Exception):
    """Raised when one or more source CSV files cannot be loaded."""


class DataLoader:
    """Load, clean and merge the raw mutual-fund CSV datasets.

    Attributes:
        data_dir: Directory containing the raw CSV files.
        logger: Logger used for execution tracing.
        investors: Raw investors dataframe.
        funds: Raw funds dataframe.
        transactions: Raw transactions dataframe.
        nav_history: Raw NAV-history dataframe.
    """

    REQUIRED_FILES: Dict[str, str] = {
        "investors": "investors.csv",
        "funds": "funds.csv",
        "transactions": "transactions.csv",
        "nav_history": "nav_history.csv",
    }

    def __init__(self, logger: logging.Logger, data_dir: str = DATA_DIR) -> None:
        """Initialise the loader.

        Args:
            logger: Configured logger instance.
            data_dir: Directory that holds the raw CSV files.
        """
        self.data_dir = data_dir
        self.logger = logger
        self.investors: pd.DataFrame = pd.DataFrame()
        self.funds: pd.DataFrame = pd.DataFrame()
        self.transactions: pd.DataFrame = pd.DataFrame()
        self.nav_history: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_data(self) -> None:
        """Read all four CSV files into dataframes.

        Raises:
            DataLoadError: If any required file is missing or unreadable.
        """
        self.logger.info("Loading raw CSV files from: %s", self.data_dir)
        for key, filename in self.REQUIRED_FILES.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                message = f"Required data file not found: {path}"
                self.logger.error(message)
                raise DataLoadError(message)
            try:
                frame = pd.read_csv(path)
            except (pd.errors.ParserError, OSError) as exc:
                message = f"Failed to read {path}: {exc}"
                self.logger.error(message)
                raise DataLoadError(message) from exc
            setattr(self, key, frame)
            self.logger.info("Loaded %s (%d rows)", filename, len(frame))

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #
    def _remove_duplicate_transactions(self) -> None:
        """Drop exact duplicate transaction rows."""
        before = len(self.transactions)
        self.transactions = self.transactions.drop_duplicates(
            subset=["TransactionID"], keep="first"
        ).reset_index(drop=True)
        removed = before - len(self.transactions)
        self.logger.info("Removed %d duplicate transactions", removed)

    def _handle_missing_values(self) -> None:
        """Impute missing values following the business rules.

        * ``AnnualIncome``  -> median income
        * ``ExpenseRatio``  -> mean expense ratio
        * ``NAV``           -> previous day's NAV (forward fill per fund)
        * ``RiskProfile``   -> ``"Moderate"``
        """
        # Annual income -> median.
        median_income = float(self.investors["AnnualIncome"].median())
        missing_income = int(self.investors["AnnualIncome"].isna().sum())
        self.investors["AnnualIncome"] = self.investors["AnnualIncome"].fillna(
            median_income
        )
        self.logger.info(
            "Filled %d missing AnnualIncome values with median %.2f",
            missing_income,
            median_income,
        )

        # Risk profile -> "Moderate".
        missing_risk = int(self.investors["RiskProfile"].isna().sum())
        self.investors["RiskProfile"] = self.investors["RiskProfile"].fillna("Moderate")
        self.logger.info(
            "Filled %d missing RiskProfile values with 'Moderate'", missing_risk
        )

        # Expense ratio -> mean.
        mean_expense = float(self.funds["ExpenseRatio"].mean())
        missing_expense = int(self.funds["ExpenseRatio"].isna().sum())
        self.funds["ExpenseRatio"] = self.funds["ExpenseRatio"].fillna(mean_expense)
        self.logger.info(
            "Filled %d missing ExpenseRatio values with mean %.4f",
            missing_expense,
            mean_expense,
        )

        # NAV -> forward fill per fund (previous day NAV).
        self.nav_history = self.nav_history.sort_values(["FundID", "Date"])
        missing_nav = int(self.nav_history["NAV"].isna().sum())
        self.nav_history["NAV"] = self.nav_history.groupby("FundID")["NAV"].ffill()
        # Back-fill any values still missing at the very start of a series.
        self.nav_history["NAV"] = self.nav_history.groupby("FundID")["NAV"].bfill()
        self.logger.info(
            "Forward-filled %d missing NAV values with previous day NAV", missing_nav
        )

    def _coerce_types(self) -> None:
        """Convert date and numeric columns to their proper dtypes."""
        self.transactions["TransactionDate"] = pd.to_datetime(
            self.transactions["TransactionDate"], errors="coerce"
        )
        self.nav_history["Date"] = pd.to_datetime(
            self.nav_history["Date"], errors="coerce"
        )
        numeric_txn_cols = ["Units", "NAV", "Amount"]
        for col in numeric_txn_cols:
            self.transactions[col] = pd.to_numeric(
                self.transactions[col], errors="coerce"
            )
        # Drop rows that failed date/numeric coercion outright.
        self.transactions = self.transactions.dropna(
            subset=["TransactionDate", "Amount", "Units", "NAV"]
        ).reset_index(drop=True)

    def _remove_outliers(self) -> None:
        """Remove statistical outliers from transactions and NAV history.

        * Investment amounts above the 99th percentile are dropped.
        * NAV daily changes beyond 3 standard deviations are dropped.
        """
        # Investment amount above 99th percentile.
        amount_threshold = float(np.percentile(self.transactions["Amount"], 99))
        before = len(self.transactions)
        self.transactions = self.transactions[
            self.transactions["Amount"] <= amount_threshold
        ].reset_index(drop=True)
        self.logger.info(
            "Removed %d transactions above 99th percentile amount (%.2f)",
            before - len(self.transactions),
            amount_threshold,
        )

        # NAV daily change beyond 3 standard deviations.
        self.nav_history = self.nav_history.sort_values(["FundID", "Date"])
        self.nav_history["NAVChange"] = self.nav_history.groupby("FundID")[
            "NAV"
        ].pct_change()
        change_std = float(self.nav_history["NAVChange"].std())
        change_mean = float(self.nav_history["NAVChange"].mean())
        lower = change_mean - 3 * change_std
        upper = change_mean + 3 * change_std
        before_nav = len(self.nav_history)
        mask = (
            self.nav_history["NAVChange"].isna()
            | self.nav_history["NAVChange"].between(lower, upper)
        )
        self.nav_history = self.nav_history[mask].reset_index(drop=True)
        self.logger.info(
            "Removed %d NAV records beyond 3 standard deviations",
            before_nav - len(self.nav_history),
        )

    def clean_data(self) -> None:
        """Run the full cleaning pipeline on the loaded dataframes."""
        self.logger.info("Starting data cleaning")
        self._remove_duplicate_transactions()
        self._coerce_types()
        self._handle_missing_values()
        self._remove_outliers()
        self.logger.info("Data cleaning completed")

    # ------------------------------------------------------------------ #
    # Merging
    # ------------------------------------------------------------------ #
    def _latest_nav(self) -> pd.DataFrame:
        """Return the most recent NAV per fund.

        Returns:
            A dataframe with columns ``FundID`` and ``LatestNAV``.
        """
        latest = (
            self.nav_history.sort_values(["FundID", "Date"])
            .groupby("FundID")
            .tail(1)[["FundID", "NAV"]]
            .rename(columns={"NAV": "LatestNAV"})
            .reset_index(drop=True)
        )
        return latest

    def _average_nav(self) -> pd.DataFrame:
        """Return the average daily NAV per fund.

        Returns:
            A dataframe with columns ``FundID`` and ``AvgDailyNAV``.
        """
        avg = (
            self.nav_history.groupby("FundID")["NAV"]
            .mean()
            .reset_index()
            .rename(columns={"NAV": "AvgDailyNAV"})
        )
        return avg

    def merge_data(self) -> pd.DataFrame:
        """Merge investors, transactions, funds and NAV history.

        Returns:
            A single cleaned and merged dataframe.
        """
        self.logger.info("Merging datasets into a single dataframe")
        merged = self.transactions.merge(self.investors, on="InvestorID", how="left")
        merged = merged.merge(self.funds, on="FundID", how="left")
        merged = merged.merge(self._latest_nav(), on="FundID", how="left")
        merged = merged.merge(self._average_nav(), on="FundID", how="left")

        # Signed amount/units so redemptions reduce holdings.
        sign = np.where(merged["TransactionType"] == "Redemption", -1.0, 1.0)
        merged["SignedUnits"] = merged["Units"] * sign
        merged["SignedAmount"] = merged["Amount"] * sign

        # Current value of the units transacted using latest NAV.
        merged["CurrentValue"] = merged["SignedUnits"] * merged["LatestNAV"]

        self.logger.info(
            "Merged dataframe created with %d rows and %d columns",
            merged.shape[0],
            merged.shape[1],
        )
        return merged
