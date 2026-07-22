"""Generate realistic sample data for the Mutual Fund Portfolio project.

This script creates four CSV files inside the ``data`` directory:

* ``investors.csv``     -> 500 investors
* ``funds.csv``         -> 50 mutual funds
* ``transactions.csv``  -> 10,000 transactions
* ``nav_history.csv``   -> 365 days of NAV history for every fund

The generated data intentionally contains a small number of missing values
and duplicate rows so that the data-cleaning pipeline has real work to do.

Run it directly::

    python generate_data.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NUM_INVESTORS: int = 500
NUM_FUNDS: int = 50
NUM_TRANSACTIONS: int = 10_000
NUM_NAV_DAYS: int = 365

RANDOM_SEED: int = 42

DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FIRST_NAMES: List[str] = [
    "Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Isha",
    "Karan", "Meera", "Arjun", "Divya", "Aditya", "Kavya", "Nikhil", "Pooja",
    "Sanjay", "Neha", "Vivek", "Riya", "Manish", "Shreya", "Amit", "Tanvi",
    "Rajesh", "Anita", "Suresh", "Deepa", "Varun", "Nisha",
]

LAST_NAMES: List[str] = [
    "Sharma", "Nair", "Verma", "Reddy", "Iyer", "Patel", "Gupta", "Singh",
    "Mehta", "Rao", "Kapoor", "Joshi", "Malhotra", "Chopra", "Bose", "Das",
    "Kulkarni", "Menon", "Pillai", "Bhat",
]

CITIES: List[str] = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata",
    "Ahmedabad", "Jaipur", "Kochi", "Lucknow", "Chandigarh",
]

RISK_PROFILES: List[str] = ["Low", "Moderate", "High"]

FUND_HOUSES: List[str] = [
    "HDFC", "ICICI", "SBI", "Kotak", "Axis", "Nippon", "Aditya Birla",
    "Franklin", "DSP", "UTI", "Mirae", "Tata",
]

CATEGORIES: List[str] = [
    "Large Cap", "Mid Cap", "Small Cap", "Multi Cap", "ELSS",
    "Debt", "Hybrid", "Index",
]

MANAGERS: List[str] = [
    "Rajesh Kumar", "Priya Sharma", "Amit Verma", "Rohan Mehta", "Neha Gupta",
    "Sanjay Rao", "Divya Iyer", "Vikram Singh", "Ananya Bose", "Karan Kapoor",
]

BENCHMARKS = {
    "Large Cap": "Nifty 100 TRI",
    "Mid Cap": "Nifty Midcap 150 TRI",
    "Small Cap": "Nifty Smallcap 250 TRI",
    "Multi Cap": "Nifty 500 TRI",
    "ELSS": "Nifty 500 TRI",
    "Debt": "CRISIL Composite Bond Index",
    "Hybrid": "CRISIL Hybrid 35+65 Index",
    "Index": "Nifty 50 TRI",
}


def _ensure_data_dir() -> None:
    """Create the data directory if it does not already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def generate_investors(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the investors dataframe."""
    records = []
    for i in range(1, NUM_INVESTORS + 1):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        age = int(rng.integers(21, 70))
        city = str(rng.choice(CITIES))
        income = int(rng.normal(1_200_000, 600_000))
        income = max(250_000, income)
        risk = str(rng.choice(RISK_PROFILES, p=[0.3, 0.45, 0.25]))
        records.append(
            {
                "InvestorID": f"INV{i:03d}",
                "InvestorName": name,
                "Age": age,
                "City": city,
                "AnnualIncome": income,
                "RiskProfile": risk,
            }
        )
    df = pd.DataFrame(records)

    # Inject a handful of missing values for cleaning to handle.
    missing_income_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[missing_income_idx, "AnnualIncome"] = np.nan
    missing_risk_idx = rng.choice(df.index, size=12, replace=False)
    df.loc[missing_risk_idx, "RiskProfile"] = np.nan
    return df


def generate_funds(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the funds dataframe."""
    records = []
    for i in range(1, NUM_FUNDS + 1):
        house = str(rng.choice(FUND_HOUSES))
        category = str(rng.choice(CATEGORIES))
        fund_name = f"{house} {category} Fund"
        manager = str(rng.choice(MANAGERS))
        expense = round(float(rng.uniform(0.3, 2.2)), 2)
        records.append(
            {
                "FundID": f"F{i:03d}",
                "FundName": fund_name,
                "Category": category,
                "FundManager": manager,
                "ExpenseRatio": expense,
                "Benchmark": BENCHMARKS[category],
            }
        )
    df = pd.DataFrame(records)

    # Inject a few missing expense ratios.
    missing_exp_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[missing_exp_idx, "ExpenseRatio"] = np.nan
    return df


def generate_nav_history(rng: np.random.Generator, fund_ids: List[str]) -> pd.DataFrame:
    """Generate 365 days of NAV history for every fund using a random walk."""
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=d) for d in range(NUM_NAV_DAYS)]

    frames = []
    for fund_id in fund_ids:
        start_nav = float(rng.uniform(10, 250))
        drift = rng.uniform(-0.0003, 0.0008)
        vol = rng.uniform(0.004, 0.02)
        shocks = rng.normal(drift, vol, size=NUM_NAV_DAYS)
        nav_series = start_nav * np.cumprod(1 + shocks)
        frames.append(
            pd.DataFrame(
                {
                    "FundID": fund_id,
                    "Date": [d.isoformat() for d in dates],
                    "NAV": np.round(nav_series, 4),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    # Inject a few missing NAV values for forward-fill handling.
    missing_nav_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[missing_nav_idx, "NAV"] = np.nan
    return df


def generate_transactions(
    rng: np.random.Generator,
    investor_ids: List[str],
    fund_ids: List[str],
    nav_history: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the transactions dataframe using real NAVs from nav_history."""
    nav_lookup = {
        (row.FundID, row.Date): row.NAV
        for row in nav_history.dropna(subset=["NAV"]).itertuples(index=False)
    }
    available_dates = sorted(nav_history["Date"].unique())

    records = []
    for i in range(1, NUM_TRANSACTIONS + 1):
        investor_id = str(rng.choice(investor_ids))
        fund_id = str(rng.choice(fund_ids))
        txn_date = str(rng.choice(available_dates))
        nav = nav_lookup.get((fund_id, txn_date))
        if nav is None or np.isnan(nav):
            # Fall back to any known NAV for that fund.
            nav = float(
                nav_history.loc[nav_history["FundID"] == fund_id, "NAV"]
                .dropna()
                .iloc[0]
            )
        txn_type = str(rng.choice(["Purchase", "Redemption"], p=[0.75, 0.25]))
        units = int(rng.integers(10, 1000))
        amount = round(units * float(nav), 2)
        records.append(
            {
                "TransactionID": f"T{i:05d}",
                "InvestorID": investor_id,
                "FundID": fund_id,
                "TransactionDate": txn_date,
                "TransactionType": txn_type,
                "Units": units,
                "NAV": round(float(nav), 4),
                "Amount": amount,
            }
        )
    df = pd.DataFrame(records)

    # Inject duplicate transactions (exact copies) for de-duplication logic.
    dup_rows = df.sample(n=50, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


def main() -> None:
    """Generate all CSV files and write them to the data directory."""
    _ensure_data_dir()
    rng = np.random.default_rng(RANDOM_SEED)

    investors = generate_investors(rng)
    funds = generate_funds(rng)
    nav_history = generate_nav_history(rng, funds["FundID"].tolist())
    transactions = generate_transactions(
        rng, investors["InvestorID"].tolist(), funds["FundID"].tolist(), nav_history
    )

    investors.to_csv(os.path.join(DATA_DIR, "investors.csv"), index=False)
    funds.to_csv(os.path.join(DATA_DIR, "funds.csv"), index=False)
    transactions.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
    nav_history.to_csv(os.path.join(DATA_DIR, "nav_history.csv"), index=False)

    print("Sample data generated successfully in:", DATA_DIR)
    print(f"  investors.csv     -> {len(investors):,} rows")
    print(f"  funds.csv         -> {len(funds):,} rows")
    print(f"  transactions.csv  -> {len(transactions):,} rows")
    print(f"  nav_history.csv   -> {len(nav_history):,} rows")


if __name__ == "__main__":
    main()
