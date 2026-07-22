# Mutual Fund Portfolio Performance & Risk Analysis

A production-quality Python application that analyses mutual-fund investment
data to produce portfolio performance metrics, risk statistics, investor and
fund insights, professional charts and CSV reports.

---

## Project Overview

The tool ingests four datasets (investors, funds, transactions and NAV
history), cleans and merges them, and then computes a comprehensive set of
analytics:

- **NumPy statistics** — mean investment, median income, NAV standard
  deviation, 90th/95th percentile fund returns and income/investment/NAV
  correlations.
- **Investor analysis** — top investors, large investors, high-risk investors,
  active investors and per-investor profit/loss.
- **Fund analysis** — best/worst performing funds, highest expense ratio,
  highest AUM and most popular fund.
- **Financial metrics** — total portfolio value, portfolio return %, absolute
  return, annualized return, CAGR, diversification score, average holding
  period, expense-ratio impact, a simplified Sharpe ratio, category-wise
  investment %, fund allocation % and investor-wise profit/loss.
- **Visualizations** — six professional matplotlib charts.
- **Reports** — four CSV reports.

The code is organised into cohesive, single-responsibility modules and a
central `FundPortfolio` orchestration class. It uses type hints, docstrings,
logging and exception handling throughout, and follows PEP-8.

---

## Folder Structure

```
MutualFundPortfolio/
│
├── data/
│   ├── investors.csv          # 500 investors
│   ├── funds.csv              # 50 mutual funds
│   ├── transactions.csv       # 10,000 transactions
│   └── nav_history.csv        # 365 days of NAV history per fund
│
├── reports/                   # Generated CSV reports
├── charts/                    # Generated PNG charts
├── logs/                      # Timestamped execution logs
│
├── fund_portfolio.py          # FundPortfolio orchestration class
├── data_loader.py             # Loading, cleaning, merging, outlier removal
├── analytics.py               # NumPy stats, investor/fund analysis, metrics
├── visualizations.py          # matplotlib chart generation
├── report_generator.py        # CSV report export
├── utils.py                   # Logging + reusable helpers
├── generate_data.py           # Realistic sample-data generator
├── main.py                    # Program entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Requirements

- Python 3.9 or newer
- Dependencies (see `requirements.txt`):
  - `pandas`
  - `numpy`
  - `matplotlib`

---

## Installation

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

On the first run, if the CSV files under `data/` are missing, the application
automatically generates a realistic sample dataset (500 investors, 50 funds,
10,000 transactions and 365 days of NAV history). To (re)generate the sample
data explicitly:

```bash
python generate_data.py
```

The program executes the full pipeline in order: **load → clean → merge →
analyse → chart → report → summary**, and writes a timestamped execution log to
`logs/`.

---

## Output Description

### Console
A formatted summary of portfolio value, returns, risk metrics, NumPy
statistics and fund/investor highlights.

### Charts (`charts/`)
| File | Description |
| --- | --- |
| `portfolio_allocation_pie.png` | Portfolio allocation by fund (pie chart) |
| `fund_investment_bar.png` | Top funds by total investment (bar chart) |
| `monthly_investment_trend.png` | Monthly investment trend (line chart) |
| `category_returns_bar.png` | Category-wise returns (bar chart) |
| `nav_movement_line.png` | NAV movement for top-traded funds (line chart) |
| `top_investors_barh.png` | Top 10 investors (horizontal bar chart) |

All charts include titles, axis labels, legends, tight layout and are saved at
high resolution.

### Reports (`reports/`)
| File | Description |
| --- | --- |
| `portfolio_summary.csv` | High-level portfolio summary |
| `fund_analysis.csv` | Per-fund metrics and returns |
| `investor_analysis.csv` | Per-investor metrics and profit/loss |
| `financial_metrics.csv` | Detailed financial metrics and allocations |

### Logs (`logs/`)
Each execution produces a timestamped `execution_YYYYMMDD_HHMMSS.log` capturing
every pipeline stage, imputation counts and outlier removals.

---

## Data Cleaning Rules

- **Duplicate transactions** are removed by `TransactionID`.
- **Missing values** are imputed as follows:
  - `AnnualIncome` → median
  - `ExpenseRatio` → mean
  - `NAV` → previous day's NAV (forward fill per fund)
  - `RiskProfile` → `"Moderate"`
- **Outliers** are removed:
  - Investment amounts above the 99th percentile
  - NAV daily changes beyond 3 standard deviations

---

## Architecture Notes

- **OOP** — `FundPortfolio`, `DataLoader`, `PortfolioAnalytics`,
  `PortfolioVisualizer` and `ReportGenerator` each own one responsibility.
- **Reusable functions** live in `utils.py` (logging, directory management,
  currency formatting, safe division, percentages).
- **Exception handling** — custom `DataLoadError` plus a top-level guard in
  `main.py`.
- **Logging** — dual file + console handlers configured in `utils.setup_logger`.
- **No global mutable state**; configuration constants are module-level and
  immutable.
```
