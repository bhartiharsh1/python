import numpy as np
import pandas as pd
from pathlib import Path
 
 
def read_data(base_path: Path) -> dict:
    return {
        "funds": pd.read_csv(base_path / "funds.csv"),
        "investors": pd.read_csv(base_path / "investors.csv"),
        "transactions": pd.read_csv(base_path / "transactions.csv"),
        "nav_history": pd.read_csv(base_path / "nav_history.csv"),
    }
 
 
def clean_data(data: dict) -> tuple[dict, dict]:
    cleaned = {}
    stats = {}
 
    for name, df in data.items():
        before = len(df)
        df = df.drop_duplicates().copy()
        after = len(df)
        stats[f"duplicates_removed_{name}"] = before - after
        cleaned[name] = df
 
    missing_before = {name: df.isna().sum().to_dict() for name, df in cleaned.items()}
 
    cleaned["nav_history"] = cleaned["nav_history"].sort_values(["FundID", "Date"]).copy()
    cleaned["nav_history"]["NAV"] = cleaned["nav_history"].groupby("FundID")["NAV"].ffill()
 
    cleaned["investors"]["InvestorType"] = cleaned["investors"]["InvestorType"].fillna("Retail")
 
    # Remove rows where NAV is negative after forward fill.
    cleaned["nav_history"] = cleaned["nav_history"][cleaned["nav_history"]["NAV"] >= 0].copy()
 
    # Convert date columns.
    cleaned["transactions"]["PurchaseDate"] = pd.to_datetime(
        cleaned["transactions"]["PurchaseDate"], errors="coerce"
    )
    cleaned["nav_history"]["Date"] = pd.to_datetime(cleaned["nav_history"]["Date"], errors="coerce")
 
    missing_after = {name: df.isna().sum().to_dict() for name, df in cleaned.items()}
    stats["missing_before"] = missing_before
    stats["missing_after"] = missing_after
 
    return cleaned, stats
 
 
def prepare_latest_nav(nav_history: pd.DataFrame) -> pd.DataFrame:
    latest = (
        nav_history.sort_values(["FundID", "Date"])
        .groupby("FundID", as_index=False)
        .tail(1)
        .loc[:, ["FundID", "NAV"]]
        .rename(columns={"NAV": "LatestNAV"})
    )
    return latest
 
 
def merge_data(cleaned: dict) -> pd.DataFrame:
    latest_nav = prepare_latest_nav(cleaned["nav_history"])
 
    merged = (
        cleaned["transactions"]
        .merge(cleaned["investors"], on="InvestorID", how="left")
        .merge(cleaned["funds"], on="FundID", how="left")
        .merge(latest_nav, on="FundID", how="left")
    )
 
    merged = merged.rename(
        columns={
            "InvestorName": "Investor Name",
            "FundName": "Fund Name",
            "InvestorType": "Investor Type",
            "UnitsPurchased": "Units Purchased",
            "PurchaseNAV": "Purchase NAV",
            "LatestNAV": "Latest NAV",
        }
    )
 
    return merged
 
 
def create_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Investment Amount"] = out["Units Purchased"] * out["Purchase NAV"]
    out["Current Value"] = out["Units Purchased"] * out["Latest NAV"]
    out["Profit"] = out["Current Value"] - out["Investment Amount"]
    out["ROI %"] = np.where(
        out["Investment Amount"] != 0,
        ((out["Current Value"] - out["Investment Amount"]) / out["Investment Amount"]) * 100,
        np.nan,
    )
    return out
 
 
def numpy_tasks(nav_history: pd.DataFrame) -> dict:
    nav_values = nav_history["NAV"].dropna().to_numpy()
 
    rolling_avg = (
        pd.Series(nav_values)
        .rolling(window=5)
        .mean()
        .to_numpy()
    )
 
    return {
        "Average NAV": float(np.mean(nav_values)) if len(nav_values) else np.nan,
        "Maximum NAV": float(np.max(nav_values)) if len(nav_values) else np.nan,
        "Minimum NAV": float(np.min(nav_values)) if len(nav_values) else np.nan,
        "Variance of NAV": float(np.var(nav_values)) if len(nav_values) else np.nan,
        "Standard Deviation of NAV": float(np.std(nav_values)) if len(nav_values) else np.nan,
        "Rolling Average (window=5)": rolling_avg,
    }
 
 
def pandas_analysis(merged_metrics: pd.DataFrame) -> dict:
    valid = merged_metrics.dropna(subset=["Latest NAV", "ROI %", "Profit"]).copy()
 
    top_investors = (
        merged_metrics.groupby(["InvestorID", "Investor Name"], as_index=False)["Investment Amount"]
        .sum()
        .sort_values("Investment Amount", ascending=False)
        .head(5)
    )
 
    fund_perf = (
        valid.groupby(["FundID", "Fund Name"], as_index=False)
        .agg(
            TotalInvestment=("Investment Amount", "sum"),
            TotalCurrentValue=("Current Value", "sum"),
            TotalProfit=("Profit", "sum"),
            AvgROI=("ROI %", "mean"),
            LatestNAV=("Latest NAV", "mean"),
        )
        .sort_values("TotalProfit", ascending=False)
    )
 
    top_profitable_funds = fund_perf.head(5)
    worst_fund = fund_perf.sort_values("AvgROI", ascending=True).head(1)
 
    highest_nav_fund = fund_perf.sort_values("LatestNAV", ascending=False).head(1)
    lowest_nav_fund = fund_perf.sort_values("LatestNAV", ascending=True).head(1)
 
    return {
        "top_investors": top_investors,
        "fund_performance": fund_perf,
        "top_profitable_funds": top_profitable_funds,
        "worst_fund": worst_fund,
        "highest_nav_fund": highest_nav_fund,
        "lowest_nav_fund": lowest_nav_fund,
    }
 
 
def groupby_analysis(merged_metrics: pd.DataFrame) -> dict:
    valid = merged_metrics.dropna(subset=["Latest NAV", "ROI %"]).copy()
 
    category_summary = (
        valid.groupby("Category", as_index=False)
        .agg(
            AverageROI=("ROI %", "mean"),
            AverageNAV=("Latest NAV", "mean"),
            TotalInvestment=("Investment Amount", "sum"),
        )
        .sort_values("TotalInvestment", ascending=False)
    )
 
    amc_summary = (
        valid.groupby("AMC", as_index=False)
        .agg(
            NumberOfFunds=("FundID", "nunique"),
            AverageNAV=("Latest NAV", "mean"),
            TotalInvestment=("Investment Amount", "sum"),
        )
        .sort_values("TotalInvestment", ascending=False)
    )
 
    state_summary = (
        valid.groupby("State", as_index=False)
        .agg(
            NumberOfInvestors=("InvestorID", "nunique"),
            TotalInvestment=("Investment Amount", "sum"),
            AverageROI=("ROI %", "mean"),
        )
        .sort_values("TotalInvestment", ascending=False)
    )
 
    investor_type_summary = (
        valid.groupby("Investor Type", as_index=False)
        .agg(
            TotalInvestment=("Investment Amount", "sum"),
            AverageProfit=("Profit", "mean"),
        )
        .sort_values("TotalInvestment", ascending=False)
    )
 
    return {
        "category_summary": category_summary,
        "amc_summary": amc_summary,
        "state_summary": state_summary,
        "investor_type_summary": investor_type_summary,
    }
 
 
def detect_issues(cleaned: dict) -> dict:
    nav = cleaned["nav_history"]
    trn = cleaned["transactions"]
    funds = cleaned["funds"]
    investors = cleaned["investors"]
 
    duplicate_nav_records = nav[nav.duplicated(subset=["FundID", "Date"], keep=False)]
    negative_nav = nav[nav["NAV"] < 0]
 
    today = pd.Timestamp.today().normalize()
    future_nav_dates = nav[nav["Date"] > today]
    future_purchase_dates = trn[trn["PurchaseDate"] > today]
 
    missing_fund_ids = trn[~trn["FundID"].isin(funds["FundID"])]
    missing_investor_ids = trn[~trn["InvestorID"].isin(investors["InvestorID"])]
    invalid_purchase_nav = trn[trn["PurchaseNAV"] < 0]
 
    return {
        "duplicate_nav_records": duplicate_nav_records,
        "negative_nav": negative_nav,
        "future_nav_dates": future_nav_dates,
        "future_purchase_dates": future_purchase_dates,
        "missing_fund_ids": missing_fund_ids,
        "missing_investor_ids": missing_investor_ids,
        "invalid_purchase_nav": invalid_purchase_nav,
    }
 
 
def finance_metrics(merged_metrics: pd.DataFrame, nav_history: pd.DataFrame) -> pd.DataFrame:
    out = merged_metrics.copy()
 
    out["Absolute Return"] = out["Current Value"] - out["Investment Amount"]
    out["Annual Return %"] = np.where(
        out["Investment Amount"] != 0,
        (((out["Current Value"] / out["Investment Amount"]) ** (1 / 1)) - 1) * 100,
        np.nan,
    )
 
    nav_volatility = float(np.std(nav_history["NAV"].dropna().to_numpy()))
    avg_return = float(np.nanmean(out["ROI %"] / 100))
    risk_free_rate = 0.06
 
    sharpe_ratio = np.nan
    if nav_volatility != 0:
        sharpe_ratio = (avg_return - risk_free_rate) / nav_volatility
 
    out["Volatility"] = nav_volatility
    out["Sharpe Ratio"] = sharpe_ratio
 
    return out
 
 
def export_reports(base_path: Path, analysis: dict, group_summaries: dict) -> None:
    topfunds_path = base_path / "TopFunds.xlsx"
    with pd.ExcelWriter(topfunds_path, engine="openpyxl") as writer:
        analysis["top_profitable_funds"].to_excel(writer, sheet_name="Top5ProfitableFunds", index=False)
        analysis["fund_performance"].to_excel(writer, sheet_name="FundPerformance", index=False)
        analysis["worst_fund"].to_excel(writer, sheet_name="WorstFund", index=False)
        analysis["highest_nav_fund"].to_excel(writer, sheet_name="HighestNAVFund", index=False)
        analysis["lowest_nav_fund"].to_excel(writer, sheet_name="LowestNAVFund", index=False)
 
    investor_summary_path = base_path / "InvestorSummary.xlsx"
    with pd.ExcelWriter(investor_summary_path, engine="openpyxl") as writer:
        analysis["top_investors"].to_excel(writer, sheet_name="Top5Investors", index=False)
        group_summaries["state_summary"].to_excel(writer, sheet_name="StateWiseInvestment", index=False)
        group_summaries["investor_type_summary"].to_excel(
            writer, sheet_name="InvestorTypeSummary", index=False
        )
 
    group_summaries["category_summary"].to_csv(base_path / "CategorySummary.csv", index=False)
 
 
def main() -> None:
    base_path = Path(__file__).resolve().parent
 
    print("=== Case Study 2: Mutual Fund Performance Analytics ===")
 
    data = read_data(base_path)
    cleaned, cleaning_stats = clean_data(data)
 
    merged = merge_data(cleaned)
    merged_metrics = create_columns(merged)
    merged_metrics = finance_metrics(merged_metrics, cleaned["nav_history"])
 
    np_results = numpy_tasks(cleaned["nav_history"])
    analysis = pandas_analysis(merged_metrics)
    group_summaries = groupby_analysis(merged_metrics)
    issues = detect_issues(cleaned)
 
    required_cols = [
        "Investor Name",
        "Fund Name",
        "Category",
        "AMC",
        "State",
        "Units Purchased",
        "Purchase NAV",
        "Latest NAV",
    ]
 
    print("\nPart 3 - Merged Data (required columns sample):")
    print(merged_metrics[required_cols].head(10).to_string(index=False))
 
    print("\nPart 5 - NumPy Results:")
    for k, v in np_results.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v[:10]}")
        else:
            print(f"{k}: {v:.4f}")
 
    print("\nPart 6 - Top 5 Investors by Investment Amount:")
    print(analysis["top_investors"].to_string(index=False))
 
    print("\nTop 5 Profitable Funds:")
    print(analysis["top_profitable_funds"].to_string(index=False))
 
    print("\nWorst Performing Fund (by Avg ROI):")
    print(analysis["worst_fund"].to_string(index=False))
 
    print("\nHighest NAV Fund:")
    print(analysis["highest_nav_fund"].to_string(index=False))
 
    print("\nLowest NAV Fund:")
    print(analysis["lowest_nav_fund"].to_string(index=False))
 
    print("\nPart 7 - GroupBy Summaries:")
    print("\nCategory Summary:")
    print(group_summaries["category_summary"].to_string(index=False))
    print("\nAMC Summary:")
    print(group_summaries["amc_summary"].to_string(index=False))
    print("\nState Summary:")
    print(group_summaries["state_summary"].to_string(index=False))
    print("\nInvestor Type Summary:")
    print(group_summaries["investor_type_summary"].to_string(index=False))
 
    print("\nPart 8 - Detected Issues:")
    for name, issue_df in issues.items():
        print(f"{name}: {len(issue_df)}")
 
    print("\nPart 2 - Cleaning Stats:")
    print(cleaning_stats)
 
    export_reports(base_path, analysis, group_summaries)
 
    print("\nPart 10 - Reports exported:")
    print("TopFunds.xlsx")
    print("InvestorSummary.xlsx")
    print("CategorySummary.csv")
 
 
if __name__ == "__main__":
    main()
 
 