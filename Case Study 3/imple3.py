import os
import pandas as pd
import numpy as np
 
# ==========================================================
# READ CSV FILES
# ==========================================================
 
# Resolve CSV paths relative to this script so it runs from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
customers = pd.read_csv(os.path.join(BASE_DIR, "customers.csv"))
loan_applications = pd.read_csv(os.path.join(BASE_DIR, "loan_application.csv"))
loan_payments = pd.read_csv(os.path.join(BASE_DIR, "loan_payments.csv"))
 
# Normalize keys so all datasets can be merged reliably.
customers["CustomerIDKey"] = pd.to_numeric(
    customers["CustomerID"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)
 
loan_applications["CustomerIDKey"] = pd.to_numeric(
    loan_applications["CustomerID"],
    errors="coerce"
)
 
loan_applications["LoanIDKey"] = "L" + (
    pd.to_numeric(
        loan_applications["LoanID"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    ) - 900
).astype("Int64").astype(str)
 
loan_payments["LoanIDKey"] = loan_payments["LoanID"].astype(str)
 
# ==========================================================
# DATA CLEANING
# ==========================================================
 
# Remove duplicate records
customers.drop_duplicates(inplace=True)
loan_applications.drop_duplicates(inplace=True)
loan_payments.drop_duplicates(inplace=True)
 
# Remove duplicate Loan IDs
loan_applications.drop_duplicates(subset="LoanID", inplace=True)
loan_payments.drop_duplicates(subset="LoanID", inplace=True)
 
# Missing Values
customers["Salary"] = customers["Salary"].fillna(customers["Salary"].median())
if "CreditScore" not in customers.columns:
    customers["CreditScore"] = 700.0
else:
    customers["CreditScore"] = customers["CreditScore"].fillna(customers["CreditScore"].mean())
 
# Convert Dates
loan_applications["ApplicationDate"] = pd.to_datetime(
    loan_applications["ApplicationDate"], errors="coerce"
)
 
loan_payments["LastPaymentDate"] = pd.to_datetime(
    loan_payments["LastPaymentDate"], errors="coerce"
)
 
# Remove negative Loan Amount
loan_applications = loan_applications[
    loan_applications["LoanAmount"] >= 0
]
 
# Remove invalid EMI Amount
loan_payments = loan_payments[
    loan_payments["EMIAmount"] > 0
]
 
# Remove future payment dates
today = pd.Timestamp.today()
 
loan_payments = loan_payments[
    loan_payments["LastPaymentDate"] <= today
]
 
# Build fields expected by downstream analysis using available payment data.
loan_payments["AmountPaid"] = loan_payments["EMIAmount"] * loan_payments["PaidEMIs"]
loan_payments["PaymentStatus"] = np.where(
    loan_payments["PendingEMIs"] > 0,
    "Pending",
    "Paid"
)
 
# ==========================================================
# MERGE DATASETS
# ==========================================================
 
df = customers.merge(
    loan_applications,
    on="CustomerIDKey",
    how="inner",
    suffixes=("_cust", "_app")
)
 
df = df.merge(
    loan_payments,
    on="LoanIDKey",
    how="inner",
    suffixes=("", "_pay")
)
 
# ==========================================================
# NEW COLUMNS
# ==========================================================
 
df["MonthlyIncome"] = df["Salary"] / 12
 
df["DebtToIncome"] = np.where(
    df["Salary"] > 0,
    df["LoanAmount"] / df["Salary"],
    np.nan
)
 
df["EMIDue"] = df["EMIAmount"] * df["PendingEMIs"]
 
df["PaymentCompletion"] = (
    df["PaidEMIs"] / (df["PaidEMIs"] + df["PendingEMIs"])
) * 100
 
# ==========================================================
# NUMPY TASKS
# ==========================================================
 
loan = df["LoanAmount"].values
 
print("Average Loan:", np.mean(loan))
print("Median Loan:", np.median(loan))
print("Maximum Loan:", np.max(loan))
print("Minimum Loan:", np.min(loan))
print("Standard Deviation:", np.std(loan))
print("Variance:", np.var(loan))
print("25th Percentile:", np.percentile(loan, 25))
print("75th Percentile:", np.percentile(loan, 75))
 
# ==========================================================
# PANDAS ANALYSIS
# ==========================================================
 
top10_loan = df.nlargest(10, "LoanAmount")
 
top10_salary = df.nlargest(10, "Salary")
 
low_credit = df[df["CreditScore"] < 650]
 
loan_20lakh = df[df["LoanAmount"] > 2000000]
 
pending_payment = df[df["PaymentStatus"] == "Pending"]
 
fully_paid = df[df["PaymentStatus"] == "Paid"]
 
# ==========================================================
# GROUP BY CITY
# ==========================================================
 
city_summary = df.groupby("City").agg(
    Customers=("CustomerID_cust", "count"),
    AverageSalary=("Salary", "mean"),
    TotalLoan=("LoanAmount", "sum")
)
 
# ==========================================================
# GROUP BY LOAN TYPE
# ==========================================================
 
loan_type_summary = df.groupby("LoanType").agg(
    NumberOfLoans=("LoanID", "count"),
    AverageLoan=("LoanAmount", "mean"),
    TotalLoan=("LoanAmount", "sum")
)
 
# ==========================================================
# GROUP BY LOAN STATUS
# ==========================================================
 
loan_status = df.groupby("LoanStatus").size()
 
# ==========================================================
# GROUP BY PAYMENT STATUS
# ==========================================================
 
payment_status = df.groupby("PaymentStatus").agg(
    Count=("LoanID", "count"),
    TotalPaid=("AmountPaid", "sum")
)
 
# ==========================================================
# BUSINESS RULES
# ==========================================================
 
flagged = df[
    (df["LoanAmount"] > 3000000) |
    (df["CreditScore"] < 650) |
    (df["Salary"] < 30000) |
    (df["DebtToIncome"] > 5) |
    (df["EMIDue"] > 10000) |
    (df["PaymentStatus"] == "Pending") |
    (df["LoanStatus"] == "Rejected")
]
 
# ==========================================================
# FINANCE METRICS
# ==========================================================
 
total_portfolio = df["LoanAmount"].sum()
 
amount_collected = df["AmountPaid"].sum()
 
df["Outstanding"] = (df["LoanAmount"] - df["AmountPaid"]).clip(lower=0)
 
loan_recovery = (
    amount_collected / total_portfolio
) * 100
 
default_percent = (
    len(df[df["PaymentStatus"] == "Pending"])
    / len(df)
) * 100
 
average_emi = df["EMIAmount"].mean()
 
average_credit = df["CreditScore"].mean()
 
print("\nFinance Metrics")
print("Total Portfolio:", total_portfolio)
print("Amount Collected:", amount_collected)
print("Loan Recovery %:", loan_recovery)
print("Default %:", default_percent)
print("Average EMI:", average_emi)
print("Average Credit Score:", average_credit)
 
# ==========================================================
# EXPORT REPORTS
# ==========================================================
 
with pd.ExcelWriter("LoanSummary.xlsx") as writer:
    city_summary.to_excel(writer, sheet_name="City Summary")
    loan_type_summary.to_excel(writer, sheet_name="Loan Type")
    loan_status.to_excel(writer, sheet_name="Loan Status")
 
with pd.ExcelWriter("CustomerLoanReport.xlsx") as writer:
    df.to_excel(writer, index=False)
 
pending_payment.to_csv(os.path.join(BASE_DIR, "PendingPayments.csv"), index=False)
 
# ==========================================================
# DISPLAY OUTPUTS
# ==========================================================
 
print("\nTop 10 Loan Customers")
print(top10_loan)
 
print("\nCustomers with Low Credit Score")
print(low_credit)
 
print("\nPending Loan Payments")
print(pending_payment)
 
print("\nCity-wise Loan Summary")
print(city_summary)
 
print("\nLoan Type Summary")
print(loan_type_summary)
 
print("\nLoan Recovery Report")
print({
    "Total Portfolio": total_portfolio,
    "Collected": amount_collected,
    "Recovery %": loan_recovery,
    "Default %": default_percent
})
 