import pandas as pd
import numpy as np
import json
 
# ====================================================
# LOAN CLASS (OOP)
# ====================================================
 
class Loan:
    def __init__(self, customer_id, loan_amount, interest_rate, tenure):
        self.customer_id = customer_id
        self.loan_amount = loan_amount
        self.interest_rate = interest_rate
        self.tenure = tenure
 
    def calculate_emi(self):
        r = self.interest_rate / (12 * 100)
        n = self.tenure
 
        if r == 0:
            return self.loan_amount / n
 
        emi = (self.loan_amount * r * (1 + r) ** n) / (((1 + r) ** n) - 1)
        return emi
 

 
def read_csv(file_name):
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"{file_name} not found.")
        return None
    except pd.errors.ParserError:
        print(f"{file_name} is corrupted.")
        return None
    except Exception as e:
        print(e)
        return None
 
 
customers = read_csv("customers.csv")
loans = read_csv("loans.csv")
credit = read_csv("credit_scores.csv")
 
if customers is None or loans is None or credit is None:
    exit()
 

 
df = customers.merge(loans, on="CustomerID")
df = df.merge(credit, on="CustomerID")
 
 
# ====================================================
# HANDLE MISSING VALUES
# ====================================================
 
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
 
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].mean())
 
df["InterestRate"] = df["InterestRate"].fillna(df["InterestRate"].median())
 
 
# ====================================================
# REMOVE OUTLIERS
# ====================================================
 
limit = np.percentile(df["LoanAmount"], 99)
 
df = df[df["LoanAmount"] <= limit]
 
 
# ====================================================
# NUMPY CALCULATIONS
# ====================================================
 
mean_loan = np.mean(df["LoanAmount"])
 
median_salary = np.median(df["Salary"])
 
percentile_interest = np.percentile(df["InterestRate"], 90)
 
correlation = np.corrcoef(df["Salary"], df["LoanAmount"])[0][1]
 
std_dev = np.std(df["LoanAmount"])
 
 
# ====================================================
# FINANCE METRICS
# ====================================================
 
# Debt to Income Ratio
 
df["DebtToIncome"] = df["LoanAmount"] / df["Salary"]
 
# Loan Utilization
 
if "LoanLimit" in df.columns:
    df["LoanUtilization"] = df["LoanAmount"] / df["LoanLimit"]
else:
    df["LoanUtilization"] = 0
 
# EMI
 
emis = []
 
if "TenureMonths" in df.columns:
    tenure_col = "TenureMonths"
elif "Tenure" in df.columns:
    tenure_col = "Tenure"
else:
    raise KeyError("Missing tenure column: expected 'TenureMonths' or 'Tenure'.")
 
for _, row in df.iterrows():
    loan = Loan(
        row["CustomerID"],
        row["LoanAmount"],
        row["InterestRate"],
        row[tenure_col]
    )
    emis.append(loan.calculate_emi())
 
df["EMI"] = emis
 
average_emi = df["EMI"].mean()
 
default_percent = (df["DefaultFlag"].sum() / len(df)) * 100
 
if "LoanStatus" in df.columns:
    npa_percent = (df["LoanStatus"] == "NPA").sum() / len(df) * 100
else:
    npa_percent = default_percent
 
LGD = 0.40
 
df["ExpectedLoss"] = (
    (1 - df["CreditScore"] / 850)
    * LGD
    * df["LoanAmount"]
)
 
expected_loss = df["ExpectedLoss"].sum()
 
 
# ====================================================
# HIGH RISK CUSTOMERS
# ====================================================
 
high_risk = df[
    (df["CreditScore"] < 650) &
    (df["Salary"] < 60000) &
    (df["LoanAmount"] > 1000000) &
    (df["DefaultFlag"] == 1)
]
 
top20 = high_risk.sort_values(
    by="LoanAmount",
    ascending=False
).head(20)
 
 
# ====================================================
# SAVE OUTPUT FILES
# ====================================================
 
top20.to_csv("high_risk_customers.csv", index=False)
 
summary = {
    "Mean Loan Amount": float(mean_loan),
    "Median Salary": float(median_salary),
    "90 Percentile Interest Rate": float(percentile_interest),
    "Correlation": float(correlation),
    "Standard Deviation": float(std_dev),
    "Default Percentage": float(default_percent),
    "NPA Percentage": float(npa_percent),
    "Average EMI": float(average_emi),
    "Expected Loss": float(expected_loss)
}
 
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=4)
 
with pd.ExcelWriter("risk_report.xlsx") as writer:
    df.to_excel(writer, sheet_name="Merged Data", index=False)
    top20.to_excel(writer, sheet_name="High Risk", index=False)
 
print("Risk Report Generated Successfully!")