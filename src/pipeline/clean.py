"""
clean.py
Stage 2 of the pipeline: Cleaning.

Addresses the data quality problems identified in the Module 2 Data Dictionary
and Module 3 root-cause review: missing values, duplicate records, invalid
categorical labels, and implausible outliers. Every correction is counted and
written to the privacy audit log so the transformation is reproducible and
traceable, per the Module 3 rubric's reproducibility requirement.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from audit_log import audit_event

INTERIM_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "interim"

VALID_CATEGORIES = {
    "POS Purchase", "ATM Withdrawal", "Airtime/Data", "Bill Payment",
    "Transfer", "Salary Credit", "Loan Repayment", "Savings Deposit",
}
CATEGORY_FIX_MAP = {
    "pos purchase": "POS Purchase",
    "atm-withdrawal": "ATM Withdrawal",
    "transfer": "Transfer",
    "n/a": "Other/Unclassified",
}

def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop exact duplicate customer records (simulated double registration)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    n_dupes = before - len(df)

    # Invalid ages (negative, or above plausible human lifespan) -> null, then impute with median
    invalid_age_mask = (df["age"] < 18) | (df["age"] > 100)
    n_invalid_age = int(invalid_age_mask.sum())
    df.loc[invalid_age_mask, "age"] = np.nan

    n_missing_age = int(df["age"].isna().sum())
    df["age"] = df["age"].fillna(df["age"].median())

    n_missing_gender = int(df["gender"].isna().sum())
    df["gender"] = df["gender"].fillna("Unspecified")

    n_missing_location = int(df["location"].isna().sum())
    df["location"] = df["location"].fillna("Unknown")

    n_missing_balance = int(df["account_balance"].isna().sum())
    df["account_balance"] = df["account_balance"].fillna(df["account_balance"].median())

    df["age"] = df["age"].astype(int)

    audit_event("clean", "customers", rows=len(df), note=(
        f"dropped {n_dupes} duplicate customer records; fixed {n_invalid_age} invalid ages; "
        f"imputed {n_missing_age} missing ages (median), {n_missing_gender} missing genders, "
        f"{n_missing_location} missing locations, {n_missing_balance} missing balances"
    ))
    return df

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.drop_duplicates(keep="first")
    n_dupes = before - len(df)

    # Standardize inconsistent category labels (case-insensitive match)
    df["txn_category"] = df["txn_category"].astype(str).str.strip()
    lower_series = df["txn_category"].str.lower()
    mapped = lower_series.map(CATEGORY_FIX_MAP)
    would_change = mapped.notna() & (mapped != df["txn_category"])
    n_dirty = int(would_change.sum())
    df["txn_category"] = mapped.fillna(df["txn_category"])
    df.loc[~df["txn_category"].isin(VALID_CATEGORIES), "txn_category"] = "Other/Unclassified"

    # Missing transaction amounts -> impute with category median
    n_missing_amount = int(df["txn_amount"].isna().sum())
    df["txn_amount"] = df.groupby("txn_category")["txn_amount"].transform(lambda s: s.fillna(s.median()))
    df["txn_amount"] = df["txn_amount"].fillna(df["txn_amount"].median())

    # Cap outliers at the 99.5th percentile (winsorize) rather than dropping, to preserve volume
    cap = df["txn_amount"].quantile(0.995)
    n_capped = int((df["txn_amount"] > cap).sum())
    df["txn_amount"] = df["txn_amount"].clip(upper=cap)

    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    n_bad_dates = int(df["txn_date"].isna().sum())
    df = df.dropna(subset=["txn_date"])

    audit_event("clean", "transactions", rows=len(df), note=(
        f"dropped {n_dupes} duplicate transactions; standardized {n_dirty} dirty category labels; "
        f"imputed {n_missing_amount} missing amounts; capped {n_capped} outliers at p99.5; "
        f"dropped {n_bad_dates} unparsable dates"
    ))
    return df

def run():
    customers = pd.read_csv(INTERIM_DIR / "customers_ingested.csv")
    txns = pd.read_csv(INTERIM_DIR / "transactions_ingested.csv")

    customers_clean = clean_customers(customers)
    txns_clean = clean_transactions(txns)

    customers_clean.to_csv(INTERIM_DIR / "customers_clean.csv", index=False)
    txns_clean.to_csv(INTERIM_DIR / "transactions_clean.csv", index=False)
    return customers_clean, txns_clean

if __name__ == "__main__":
    c, t = run()
    print(f"Cleaned customers: {len(c)} rows")
    print(f"Cleaned transactions: {len(t)} rows")
