"""
integrate.py
Stage 4 of the pipeline: Integration.

Joins the cleaned customer demographic table with the transaction-derived
behavioral features into a single customer-level analytics table.

Why this stage exists, explicitly tied to the project's analytics goals:
FirstBank's Module 1 problem is that relationship managers segment and
target customers using demographic proxies alone (age, branch location)
because no single table links who a customer is to how they actually
behave. Cleaning and Transformation (Stages 2-3) fix each side of that gap
separately: cleaning makes the demographic and transaction records
trustworthy, transformation turns raw transactions into behavioral
signals (recency, frequency, monetary value). Neither one, alone, is
usable for the Module 1 goal of behavioral segmentation - a demographic
table with no behavior is what caused the problem in the first place, and
a behavioral table with no demographics cannot be joined back to a
customer record for a relationship manager to act on. Integration is the
step that removes that gap: it produces the single customer-level table
that both the Module 4 propensity model and the Module 1/2 Power BI
segment-explorer dashboard train on and read from, so every downstream
analytics goal in this project depends on this join existing and being
correct, not just clean inputs existing separately.
"""
import pandas as pd
from pathlib import Path
from audit_log import audit_event

INTERIM_DIR = Path("/home/claude/m3/repo/data/interim")
PROCESSED_DIR = Path("/home/claude/m3/repo/data/processed")

def run():
    customers = pd.read_csv(INTERIM_DIR / "customers_clean.csv")
    features = pd.read_csv(INTERIM_DIR / "customer_txn_features.csv")


    final = customers.merge(features, on="customer_id", how="left")

    # Customers with no transactions in the window get zero-activity features, not nulls
    fill_cols = [c for c in final.columns if c not in customers.columns]
    final[fill_cols] = final[fill_cols].fillna(0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "customer_analytics_table.csv"
    final.to_csv(out_path, index=False)

    audit_event("integrate", "customer_analytics_table", rows=len(final), note=(
        f"joined {len(customers)} customer records with behavioral features into "
        f"final analytics table with {final.shape[1]} columns, written to data/processed/"
    ))
    return final

if __name__ == "__main__":
    final = run()
    print(f"Final integrated analytics table: {final.shape[0]} rows, {final.shape[1]} columns")
    print(final.head(3).to_string())
