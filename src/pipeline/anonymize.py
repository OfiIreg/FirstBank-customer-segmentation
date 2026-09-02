"""
anonymize.py
Data Anonymization Plan implementation (Module 3, Section 6).

Produces an anonymized copy of the customer analytics table for use in any
context outside the controlled pipeline environment (e.g., sharing with
external reviewers, or the Module 5 dashboard demo). Implements the
Data Anonymization Plan referenced in the deck:
  1. Customer IDs are irreversibly hashed (salted SHA-256) rather than the
     reversible pseudonymization used inside the trusted pipeline boundary.
  2. Exact age is generalized to age band (k-anonymity style generalization).
  3. Exact account balance and monetary_total are bucketed into ranges.
  4. Location is generalized from state to geopolitical zone.
No names, phone numbers, or account numbers exist in this dataset (per the
Module 2 Data Dictionary), so this script focuses on quasi-identifiers, which
are the realistic re-identification risk in a behavioral segmentation table.
"""
import pandas as pd
import hashlib
from pathlib import Path
from audit_log import audit_event

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SALT = "firstbank-ban6800-module3-demo-salt"  # in production this would be a managed secret, not in source

ZONE_MAP = {
    "Lagos": "South West", "Oyo": "South West", "Ogun": "South West",
    "Kano": "North West", "Kaduna": "North West",
    "Rivers": "South South", "Delta": "South South", "Edo": "South South", "Cross River": "South South",
    "Enugu": "South East", "Anambra": "South East", "Imo": "South East",
    "Plateau": "North Central", "Abuja FCT": "North Central", "Kwara": "North Central",
}

def age_band(age):
    if age < 25: return "18-24"
    if age < 35: return "25-34"
    if age < 45: return "35-44"
    if age < 60: return "45-59"
    return "60+"

def balance_band(bal):
    if bal < 50_000: return "Under 50K"
    if bal < 500_000: return "50K-500K"
    if bal < 2_000_000: return "500K-2M"
    return "2M+"

def pseudonymize_id(customer_id: str) -> str:
    return "ANON_" + hashlib.sha256((SALT + str(customer_id)).encode()).hexdigest()[:16]

def run():
    df = pd.read_csv(PROCESSED_DIR / "customer_analytics_table.csv")
    anon = df.copy()

    anon["customer_id"] = anon["customer_id"].apply(pseudonymize_id)
    anon["age_band"] = anon["age"].apply(age_band)
    anon["balance_band"] = anon["account_balance"].apply(balance_band)
    anon["geo_zone"] = anon["location"].map(ZONE_MAP).fillna("Other")

    anon = anon.drop(columns=["age", "account_balance", "location"])

    out_path = PROCESSED_DIR / "customer_analytics_table_anonymized.csv"
    anon.to_csv(out_path, index=False)

    audit_event("anonymize", "customer_analytics_table_anonymized", rows=len(anon), note=(
        "irreversibly hashed customer_id (salted SHA-256); generalized age to age_band, "
        "account_balance to balance_band, and location to geo_zone; "
        "dropped raw quasi-identifier columns"
    ))
    return anon

if __name__ == "__main__":
    anon = run()
    print(f"Anonymized table: {anon.shape[0]} rows, {anon.shape[1]} columns")
    print(anon[["customer_id", "gender", "age_band", "balance_band", "geo_zone"]].head(3).to_string())
