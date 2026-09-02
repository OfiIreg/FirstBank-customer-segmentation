"""
ingest.py
Stage 1 of the pipeline: Ingestion.

Reads the raw customer and transaction extracts (standing in for a core-banking
export, per the Module 1 Vision Document's data source description) and copies
them, unmodified, into the interim zone. Every read is written to the privacy
audit log required in Section 6 of the Module 3 brief.
"""
import pandas as pd
from pathlib import Path
from audit_log import audit_event

RAW_DIR = Path("/home/claude/m3/repo/data/raw")
INTERIM_DIR = Path("/home/claude/m3/repo/data/interim")

def ingest():
    customers = pd.read_csv(RAW_DIR / "customers_raw.csv")
    audit_event("ingest", "customers_raw.csv", rows=len(customers), note="read from raw zone")

    txns = pd.read_csv(RAW_DIR / "transactions_raw.csv")
    audit_event("ingest", "transactions_raw.csv", rows=len(txns), note="read from raw zone")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(INTERIM_DIR / "customers_ingested.csv", index=False)
    txns.to_csv(INTERIM_DIR / "transactions_ingested.csv", index=False)

    audit_event("ingest", "customers_ingested.csv", rows=len(customers), note="written to interim zone")
    audit_event("ingest", "transactions_ingested.csv", rows=len(txns), note="written to interim zone")
    return len(customers), len(txns)

if __name__ == "__main__":
    n_cust, n_txn = ingest()
    print(f"Ingested {n_cust} customer records and {n_txn} transaction records.")
