"""
transform.py
Stage 3 of the pipeline: Transformation.

Converts row-level transactions into customer-level behavioral features
(recency, frequency, monetary aggregates, and category mix), the same
feature shape the Module 1 solution overview and Module 2 architecture
described feeding into the clustering and propensity models in Module 4.
"""
import pandas as pd
from pathlib import Path
from audit_log import audit_event

INTERIM_DIR = Path("/home/claude/m3/repo/data/interim")

def transform_transactions(txns: pd.DataFrame) -> pd.DataFrame:
    txns["txn_date"] = pd.to_datetime(txns["txn_date"])
    snapshot_date = txns["txn_date"].max() + pd.Timedelta(days=1)

    agg = txns.groupby("customer_id").agg(
        recency_days=("txn_date", lambda s: (snapshot_date - s.max()).days),
        frequency=("txn_date", "count"),
        monetary_total=("txn_amount", "sum"),
        monetary_avg=("txn_amount", "mean"),
    ).reset_index()

    # Share of spend by category, one column per category (simple one-hot style feature)
    cat_share = (
        txns.groupby(["customer_id", "txn_category"])["txn_amount"].sum().unstack(fill_value=0)
    )
    cat_share = cat_share.div(cat_share.sum(axis=1), axis=0).add_prefix("share_")
    cat_share = cat_share.reset_index()

    features = agg.merge(cat_share, on="customer_id", how="left")
    audit_event("transform", "transactions", rows=len(features), note=(
        "aggregated to customer level: recency, frequency, monetary_total, monetary_avg, "
        f"and spend share across {cat_share.shape[1] - 1} transaction categories"
    ))
    return features

def run():
    txns_clean = pd.read_csv(INTERIM_DIR / "transactions_clean.csv")
    features = transform_transactions(txns_clean)
    features.to_csv(INTERIM_DIR / "customer_txn_features.csv", index=False)
    return features

if __name__ == "__main__":
    f = run()
    print(f"Transformed to {len(f)} customer-level feature rows, {f.shape[1]} columns")
