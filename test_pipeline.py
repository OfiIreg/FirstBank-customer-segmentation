"""
test_pipeline.py
Unit Tests (Module 3 technical requirement) for the core pipeline functions.
Run with: pytest tests/test_pipeline.py -v
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipeline"))

from clean import clean_customers, clean_transactions
from transform import transform_transactions
from anonymize import age_band, balance_band, pseudonymize_id


# ---------- clean_customers ----------

def test_clean_customers_drops_duplicates():
    df = pd.DataFrame({
        "customer_id": ["A1", "A1", "A2"],
        "age": [30, 30, 40],
        "gender": ["Male", "Male", "Female"],
        "location": ["Lagos", "Lagos", "Kano"],
        "account_balance": [1000.0, 1000.0, 2000.0],
    })
    out = clean_customers(df)
    assert len(out) == 2
    assert out["customer_id"].is_unique

def test_clean_customers_fixes_invalid_age():
    df = pd.DataFrame({
        "customer_id": ["A1", "A2", "A3"],
        "age": [-1, 200, 35],
        "gender": ["Male", "Female", "Male"],
        "location": ["Lagos", "Kano", "Enugu"],
        "account_balance": [1000.0, 2000.0, 3000.0],
    })
    out = clean_customers(df)
    assert out["age"].between(18, 100).all()

def test_clean_customers_imputes_missing_values():
    df = pd.DataFrame({
        "customer_id": ["A1", "A2"],
        "age": [30, np.nan],
        "gender": ["Male", np.nan],
        "location": ["Lagos", np.nan],
        "account_balance": [1000.0, np.nan],
    })
    out = clean_customers(df)
    assert out["age"].isna().sum() == 0
    assert out["gender"].isna().sum() == 0
    assert out["location"].isna().sum() == 0
    assert out["account_balance"].isna().sum() == 0
    assert (out["gender"] == "Unspecified").any()


# ---------- clean_transactions ----------

def test_clean_transactions_standardizes_categories():
    df = pd.DataFrame({
        "customer_id": ["A1", "A1", "A1"],
        "txn_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "txn_amount": [100.0, 200.0, 150.0],
        "txn_category": ["pos purchase", "ATM-Withdrawal", "TRANSFER"],
    })
    out = clean_transactions(df)
    assert set(out["txn_category"]) <= {"POS Purchase", "ATM Withdrawal", "Transfer"}

def test_clean_transactions_caps_outliers():
    amounts = [100.0] * 200 + [50_000_000.0]  # one extreme outlier
    df = pd.DataFrame({
        "customer_id": ["A1"] * 201,
        "txn_date": ["2024-01-01"] * 201,
        "txn_amount": amounts,
        "txn_category": ["Transfer"] * 201,
    })
    out = clean_transactions(df)
    assert out["txn_amount"].max() < 50_000_000.0

def test_clean_transactions_drops_unparsable_dates():
    df = pd.DataFrame({
        "customer_id": ["A1", "A2"],
        "txn_date": ["2024-01-01", "not-a-date"],
        "txn_amount": [100.0, 200.0],
        "txn_category": ["Transfer", "Transfer"],
    })
    out = clean_transactions(df)
    assert len(out) == 1


# ---------- transform_transactions ----------

def test_transform_produces_one_row_per_customer():
    df = pd.DataFrame({
        "customer_id": ["A1", "A1", "A2"],
        "txn_date": ["2024-01-01", "2024-01-05", "2024-01-03"],
        "txn_amount": [100.0, 200.0, 300.0],
        "txn_category": ["Transfer", "POS Purchase", "Transfer"],
    })
    out = transform_transactions(df)
    assert len(out) == 2
    assert set(out.columns) >= {"customer_id", "recency_days", "frequency", "monetary_total", "monetary_avg"}

def test_transform_frequency_is_correct():
    df = pd.DataFrame({
        "customer_id": ["A1", "A1", "A1", "A2"],
        "txn_date": ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-03"],
        "txn_amount": [100.0, 200.0, 300.0, 300.0],
        "txn_category": ["Transfer", "POS Purchase", "Transfer", "Transfer"],
    })
    out = transform_transactions(df).set_index("customer_id")
    assert out.loc["A1", "frequency"] == 3
    assert out.loc["A2", "frequency"] == 1


# ---------- anonymize helpers ----------

def test_age_band_boundaries():
    assert age_band(18) == "18-24"
    assert age_band(24) == "18-24"
    assert age_band(25) == "25-34"
    assert age_band(60) == "60+"
    assert age_band(99) == "60+"

def test_balance_band_boundaries():
    assert balance_band(1000) == "Under 50K"
    assert balance_band(100_000) == "50K-500K"
    assert balance_band(1_000_000) == "500K-2M"
    assert balance_band(5_000_000) == "2M+"

def test_pseudonymize_id_is_deterministic_and_irreversible():
    a = pseudonymize_id("FB100001")
    b = pseudonymize_id("FB100001")
    c = pseudonymize_id("FB100002")
    assert a == b               # same input -> same hash (needed for consistent joins)
    assert a != c                # different input -> different hash
    assert "FB100001" not in a   # original id is not recoverable from the output
    assert a.startswith("ANON_")
