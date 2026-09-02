"""
prepare_features.py
Data Readiness & Feature Set (Module 4, Section 2).

Builds the model-ready feature matrix from the Module 3 clean dataset plus
the target defined in make_target.py. Documents every preparation step so
the process is auditable, per the reproducibility standard set in Module 3.

Leakage prevention:
- customer_id is dropped (identifier, not a feature).
- All features are derived from data available BEFORE the campaign decision
  point (transaction history up to the snapshot date used in Module 3's
  transform.py); none are derived from the outcome itself.
- Scaling/encoding statistics (scaler mean/std, one-hot categories) are
  fit on the TRAIN split only and applied to test, to avoid test-set
  information leaking into preprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = Path("/home/claude/m4/repo/data/processed/customer_model_dataset.csv")
OUT_DIR = Path("/home/claude/m4/repo/data/processed")

NUMERIC_FEATURES = [
    "age", "account_balance", "recency_days", "frequency",
    "monetary_total", "monetary_avg",
    "share_ATM Withdrawal", "share_Airtime/Data", "share_Bill Payment",
    "share_Loan Repayment", "share_Other/Unclassified", "share_POS Purchase",
    "share_Salary Credit", "share_Savings Deposit", "share_Transfer",
]
CATEGORICAL_FEATURES = ["location"]  # gender is EXCLUDED from model features (see note below)
PROTECTED_ATTRIBUTES = ["gender", "age"]  # retained separately for fairness analysis only
TARGET = "cross_sell_conversion"


def load_and_split(test_size=0.2, seed=42):
    df = pd.read_csv(DATA_PATH)

    # gender is a protected attribute (Module 1/2 Fairness Objectives) and is
    # deliberately excluded from the model's input features so the model
    # cannot use it directly; it is retained in a separate frame for the
    # fairness audit in Section 6, since fairness must be measured even when
    # a protected attribute isn't a model input (proxy bias via correlated
    # features, e.g. age, is exactly what that audit checks for).
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    protected = df[PROTECTED_ATTRIBUTES].copy()
    customer_ids = df["customer_id"].copy()

    X_train, X_test, y_train, y_test, prot_train, prot_test, id_train, id_test = train_test_split(
        X, y, protected, customer_ids,
        test_size=test_size, stratify=y, random_state=seed,  # stratified hold-out (Section 3)
    )
    return X_train, X_test, y_train, y_test, prot_train, prot_test, id_train, id_test


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def fitted_feature_names(preprocessor):
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, prot_train, prot_test, id_train, id_test = load_and_split()

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)  # fit on TRAIN only -> no leakage into test

    feature_names = fitted_feature_names(preprocessor)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, OUT_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, OUT_DIR / "feature_names.joblib")

    for name, obj in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test),
                       ("prot_train", prot_train), ("prot_test", prot_test),
                       ("id_train", id_train), ("id_test", id_test)]:
        obj.to_csv(OUT_DIR / f"{name}.csv", index=False)

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Features after encoding: {len(feature_names)}")
    print(f"Train conversion rate: {y_train.mean():.1%} | Test conversion rate: {y_test.mean():.1%}")
    print(f"Feature names: {feature_names}")
