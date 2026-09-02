"""
serialize_model.py
Model Serialization (Module 4 technical requirement): bundles the champion
model together with its preprocessor and feature names into a single
joblib artifact the FastAPI app can load directly.
"""
import joblib
from pathlib import Path

DATA_DIR = Path("/home/claude/m4/repo/data/processed")
ASSET_DIR = Path("/home/claude/m4/assets")
OUT_DIR = Path("/home/claude/m4/repo/src/api")

if __name__ == "__main__":
    model = joblib.load(ASSET_DIR / "logreg_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")
    feature_names = joblib.load(DATA_DIR / "feature_names.joblib")

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_features": [
            "age", "account_balance", "recency_days", "frequency",
            "monetary_total", "monetary_avg",
            "share_ATM Withdrawal", "share_Airtime/Data", "share_Bill Payment",
            "share_Loan Repayment", "share_Other/Unclassified", "share_POS Purchase",
            "share_Salary Credit", "share_Savings Deposit", "share_Transfer",
        ],
        "categorical_features": ["location"],
        "model_version": "1",
        "model_name": "firstbank-cross-sell-propensity-model",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_DIR / "model_bundle.joblib")
    print(f"Saved model bundle to {OUT_DIR / 'model_bundle.joblib'}")
