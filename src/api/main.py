"""
main.py
FastAPI application (Module 4 technical requirement) serving the FirstBank
cross-sell propensity model. Loads the serialized model bundle produced by
serialize_model.py and exposes a /predict endpoint.

Run locally with:
    uvicorn main:app --reload --port 8000

Then POST to http://localhost:8000/predict with a JSON body matching the
CustomerFeatures schema below, or see /docs for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path
from typing import Literal
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "monitoring"))
from explanation_store import log_explanation

BUNDLE_PATH = Path(__file__).parent / "model_bundle.joblib"
BACKGROUND_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "X_train.csv"

app = FastAPI(
    title="FirstBank Cross-Sell Propensity API",
    description="Predicts the probability a FirstBank retail customer converts "
                "on a targeted cross-sell offer. BAN6800 Module 4, extended in the "
                "Final Project with an audit-ready explanation store.",
    version="1.1",
)

bundle = None
background_df = None


@app.on_event("startup")
def load_model():
    global bundle, background_df
    bundle = joblib.load(BUNDLE_PATH)
    if BACKGROUND_PATH.exists():
        feat_cols = bundle["numeric_features"] + bundle["categorical_features"]
        raw = pd.read_csv(BACKGROUND_PATH)[feat_cols].sample(min(100, len(pd.read_csv(BACKGROUND_PATH))), random_state=42)
        transformed = bundle["preprocessor"].transform(raw)
        background_df = pd.DataFrame(transformed, columns=bundle["preprocessor"].get_feature_names_out())


LOCATIONS = Literal[
    "Lagos", "Kano", "Rivers", "Oyo", "Kaduna", "Ogun", "Enugu", "Delta",
    "Anambra", "Edo", "Plateau", "Abuja FCT", "Cross River", "Imo", "Kwara", "Unknown",
]


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, example=38)
    account_balance: float = Field(..., ge=0, example=250000.0)
    recency_days: int = Field(..., ge=0, example=14)
    frequency: int = Field(..., ge=0, example=8)
    monetary_total: float = Field(..., ge=0, example=180000.0)
    monetary_avg: float = Field(..., ge=0, example=22500.0)
    share_ATM_Withdrawal: float = Field(0.0, ge=0, le=1, alias="share_ATM Withdrawal")
    share_Airtime_Data: float = Field(0.0, ge=0, le=1, alias="share_Airtime/Data")
    share_Bill_Payment: float = Field(0.0, ge=0, le=1, alias="share_Bill Payment")
    share_Loan_Repayment: float = Field(0.0, ge=0, le=1, alias="share_Loan Repayment")
    share_Other_Unclassified: float = Field(0.0, ge=0, le=1, alias="share_Other/Unclassified")
    share_POS_Purchase: float = Field(0.0, ge=0, le=1, alias="share_POS Purchase")
    share_Salary_Credit: float = Field(0.0, ge=0, le=1, alias="share_Salary Credit")
    share_Savings_Deposit: float = Field(0.0, ge=0, le=1, alias="share_Savings Deposit")
    share_Transfer: float = Field(0.0, ge=0, le=1, alias="share_Transfer")
    location: LOCATIONS = "Lagos"

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_conversion_probability: float
    predicted_label: int
    model_version: str
    model_name: str


@app.get("/")
def root():
    return {"status": "ok", "message": "FirstBank Cross-Sell Propensity API. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bundle is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = features.dict(by_alias=True)
    X = pd.DataFrame([row])[bundle["numeric_features"] + bundle["categorical_features"]]

    X_transformed = bundle["preprocessor"].transform(X)
    proba = float(bundle["model"].predict_proba(X_transformed)[0, 1])
    label = int(proba >= 0.5)

    # Explanation Store: log a SHAP explanation alongside every prediction,
    # so any scored customer's decision can be audited later (Final Project,
    # Section 5), not just the prediction itself.
    if background_df is not None:
        try:
            X_t_df = pd.DataFrame(X_transformed, columns=bundle["preprocessor"].get_feature_names_out())
            log_explanation(
                bundle["model"], background_df, X_t_df,
                list(bundle["preprocessor"].get_feature_names_out()),
                proba, label, customer_ref=row.get("location", "unspecified"),
            )
        except Exception:
            pass  # explanation logging must never block a prediction response

    return PredictionResponse(
        predicted_conversion_probability=round(proba, 4),
        predicted_label=label,
        model_version=bundle["model_version"],
        model_name=bundle["model_name"],
    )
