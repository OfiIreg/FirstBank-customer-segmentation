"""
explanation_store.py
Explanation Store (Final Project, Section 5): persists a SHAP explanation
alongside every prediction the API serves, so any scored customer's
decision can be reconstructed and audited later, not just the prediction
itself. This extends the privacy audit logging pattern established in
Module 3 (logs/privacy_audit_log.jsonl) to cover model explanations.

Each entry is one append-only JSON line:
{
  "timestamp": ..., "customer_ref": ..., "predicted_probability": ...,
  "predicted_label": ..., "top_features": [{"feature": ..., "shap_value": ...}, ...]
}

This is what lets FirstBank answer "why did this specific customer get
this score" months later, per the contestability commitment in the
Module 1 Ethical AI Vision and the Module 5 Transparency Statement.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import shap

STORE_DIR = Path(__file__).parent / "logs"
STORE_DIR.mkdir(parents=True, exist_ok=True)
STORE_FILE = STORE_DIR / "explanation_store.jsonl"

logger = logging.getLogger("explanation_store")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(STORE_FILE)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

_explainer_cache = {}


def get_explainer(model, background_df):
    key = id(model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.LinearExplainer(model, background_df)
    return _explainer_cache[key]


def log_explanation(model, background_df, X_row_df, feature_names, proba, label,
                     customer_ref="unspecified", top_n=5):
    """Compute and persist a SHAP explanation for one scored customer."""
    explainer = get_explainer(model, background_df)
    shap_values = explainer(X_row_df)

    vals = shap_values.values[0]
    order = np.argsort(-np.abs(vals))[:top_n]
    top_features = [
        {"feature": feature_names[i], "shap_value": round(float(vals[i]), 4)}
        for i in order
    ]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_ref": customer_ref,
        "predicted_probability": round(float(proba), 4),
        "predicted_label": int(label),
        "top_features": top_features,
    }
    logger.info(json.dumps(entry))
    return entry
