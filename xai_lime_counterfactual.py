"""
xai_lime_counterfactual.py
Explainable AI (Module 4, Section 6): Local Explanations via LIME (as an
independent cross-check on the SHAP local explanations) and Counterfactual
Explanations ("what would need to change for a different outcome").

Counterfactual approach: a lightweight, documented grid-search counterfactual
generator (perturbing one actionable feature at a time along its observed
range and re-scoring with the trained model) is used in place of a package
like DiCE, since DiCE's dependency stack was not reliably installable in
this environment. The method finds the same thing DiCE looks for: the
smallest change in an actionable feature that flips the model's decision.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

DATA_DIR = Path("/home/claude/m4/repo/data/processed")
ASSET_DIR = Path("/home/claude/m4/assets")


def run_lime():
    model = joblib.load(ASSET_DIR / "logreg_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")
    feature_names = joblib.load(DATA_DIR / "feature_names.joblib")

    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    id_test = pd.read_csv(DATA_DIR / "id_test.csv")

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["No Conversion", "Conversion"],
        mode="classification", discretize_continuous=True, random_state=42,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    idx = np.where((y_test.values == 1) & (y_proba > 0.5))[0][0]

    exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=8)
    lines = [f"LIME local explanation for customer_id={id_test.iloc[idx, 0]} "
             f"(predicted P(convert)={y_proba[idx]:.2f}, actual=Conversion)", ""]
    for feat, weight in exp.as_list():
        direction = "pushes toward CONVERSION" if weight > 0 else "pushes toward NO conversion"
        lines.append(f"  {feat:35s}  weight={weight:+.3f}  ({direction})")

    exp.save_to_file(str(ASSET_DIR / "lime_explanation.html"))
    with open(ASSET_DIR / "lime_explanation.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return exp


def run_counterfactual():
    """
    For a customer predicted NOT to convert, search for the smallest change
    in one actionable feature (frequency, recency_days, or monetary_total)
    that flips the prediction to "likely to convert" (P >= 0.5). Actionable
    here means a feature the business could realistically influence through
    an intervention (e.g., an engagement nudge that increases transaction
    frequency), unlike age or gender, which are excluded from this search.
    """
    model = joblib.load(ASSET_DIR / "logreg_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")
    feature_names = joblib.load(DATA_DIR / "feature_names.joblib")

    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    id_test = pd.read_csv(DATA_DIR / "id_test.csv")

    X_test = preprocessor.transform(X_test_raw)
    y_proba = model.predict_proba(X_test)[:, 1]

    non_converter_idx = np.where((y_test.values == 0) & (y_proba < 0.3))[0][0]
    original_row = X_test_raw.iloc[non_converter_idx].copy()
    original_prob = y_proba[non_converter_idx]

    results = [f"Counterfactual search for customer_id={id_test.iloc[non_converter_idx, 0]} "
               f"(current predicted P(convert)={original_prob:.2f})", ""]

    ACTIONABLE_FEATURES = ["frequency", "monetary_total", "recency_days"]
    for feat in ACTIONABLE_FEATURES:
        current_val = original_row[feat]
        if feat == "recency_days":
            search_range = np.linspace(current_val, max(0, current_val - 60), 30)  # more recent = lower
        else:
            search_range = np.linspace(current_val, current_val * 3, 30)  # more engagement = higher

        found = None
        for val in search_range:
            trial_row = original_row.copy()
            trial_row[feat] = val
            trial_X = preprocessor.transform(pd.DataFrame([trial_row]))
            trial_prob = model.predict_proba(trial_X)[0, 1]
            if trial_prob >= 0.5:
                found = (val, trial_prob)
                break

        if found:
            new_val, new_prob = found
            pct_change = (new_val - current_val) / (abs(current_val) + 1e-6) * 100
            results.append(
                f"  If {feat} changed from {current_val:.1f} to {new_val:.1f} "
                f"({pct_change:+.0f}%), predicted P(convert) rises from "
                f"{original_prob:.2f} to {new_prob:.2f} -> crosses the 0.5 decision threshold."
            )
        else:
            results.append(f"  No realistic change in {feat} alone (within 3x its current value) "
                            f"flips this customer's prediction.")

    with open(ASSET_DIR / "counterfactual_explanation.txt", "w") as f:
        f.write("\n".join(results))
    print("\n".join(results))
    return results


if __name__ == "__main__":
    run_lime()
    print("\n" + "=" * 70 + "\n")
    run_counterfactual()
