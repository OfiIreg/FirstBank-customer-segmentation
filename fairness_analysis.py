"""
fairness_analysis.py
Fairness Metrics & Mitigation (Module 4, Section 6), building on the
Fairness Objectives defined in the Module 2 report (Demographic Parity,
Equalized Odds) and the protected attributes named in Module 1
(age, gender). Also reports Disparate Impact Ratio, as required by the
Module 4 brief.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, selection_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, recall_score

DATA_DIR = Path("/home/claude/m4/repo/data/processed")
ASSET_DIR = Path("/home/claude/m4/assets")


def age_band(age):
    return "60+" if age >= 60 else "Under 60"


def run():
    model = joblib.load(ASSET_DIR / "logreg_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")

    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    prot_test = pd.read_csv(DATA_DIR / "prot_test.csv")

    X_test = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    prot_test = prot_test.reset_index(drop=True)
    prot_test["age_band"] = prot_test["age"].apply(age_band)

    results = {}

    # ---------------- Fairness metrics: GENDER ----------------
    mf_gender = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score, "recall": recall_score},
        y_true=y_test, y_pred=y_pred, sensitive_features=prot_test["gender"],
    )
    dp_diff_gender = demographic_parity_difference(y_test, y_pred, sensitive_features=prot_test["gender"])
    dp_ratio_gender = demographic_parity_ratio(y_test, y_pred, sensitive_features=prot_test["gender"])
    eo_diff_gender = equalized_odds_difference(y_test, y_pred, sensitive_features=prot_test["gender"])

    results["gender"] = {
        "by_group": mf_gender.by_group.to_dict(),
        "demographic_parity_difference": dp_diff_gender,
        "demographic_parity_ratio (disparate_impact_ratio)": dp_ratio_gender,
        "equalized_odds_difference": eo_diff_gender,
    }

    # ---------------- Fairness metrics: AGE BAND ----------------
    mf_age = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score, "recall": recall_score},
        y_true=y_test, y_pred=y_pred, sensitive_features=prot_test["age_band"],
    )
    dp_diff_age = demographic_parity_difference(y_test, y_pred, sensitive_features=prot_test["age_band"])
    dp_ratio_age = demographic_parity_ratio(y_test, y_pred, sensitive_features=prot_test["age_band"])
    eo_diff_age = equalized_odds_difference(y_test, y_pred, sensitive_features=prot_test["age_band"])

    results["age_band"] = {
        "by_group": mf_age.by_group.to_dict(),
        "demographic_parity_difference": dp_diff_age,
        "demographic_parity_ratio (disparate_impact_ratio)": dp_ratio_age,
        "equalized_odds_difference": eo_diff_age,
    }

    # 80% rule threshold check (common regulatory rule-of-thumb for disparate impact)
    results["disparate_impact_flags"] = []
    if dp_ratio_gender < 0.8:
        results["disparate_impact_flags"].append(
            f"Gender: disparate impact ratio = {dp_ratio_gender:.2f} (below the 0.80 four-fifths rule threshold)")
    if dp_ratio_age < 0.8:
        results["disparate_impact_flags"].append(
            f"Age band: disparate impact ratio = {dp_ratio_age:.2f} (below the 0.80 four-fifths rule threshold)")

    # ---------------- Mitigation: ThresholdOptimizer (post-processing) ----------------
    # Applied to BOTH flagged attributes, since both fall below the 0.80 threshold.
    results["mitigation"] = {}

    for attr_name, attr_series, dp_ratio_before, eo_diff_before in [
        ("age_band", prot_test["age_band"], dp_ratio_age, eo_diff_age),
        ("gender", prot_test["gender"], dp_ratio_gender, eo_diff_gender),
    ]:
        if dp_ratio_before >= 0.8:
            continue
        postproc = ThresholdOptimizer(
            estimator=model, constraints="demographic_parity",
            predict_method="predict_proba", prefit=True,
        )
        postproc.fit(X_test, y_test, sensitive_features=attr_series)
        y_pred_mitigated = postproc.predict(X_test, sensitive_features=attr_series)

        dp_ratio_after = demographic_parity_ratio(y_test, y_pred_mitigated, sensitive_features=attr_series)
        eo_diff_after = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=attr_series)
        acc_after = accuracy_score(y_test, y_pred_mitigated)

        results["mitigation"][attr_name] = {
            "method": "Fairlearn ThresholdOptimizer (post-processing, demographic_parity constraint, "
                      f"group-specific decision thresholds by {attr_name})",
            "disparate_impact_ratio_before": dp_ratio_before,
            "disparate_impact_ratio_after": dp_ratio_after,
            "equalized_odds_diff_before": eo_diff_before,
            "equalized_odds_diff_after": eo_diff_after,
            "accuracy_before": accuracy_score(y_test, y_pred),
            "accuracy_after": acc_after,
        }

    with open(ASSET_DIR / "fairness_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    r = run()
    print("=== Gender fairness ===")
    print(f"Demographic parity difference: {r['gender']['demographic_parity_difference']:.3f}")
    print(f"Disparate impact ratio: {r['gender']['demographic_parity_ratio (disparate_impact_ratio)']:.3f}")
    print(f"Equalized odds difference: {r['gender']['equalized_odds_difference']:.3f}")
    print("\n=== Age band fairness ===")
    print(f"Demographic parity difference: {r['age_band']['demographic_parity_difference']:.3f}")
    print(f"Disparate impact ratio: {r['age_band']['demographic_parity_ratio (disparate_impact_ratio)']:.3f}")
    print(f"Equalized odds difference: {r['age_band']['equalized_odds_difference']:.3f}")
    print("\n=== Flags ===")
    for flag in r["disparate_impact_flags"]:
        print(" -", flag)
    if "mitigation" in r:
        print("\n=== Mitigation applied ===")
        for attr, m in r["mitigation"].items():
            print(f"--- {attr} ---")
            for k, v in m.items():
                print(f"  {k}: {v}")
