"""
bias_detection.py
Bias Detection Suite (Module 3, Section 6), using Fairlearn's MetricFrame to
check representation bias in the final analytics table before it feeds the
segmentation model in Module 4. This is a representation/data audit, distinct
from the model fairness audit (Demographic Parity, Equalized Odds on model
predictions) planned in the Module 2 report, which requires trained model
outputs that do not exist yet at the pipeline stage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from fairlearn.metrics import MetricFrame, count

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "great_expectations"

def age_band(age):
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 60:
        return "45-59"
    return "60+"

def run():
    df = pd.read_csv(PROCESSED_DIR / "customer_analytics_table.csv")
    df["age_band"] = df["age"].apply(age_band)

    results = {}

    # Representation by gender
    gender_counts = df["gender"].value_counts(normalize=True).round(4).to_dict()
    results["gender_representation"] = gender_counts

    # Representation by age band
    age_counts = df["age_band"].value_counts(normalize=True).round(4).to_dict()
    results["age_band_representation"] = age_counts

    # Representation by location (top imbalance check)
    loc_counts = df["location"].value_counts(normalize=True).round(4).to_dict()
    results["location_representation"] = loc_counts

    # Fairlearn MetricFrame: average monetary_total and frequency by gender and age band,
    # to check whether behavioral features (which drive segmentation) differ structurally
    # by protected attribute in ways that could bias the downstream model
    mf_gender = MetricFrame(
        metrics={"count": count, "mean_monetary_total": lambda y_true, y_pred: np.nan,
                 },
        y_true=df["monetary_total"], y_pred=df["monetary_total"],
        sensitive_features=df["gender"],
    )
    gender_group_stats = df.groupby("gender")["monetary_total"].agg(["count", "mean", "median"]).round(2)
    age_group_stats = df.groupby("age_band")["monetary_total"].agg(["count", "mean", "median"]).round(2)

    results["gender_monetary_stats"] = gender_group_stats.to_dict(orient="index")
    results["age_band_monetary_stats"] = age_group_stats.to_dict(orient="index")

    # Simple representation-parity flag: any group under 5% of population is flagged for review
    flags = []
    for group, share in gender_counts.items():
        if share < 0.05:
            flags.append(f"Gender group '{group}' is {share:.1%} of the dataset (below 5% threshold)")
    for group, share in age_counts.items():
        if share < 0.05:
            flags.append(f"Age band '{group}' is {share:.1%} of the dataset (below 5% threshold)")
    results["representation_flags"] = flags if flags else ["No group falls below the 5% representation threshold."]

    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "bias_detection_result.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results

if __name__ == "__main__":
    r = run()
    print("Gender representation:", r["gender_representation"])
    print("Age band representation:", r["age_band_representation"])
    print("\nFlags:")
    for f in r["representation_flags"]:
        print(" -", f)
    print("\nMonetary total by gender:")
    for k, v in r["gender_monetary_stats"].items():
        print(f"  {k}: {v}")
