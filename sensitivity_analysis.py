"""
sensitivity_analysis.py
Sensitivity Analysis (Module 4, Section 6): tests how much the champion
model's predictions move under small, realistic perturbations of the input
features, to check the model isn't brittle (small data noise flipping
decisions) before it is trusted for the Module 5 dashboard.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

DATA_DIR = Path("/home/claude/m4/repo/data/processed")
ASSET_DIR = Path("/home/claude/m4/assets")

SEED = 42
np.random.seed(SEED)

PERTURB_FEATURES = ["frequency", "monetary_total", "account_balance", "recency_days"]
NOISE_LEVELS = [0.02, 0.05, 0.10]  # +/- 2%, 5%, 10% Gaussian noise


def run():
    model = joblib.load(ASSET_DIR / "logreg_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")

    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    X_test = preprocessor.transform(X_test_raw)
    baseline_pred = model.predict(X_test)
    baseline_proba = model.predict_proba(X_test)[:, 1]

    results = {}
    for noise_level in NOISE_LEVELS:
        flips = 0
        proba_shifts = []
        n_trials = 200
        rng = np.random.RandomState(SEED)
        sample_idx = rng.choice(len(X_test_raw), size=n_trials, replace=False)

        for i in sample_idx:
            row = X_test_raw.iloc[i].copy()
            for feat in PERTURB_FEATURES:
                noise = rng.normal(0, noise_level * abs(row[feat] if row[feat] != 0 else 1))
                row[feat] = max(0, row[feat] + noise)
            perturbed_X = preprocessor.transform(pd.DataFrame([row]))
            perturbed_pred = model.predict(perturbed_X)[0]
            perturbed_proba = model.predict_proba(perturbed_X)[0, 1]

            if perturbed_pred != baseline_pred[i]:
                flips += 1
            proba_shifts.append(abs(perturbed_proba - baseline_proba[i]))

        results[f"noise_{int(noise_level*100)}pct"] = {
            "n_trials": n_trials,
            "decision_flip_rate": flips / n_trials,
            "mean_abs_probability_shift": float(np.mean(proba_shifts)),
            "max_abs_probability_shift": float(np.max(proba_shifts)),
        }

    with open(ASSET_DIR / "sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    r = run()
    print("Sensitivity analysis (perturbing frequency, monetary_total, account_balance, recency_days):\n")
    for level, stats in r.items():
        print(f"{level}: decision flip rate = {stats['decision_flip_rate']:.1%}, "
              f"mean |P shift| = {stats['mean_abs_probability_shift']:.3f}, "
              f"max |P shift| = {stats['max_abs_probability_shift']:.3f}")
