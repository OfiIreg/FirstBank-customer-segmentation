"""
xai_shap.py
Explainable AI (Module 4, Section 6): Global Feature Importance (SHAP) and
Local Explanations (SHAP force/waterfall for individual predictions), run
against the champion model (Logistic Regression) registered in MLflow.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/home/claude/m4/repo/data/processed")
ASSET_DIR = Path("/home/claude/m4/assets")

NAVY = "#1F3864"

def run():
    model = joblib.load(ASSET_DIR / "logreg_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")
    feature_names = joblib.load(DATA_DIR / "feature_names.joblib")

    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Linear model -> SHAP LinearExplainer (exact, fast, appropriate for the champion model)
    explainer = shap.LinearExplainer(model, X_train_df)
    shap_values = explainer(X_test_df)

    # ---------------- Global: SHAP summary (beeswarm) ----------------
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=12)
    plt.title("Figure 4. SHAP Global Feature Importance (Logistic Regression)", fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=15)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------- Global: mean |SHAP| bar ----------------
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=12)
    plt.title("Figure 5. Mean |SHAP Value| by Feature (Global Importance)", fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=15)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------- Local: waterfall for one converter and one non-converter ----------------
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    converter_idx = np.where((y_test.values == 1) & (y_pred_proba > 0.5))[0]
    non_converter_idx = np.where((y_test.values == 0) & (y_pred_proba < 0.3))[0]

    for label, idx_arr, fname in [
        ("predicted_converter", converter_idx, "shap_local_converter.png"),
        ("predicted_non_converter", non_converter_idx, "shap_local_non_converter.png"),
    ]:
        i = idx_arr[0]
        plt.figure(figsize=(8.5, 5.5))
        shap.plots.waterfall(shap_values[i], show=False, max_display=10)
        plt.title(f"Figure 6{'a' if 'non' not in label else 'b'}. Local Explanation: {label.replace('_', ' ').title()}\n"
                  f"(customer_id={pd.read_csv(DATA_DIR / 'id_test.csv').iloc[i, 0]}, predicted P(convert)={y_pred_proba[i]:.2f})",
                  fontsize=10.5, fontweight="bold", color=NAVY, loc="left")
        plt.tight_layout()
        plt.savefig(ASSET_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close()

    # Save mean |SHAP| ranking as a table for the report
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(ASSET_DIR / "shap_feature_importance.csv", index=False)

    print("SHAP analysis complete.")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


if __name__ == "__main__":
    run()
