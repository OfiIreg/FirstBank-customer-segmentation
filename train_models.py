"""
train_models.py
Train/Test Protocol & Baseline (Section 3) and Model Development (Section 4).

Trains a majority-class baseline plus three algorithms (Logistic Regression,
Random Forest, Gradient Boosting), each tuned with stratified 5-fold cross-
validated randomized search, and logs every run (hyperparameters, metrics,
and the fitted model artifact) to MLflow, per the Module 4 requirement that
the MLflow log be a real, timestamped record of the actual experiments.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)

DATA_DIR = Path("/home/claude/m4/repo/data/processed")
ARTIFACT_DIR = Path("/home/claude/m4/assets")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("sqlite:////home/claude/m4/mlruns/mlflow.db")
mlflow.set_experiment("firstbank-cross-sell-propensity")

SEED = 42


def load_split():
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    return X_train, X_test, y_train, y_test, preprocessor


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return metrics, cm, (fpr, tpr), y_pred, y_proba


def run():
    X_train, X_test, y_train, y_test, preprocessor = load_split()
    results = {}

    # ---------------- Baseline: majority class ----------------
    with mlflow.start_run(run_name="baseline_majority_class"):
        baseline = DummyClassifier(strategy="most_frequent", random_state=SEED)
        baseline.fit(X_train, y_train)
        metrics, cm, roc, y_pred, y_proba = evaluate(baseline, X_test, y_test, "baseline")
        mlflow.log_params({"strategy": "most_frequent"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
        mlflow.sklearn.log_model(baseline, name="model")
        results["baseline"] = {"metrics": metrics, "cm": cm.tolist(), "roc": (roc[0].tolist(), roc[1].tolist())}
        joblib.dump(baseline, ARTIFACT_DIR / "baseline_model.joblib")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---------------- Logistic Regression ----------------
    with mlflow.start_run(run_name="logistic_regression_tuned"):
        param_dist = {
            "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "class_weight": [None, "balanced"],
            "solver": ["lbfgs"],
        }
        search = RandomizedSearchCV(
            LogisticRegression(max_iter=1000, random_state=SEED), param_dist,
            n_iter=10, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        metrics, cm, roc, y_pred, y_proba = evaluate(best, X_test, y_test, "logreg")
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
        mlflow.sklearn.log_model(best, name="model")
        results["logistic_regression"] = {
            "metrics": metrics, "cm": cm.tolist(), "roc": (roc[0].tolist(), roc[1].tolist()),
            "best_params": search.best_params_, "cv_score": search.best_score_,
        }
        joblib.dump(best, ARTIFACT_DIR / "logreg_model.joblib")

    # ---------------- Random Forest ----------------
    with mlflow.start_run(run_name="random_forest_tuned"):
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [1, 3, 5, 10],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=SEED), param_dist,
            n_iter=12, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        metrics, cm, roc, y_pred, y_proba = evaluate(best, X_test, y_test, "rf")
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
        mlflow.sklearn.log_model(best, name="model")
        results["random_forest"] = {
            "metrics": metrics, "cm": cm.tolist(), "roc": (roc[0].tolist(), roc[1].tolist()),
            "best_params": search.best_params_, "cv_score": search.best_score_,
        }
        joblib.dump(best, ARTIFACT_DIR / "rf_model.joblib")

    # ---------------- Gradient Boosting ----------------
    with mlflow.start_run(run_name="gradient_boosting_tuned"):
        param_dist = {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.85, 1.0],
        }
        search = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=SEED), param_dist,
            n_iter=12, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        metrics, cm, roc, y_pred, y_proba = evaluate(best, X_test, y_test, "gbm")
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
        mlflow.sklearn.log_model(best, name="model")
        results["gradient_boosting"] = {
            "metrics": metrics, "cm": cm.tolist(), "roc": (roc[0].tolist(), roc[1].tolist()),
            "best_params": search.best_params_, "cv_score": search.best_score_,
        }
        joblib.dump(best, ARTIFACT_DIR / "gbm_model.joblib")

    with open(ARTIFACT_DIR / "model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    results = run()
    print("\n=== Summary ===")
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:22s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f}")
