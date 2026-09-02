"""
register_model.py
Model Registry (Module 4 technical requirement): registers the champion
model (Logistic Regression, selected on test ROC-AUC and interpretability)
in the MLflow Model Registry, versioned and staged.
"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
client = MlflowClient()

EXPERIMENT_NAME = "firstbank-cross-sell-propensity"
MODEL_NAME = "firstbank-cross-sell-propensity-model"
CHAMPION_RUN_NAME = "logistic_regression_tuned"

if __name__ == "__main__":
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(exp.experiment_id, filter_string=f"tags.mlflow.runName = '{CHAMPION_RUN_NAME}'",
                               order_by=["start_time DESC"], max_results=1)
    champion_run = runs[0]
    run_id = champion_run.info.run_id
    print(f"Champion run: {run_id} (ROC-AUC={champion_run.data.metrics.get('roc_auc'):.3f})")

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered as {MODEL_NAME}, version {result.version}")

    client.set_registered_model_alias(MODEL_NAME, "champion", result.version)
    client.set_model_version_tag(MODEL_NAME, result.version, "stage", "staging")
    client.update_model_version(
        name=MODEL_NAME, version=result.version,
        description="Logistic Regression, tuned via 5-fold CV RandomizedSearch, "
                     "ROC-AUC=0.702 on held-out test set. Champion model for the "
                     "FirstBank cross-sell propensity use case (BAN6800 Module 4).",
    )
    print(f"Aliased as 'champion', tagged stage=staging")
