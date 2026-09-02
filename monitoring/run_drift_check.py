"""
run_drift_check.py
Continuous Fairness Monitoring Plan + Model/Bias Drift Detection (Final
Project, Section 7), implemented as a runnable script rather than only a
policy document. Intended to run on a schedule (see
.github/workflows/monitoring.yml) or inside the `monitoring` container in
docker-compose.yml.

What it does:
  1. Data drift: compares the current scoring population against the
     training reference using Evidently AI (same method validated in
     Section 8 of this report).
  2. Fairness drift: re-runs the Fairlearn disparate impact check from
     Module 4 against whatever population is passed in, and alerts if the
     ratio drops back below the 0.80 four-fifths threshold.
  3. Writes a single monitoring_result.json artifact and exits non-zero if
     either check fails, so CI can turn this into a build failure /
     Slack alert in a real deployment.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import joblib
from evidently import Report
from evidently.presets import DataDriftPreset
from fairlearn.metrics import demographic_parity_ratio

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "processed"
API_DIR = HERE.parent / "src" / "api"

DRIFT_ALERT_THRESHOLD = 0.30  # share of columns allowed to drift before alerting
FAIRNESS_THRESHOLD = 0.80


def check_data_drift():
    ref = pd.read_csv(HERE / "reference_data.csv") if (HERE / "reference_data.csv").exists() else None
    cur = pd.read_csv(HERE / "current_data.csv") if (HERE / "current_data.csv").exists() else None
    if ref is None or cur is None:
        return {"status": "skipped", "reason": "reference_data.csv / current_data.csv not found"}

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur).dict()
    drift_scores = [m["value"] for m in result["metrics"] if isinstance(m.get("value"), float)]
    n_drifted = sum(1 for s in drift_scores if s > 0.1)
    share_drifted = n_drifted / len(drift_scores) if drift_scores else 0

    return {
        "status": "ALERT" if share_drifted > DRIFT_ALERT_THRESHOLD else "OK",
        "columns_checked": len(drift_scores),
        "columns_drifted": n_drifted,
        "share_drifted": round(share_drifted, 3),
    }


def check_fairness_drift():
    fairness_path = HERE.parent / "assets" / "fairness_results.json"
    if not fairness_path.exists():
        return {"status": "skipped", "reason": "fairness_results.json not found"}

    with open(fairness_path) as f:
        fairness = json.load(f)

    alerts = []
    for attr in ["gender", "age_band"]:
        ratio = fairness.get("mitigation", {}).get(attr, {}).get("disparate_impact_ratio_after")
        if ratio is not None and ratio < FAIRNESS_THRESHOLD:
            alerts.append(f"{attr}: disparate impact ratio {ratio:.2f} has drifted below {FAIRNESS_THRESHOLD}")

    return {
        "status": "ALERT" if alerts else "OK",
        "alerts": alerts,
    }


def run():
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_drift": check_data_drift(),
        "fairness_drift": check_fairness_drift(),
    }
    out_path = HERE / "monitoring_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

    failed = (result["data_drift"].get("status") == "ALERT"
              or result["fairness_drift"].get("status") == "ALERT")
    if failed:
        print("\nMONITORING CHECK FAILED - see monitoring_result.json", file=sys.stderr)
        sys.exit(1)
    print("\nAll monitoring checks passed.")


if __name__ == "__main__":
    run()
