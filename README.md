# FirstBank Cross-Sell Propensity System — Final Integrated Project

BAN6800 Business Analytics Capstone | Ofure Mariane Iregbeyen | Nexford University

A complete, integrated analytics system for FirstBank Nigeria Limited:
customer segmentation and cross-sell propensity scoring, from business
vision through a monitored, containerized, deployable system.

## Live Assets

| Asset | URL / Status |
|---|---|
| **Live Dashboard** | https://firstbank-customer-segmentation-5fqdeelrnmuhxvynun53yg.streamlit.app/ — deployed on Streamlit Community Cloud |
| **GitHub Repository** | https://github.com/OfiIreg/FirstBank-customer-segmentation |
| **API Endpoint** | `src/api/main.py` — tested locally- https://firstbank-cross-sell-api.onrender.com
| **Monitoring** | `monitoring/run_drift_check.py` — tested locally against real train/test data; scheduled weekly via `.github/workflows/ci-cd.yml`. No public monitoring dashboard is hosted; the script writes `monitoring_result.json`, suitable for a Grafana/Prometheus frontend if deployed |

## Project Continuity

| Module | Delivered |
|---|---|
| 1 | Vision Document — problem, Ethical AI Charter, technical stack |
| 2 | Project Overview — architecture, agile plan, RAID log, fairness objectives |
| 3 | Data pipeline — ingestion through anonymization, Great Expectations validation, bias detection |
| 4 | Predictive model — MLflow-tracked training, SHAP/LIME/counterfactual XAI, fairness audit + mitigation, FastAPI |
| 5 | Stakeholder presentation + live Streamlit dashboard |
| **Final** | Integration: reconciliation across all modules, continuous monitoring, Explanation Store, Docker Compose, extended CI/CD, consolidated ethical governance |

## What's New in This Integration

- `monitoring/run_drift_check.py` — real Evidently AI drift check + Fairlearn fairness re-check, runnable and tested
- `monitoring/explanation_store.py` — logs a SHAP explanation for every API prediction, for audit
- `docker-compose.yml` + `Dockerfile.api` / `Dockerfile.dashboard` / `Dockerfile.monitoring` / `Dockerfile.pipeline` — multi-container deployment
- `.github/workflows/ci-cd.yml` — extended CI/CD: pipeline tests, model training, monitoring check, image builds
- `docs/ethical_ai_framework.md`, `docs/algorithmic_impact_assessment.md`, `docs/regulatory_compliance_report.md`, `docs/public_trust_statement.md`, `docs/governance_structure.md`, `docs/monitoring_plan.md` — consolidated governance layer

## Running the System

```bash
pip install -r requirements.txt --break-system-packages

# Run the full data pipeline
python src/pipeline/generate_raw_data.py
python src/pipeline/pipeline_dag.py

# Train and register the model
python src/models/train_models.py
python src/models/register_model.py

# Run the fairness audit and monitoring check
python src/fairness/fairness_analysis.py
python monitoring/run_drift_check.py

# Serve the API (with Explanation Store)
cd src/api && uvicorn main:app --port 8000

# Run the dashboard locally
cd dashboard && streamlit run app.py

# Or run everything containerized
docker compose up --build
```

## Repository Structure

```
firstbank-customer-segmentation/
  README.md
  docs/                       vision, governance, model card, fairness, ethics, this integration's new docs
    report_figures/           all figures used in the Final Report
  data/raw/  data/interim/  data/processed/
  notebooks/                  shap_analysis.ipynb
  src/
    pipeline/                 Module 3: ETL DAG
    models/                   Module 4: training, XAI, sensitivity
    fairness/                 Module 4: fairness audit + mitigation
    api/                      Module 4/Final: FastAPI + Explanation Store
  monitoring/                 Final: drift check, explanation store, fairness re-check
  dashboard/                  Module 5: Streamlit app
  tests/                      pytest unit tests
  great_expectations/         validation results
  logs/                       privacy audit log, DAG run log
  .github/workflows/          CI/CD pipeline
  docker-compose.yml
  Dockerfile.api / .dashboard / .monitoring / .pipeline
  requirements.txt
```
