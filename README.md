# FirstBank Customer Segmentation — Analytics Pipeline

BAN6800 Business Analytics Capstone — Modules 1-3
Vertical: First Bank of Nigeria Limited | Problem: Retail customer segmentation

## Project Continuity
- **Module 1**: Vision Document — problem, ethics, technical stack (`docs/`)
- **Module 2**: Project Overview Report — architecture, agile plan, RAID log (`docs/`)
- **Module 3 (this module)**: Data pipeline — ingestion, cleaning, transformation,
  integration, validation, bias detection, anonymization (`src/pipeline/`)

## Data
`src/pipeline/generate_raw_data.py` produces a synthetic proxy dataset (6,090
customers / 36,267 transactions) matching the schema and characteristics of
the Kaggle "Bank Customer Segmentation" dataset identified in Module 1
(Bansal, 2021), with realistic messiness injected (missing values,
duplicates, dirty category labels, outliers) for the pipeline to resolve.

## Running the Pipeline

```bash
pip install -r requirements.txt --break-system-packages

# Generate the raw synthetic dataset
python src/pipeline/generate_raw_data.py

# Run the full orchestrated DAG (ingest -> clean -> transform -> integrate
# -> validate -> bias_check -> anonymize)
python src/pipeline/pipeline_dag.py

# Run the unit tests
pytest tests/ -v

# Run in Docker
docker build -t firstbank-pipeline .
docker run --rm firstbank-pipeline
```

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| Ingestion | `src/pipeline/ingest.py` | `data/interim/*_ingested.csv` |
| Cleaning | `src/pipeline/clean.py` | `data/interim/*_clean.csv` |
| Transformation | `src/pipeline/transform.py` | `data/interim/customer_txn_features.csv` |
| Integration | `src/pipeline/integrate.py` | `data/processed/customer_analytics_table.csv` |
| Validation | `src/pipeline/validate_ge.py` | `great_expectations/validation_result.json` |
| Bias Detection | `src/pipeline/bias_detection.py` | `great_expectations/bias_detection_result.json` |
| Anonymization | `src/pipeline/anonymize.py` | `data/processed/customer_analytics_table_anonymized.csv` |

Orchestrated end-to-end by `src/pipeline/pipeline_dag.py` (Prefect).

## Governance and Ethics

See `docs/data_governance_framework.md` and `docs/data_anonymization_plan.md`,
both built on the Ethical AI Charter and Fairness Objectives from Module 1
and the Data Privacy Plan from Module 2. Every data access or transformation
is logged to `logs/privacy_audit_log.jsonl`.

## Repository Structure

```
firstbank-customer-segmentation/
  README.md
  docs/                          vision, project overview, governance, anonymization plan
  data/raw/  data/interim/  data/processed/
  notebooks/
  src/
    pipeline/                    ingest, clean, transform, integrate, validate, bias, anonymize, DAG
    models/                      (Module 4)
    fairness/                    (Module 4)
    api/                         (Module 5)
  dashboards/segment-explorer/   (Module 5)
  tests/                         pytest unit tests
  great_expectations/            validation results and HTML report
  logs/                          privacy audit log, DAG run log
  .github/workflows/             CI pipeline
  Dockerfile
  requirements.txt
```
