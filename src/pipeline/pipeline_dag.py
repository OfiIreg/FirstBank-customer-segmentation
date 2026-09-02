"""
pipeline_dag.py
Orchestration DAG (Module 3, Section 2), implemented with Prefect.

Defines the FirstBank customer segmentation ETL workflow as a directed graph
of tasks: ingest -> clean -> transform -> integrate -> validate -> bias_check
-> anonymize. Each function is decorated as a Prefect @task, and dependencies
between tasks are expressed by passing one task's return value into the next,
which is exactly how Prefect builds its DAG at runtime (equivalent to
Airflow's `>>` operator chaining). Run directly with:

    python pipeline_dag.py

or orchestrated on a schedule via `prefect deploy` in a production environment.
"""
from prefect import flow, task, get_run_logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ingest as ingest_mod
import clean as clean_mod
import transform as transform_mod
import integrate as integrate_mod
import validate_ge as validate_mod
import bias_detection as bias_mod
import anonymize as anonymize_mod


@task(name="ingest", retries=1, retry_delay_seconds=10)
def ingest_task():
    logger = get_run_logger()
    n_cust, n_txn = ingest_mod.ingest()
    logger.info(f"Ingested {n_cust} customers, {n_txn} transactions")
    return n_cust, n_txn


@task(name="clean")
def clean_task(_upstream):
    logger = get_run_logger()
    customers, txns = clean_mod.run()
    logger.info(f"Cleaned to {len(customers)} customers, {len(txns)} transactions")
    return len(customers), len(txns)


@task(name="transform")
def transform_task(_upstream):
    logger = get_run_logger()
    features = transform_mod.run()
    logger.info(f"Transformed to {len(features)} customer feature rows")
    return len(features)


@task(name="integrate")
def integrate_task(_upstream):
    logger = get_run_logger()
    final = integrate_mod.run()
    logger.info(f"Integrated final table: {final.shape}")
    return final.shape


@task(name="validate")
def validate_task(_upstream):
    logger = get_run_logger()
    summary = validate_mod.run_validation()
    logger.info(f"Validation success={summary['success']}, "
                f"{summary['statistics']['successful_expectations']}/"
                f"{summary['statistics']['evaluated_expectations']} expectations passed")
    if not summary["success"]:
        raise ValueError("Data validation failed - halting pipeline before model handoff")
    return summary


@task(name="bias_check")
def bias_check_task(_upstream):
    logger = get_run_logger()
    results = bias_mod.run()
    logger.info(f"Bias flags: {results['representation_flags']}")
    return results


@task(name="anonymize")
def anonymize_task(_upstream):
    logger = get_run_logger()
    anon = anonymize_mod.run()
    logger.info(f"Anonymized output: {anon.shape}")
    return anon.shape


@flow(name="firstbank-customer-segmentation-etl")
def firstbank_etl_pipeline():
    """
    End-to-end ETL DAG for the FirstBank Nigeria customer segmentation project.
    Stage order: ingest -> clean -> transform -> integrate -> validate ->
    bias_check -> anonymize. Each stage depends on the prior stage's task
    future, so Prefect executes them in this sequence and would parallelize
    any independent branches automatically if the graph had them.
    """
    ingested = ingest_task()
    cleaned = clean_task(ingested)
    transformed = transform_task(cleaned)
    integrated = integrate_task(transformed)
    validated = validate_task(integrated)
    bias_checked = bias_check_task(validated)
    anonymized = anonymize_task(bias_checked)
    return anonymized


if __name__ == "__main__":
    result = firstbank_etl_pipeline()
    print(f"\nPipeline complete. Final anonymized table shape: {result}")
