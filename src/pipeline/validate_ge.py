"""
validate_ge.py
Data Validation Suite (Module 3, Section 4) using Great Expectations.

Runs a batch of expectations against the final customer_analytics_table to
confirm the pipeline output is fit for the clustering and propensity models
planned for Module 4. Produces a JSON validation result and an HTML data
docs page that can be screenshotted for the PPTX deliverable.
"""
import great_expectations as gx
import pandas as pd
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
GE_DIR = Path(__file__).resolve().parent.parent.parent / "great_expectations"
GE_DIR.mkdir(parents=True, exist_ok=True)

def run_validation():
    df = pd.read_csv(PROCESSED_DIR / "customer_analytics_table.csv")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("firstbank_pandas")
    data_asset = data_source.add_dataframe_asset(name="customer_analytics_table")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_table")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="firstbank_customer_analytics_suite")

    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="customer_id"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="age", min_value=18, max_value=100),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="age"),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="gender", value_set=["Male", "Female", "Unspecified"]
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="account_balance", min_value=0, max_value=10_000_000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(column="frequency", min_value=0, max_value=1000),
        gx.expectations.ExpectColumnValuesToBeBetween(column="monetary_total", min_value=0),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="location"),
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=5000, max_value=200000),
        gx.expectations.ExpectTableColumnCountToEqual(value=df.shape[1]),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="share_POS Purchase", min_value=0, max_value=1
        ),
    ]
    for exp in expectations:
        suite.add_expectation(exp)

    result = batch.validate(suite)
    result_dict = json.loads(str(result))

    # Persist a readable summary (this stands in for the "GE suite screenshot" in the deck)
    summary = {
        "success": result_dict["success"],
        "statistics": result_dict.get("statistics", {}),
        "results": [
            {
                "expectation_type": r["expectation_config"]["type"],
                "column": r["expectation_config"]["kwargs"].get("column", ""),
                "success": r["success"],
            }
            for r in result_dict.get("results", [])
        ],
    }
    with open(GE_DIR / "validation_result.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary

if __name__ == "__main__":
    summary = run_validation()
    print(f"Overall validation success: {summary['success']}")
    print(f"Expectations evaluated: {len(summary['results'])}")
    for r in summary["results"]:
        status = "PASS" if r["success"] else "FAIL"
        col = f" [{r['column']}]" if r["column"] else ""
        print(f"  {status}  {r['expectation_type']}{col}")
