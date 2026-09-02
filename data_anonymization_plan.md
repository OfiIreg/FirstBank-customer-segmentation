# Data Anonymization Plan
FirstBank Nigeria Limited — Customer Segmentation Analytics Project
BAN6800, Module 3

Implemented in `src/pipeline/anonymize.py`. This plan covers the fields in
the Module 2 Data Dictionary; the Kaggle proxy dataset contains no names,
phone numbers, or account numbers, so the realistic re-identification risk in
this table comes from quasi-identifiers (age, location, balance) rather than
direct identifiers.

## 1. Direct Identifiers

| Field | Treatment |
|---|---|
| `customer_id` | Irreversibly pseudonymized with a salted SHA-256 hash (`ANON_<16-char digest>`). The salt is held outside source control in production. The same input always produces the same output, so joins across tables still work, but the original ID cannot be recovered from the hash. |

## 2. Quasi-Identifiers (Generalization)

Exact values are generalized into bands, a standard k-anonymity technique, so
no single row can be isolated by combining age, balance, and location:

| Field | Original | Generalized to |
|---|---|---|
| `age` | Exact integer | `age_band` (18-24, 25-34, 35-44, 45-59, 60+) |
| `account_balance` | Exact amount | `balance_band` (Under 50K, 50K-500K, 500K-2M, 2M+) |
| `location` | State (e.g., Lagos) | `geo_zone` (e.g., South West) |

## 3. What Is Never Included

The pipeline never writes customer names, phone numbers, email addresses, or
BVN/NIN identifiers into any analytics table, dashboard extract, or export,
since none of these exist in the source schema defined in Module 2. If a
future module connects to FirstBank's real core banking data, this plan must
be revisited before ingestion, since real extracts would contain such fields.

## 4. When Anonymization Is Applied

The pseudonymized/generalized table
(`data/processed/customer_analytics_table_anonymized.csv`) is the only
version approved for use outside the pipeline's trusted boundary, per the
Data Governance Framework (Section 3). The identifiable
`customer_analytics_table.csv` stays inside `data/processed/` and is used
only for model training in Module 4, where re-identifiable customer IDs are
still needed to write segment labels back to the source system.

## 5. Limitations

Generalization reduces but does not eliminate re-identification risk for
very small population cells (e.g., a customer in the "60+" age band, "2M+"
balance band, and a low-population `geo_zone`). The Bias Detection Suite
(`src/pipeline/bias_detection.py`) already flags groups under 5% of the
population; the same output is used here as an early warning for small-cell
disclosure risk, and any flagged cell is reviewed by the Data Protection
Officer before the anonymized table is shared externally.
