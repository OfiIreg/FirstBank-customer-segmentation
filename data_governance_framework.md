# Data Governance Framework
FirstBank Nigeria Limited — Customer Segmentation Analytics Project
BAN6800, Module 3

This framework extends the Data Governance Principles set out in the Module 1
Vision Document and the Data Privacy Plan from the Module 2 report into
concrete access, usage, and retention rules for the pipeline built in this
module.

## 1. Data Zones and Access Control

| Zone | Contents | Who can access | Access method |
|---|---|---|---|
| `data/raw/` | Unmodified source extracts | Data Engineer only | Read-only service account; no analyst access |
| `data/interim/` | Ingested and cleaned intermediate tables | Data Engineer, Data Scientist | Read/write via pipeline service account |
| `data/processed/` | Final analytics table (identifiable) | Data Scientist, ML Engineer | Read-only, role-based, logged |
| `data/processed/*_anonymized.csv` | Anonymized analytics table | Marketing Analyst, Dashboard users, external reviewers | Read-only |

Access follows the least-privilege principle already committed to in the
Module 1 Ethical AI Charter: nobody outside the data engineering role can read
`data/raw/`, and only the anonymized table is available to roles outside the
core modeling team.

## 2. Retention Policy

- **Raw extracts** (`data/raw/`): retained for 90 days after each pipeline
  run, then deleted, consistent with the retention rule in the Module 1 Data
  Governance Principles ("as long as needed for model refresh").
- **Interim tables** (`data/interim/`): retained only until the next
  successful pipeline run overwrites them; not archived.
- **Processed analytics table**: retained for the duration of the active
  model version in production, then archived to cold storage for one year for
  audit purposes, then deleted.
- **Privacy audit log** (`logs/privacy_audit_log.jsonl`): retained for a
  minimum of one year, in line with typical financial-sector audit trail
  requirements, and is never deleted as part of a routine data refresh.

## 3. Usage Rules

- Data in `data/raw/` and `data/interim/` may only be used to produce the
  processed analytics table; it may not be exported, copied, or used for any
  other analysis.
- The anonymized table is the only version approved for use outside the
  pipeline's trusted boundary (e.g., dashboard demos, stakeholder review,
  this course submission).
- Any new use case for this data requires sign-off from the AI and Data
  Ethics Committee named in the Module 1 Vision Document, matching the
  "usage" principle already agreed there.

## 4. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| Data Engineer | Owns the pipeline, `data/raw/` and `data/interim/` access, and the Dockerfile/DAG |
| Data Scientist | Consumes `data/processed/`, builds the Module 4 model |
| Data Protection Officer (Module 1) | Reviews this framework and the anonymization plan quarterly |
| AI and Data Ethics Committee (Module 1) | Approves any new use of the data beyond segmentation |

## 5. Traceability

Every access to or transformation of the data is written to
`logs/privacy_audit_log.jsonl` by `src/pipeline/audit_log.py`, recording the
timestamp, actor, pipeline stage, dataset, row count affected, and a
human-readable note. This makes every number in the final analytics table
traceable back to the raw record and transformation that produced it, which
is what "reproducible" means in practice for this project.
