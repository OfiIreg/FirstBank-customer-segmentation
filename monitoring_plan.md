# Implementation and Monitoring Plan
FirstBank Nigeria Limited — Cross-Sell Propensity Analytics Project
BAN6800 Final Project

Builds on the monitoring hooks first proposed in the Module 4 Model Card
(Acceptance Criteria and Monitoring) and the Module 5 dashboard's Fairness
& Ethics tab, now implemented as runnable checks rather than only stated
intentions. The technical implementation is `monitoring/run_drift_check.py`,
verified working against the project's real train/test data (Section 8 of
the Final Report).

## 1. Deployment Plan

1. **Pilot**: deploy the model behind the FastAPI `/predict` endpoint to a
   limited campaign segment (per the Module 5 "Next Steps" slide).
2. **Validate**: compare actual campaign conversion against predicted
   probability for the pilot segment.
3. **Scale**: if acceptance criteria (Section 3, below) hold for one full
   campaign cycle, extend to the full retail base.

## 2. Performance Tracking KPIs

| KPI | Target | Source |
|---|---|---|
| Cross-sell conversion uplift | +15% within 2 quarters | Module 1 goal |
| Model ROC-AUC (rolling 90-day) | ≥ 0.70 | Model Card acceptance criteria |
| Disparate impact ratio (gender, age band) | ≥ 0.80 | Fairlearn, re-checked monthly |
| Data drift (share of features drifted) | < 30% of monitored columns | Evidently AI, `run_drift_check.py` |

## 3. Continuous Fairness Monitoring Plan

`monitoring/run_drift_check.py` re-runs the Module 4 disparate impact
calculation against the current scoring population every time it executes
(scheduled weekly via `.github/workflows/ci-cd.yml`'s cron trigger, or
on-demand in the `monitoring` Docker Compose service). If either
protected attribute's disparate impact ratio drops below 0.80, the script
exits non-zero, which the CI/CD job resolves to a failed build (Section
8), and the escalation path in `governance_structure.md` takes over.

## 4. Model Drift and Bias Drift Detection

- **Data drift**: Evidently AI's `DataDriftPreset` compares the current
  scoring population's feature distributions against the training
  reference. Verified working (Section 8): 0 of 17 columns drifted on the
  actual train/test split, confirming a clean baseline.
- **Bias drift**: distinct from data drift — this checks whether
  fairness, not just distribution, has moved. Handled by the fairness
  check in Section 3, run on the same schedule.
- **Threshold**: more than 30% of monitored columns drifting, or either
  protected attribute's disparate impact ratio falling below 0.80,
  triggers an alert.

## 5. Incident Response Plan (Ethical Failures)

| Step | Action | Owner |
|---|---|---|
| 1. Detect | Automated alert from `run_drift_check.py` or a manual report (e.g., a customer complaint routed through the DPO) | Monitoring system / any staff member |
| 2. Contain | Model's campaign use is paused by the AI & Data Ethics Committee (standing authority, `governance_structure.md`) | AI & Data Ethics Committee |
| 3. Investigate | Root cause traced using the Explanation Store (`monitoring/explanation_store.py`) — every prediction has a logged SHAP explanation, so a specific decision can be reconstructed | Data Scientist |
| 4. Remediate | Retrain, re-mitigate, or roll back to the previous MLflow model version, whichever the investigation supports | Chief Data and Analytics Officer |
| 5. Report | Document the incident and resolution in the Ethical Risk Register (Module 1), updated | Data Protection Officer |

## 6. Decommissioning Plan

A model version is retired when: (a) a newer MLflow-registered version
meets acceptance criteria and has passed one full campaign cycle, or (b)
an unresolved fairness or performance failure makes continued use
untenable. Decommissioning steps:

1. Change the MLflow Model Registry alias from `champion` to `archived`
   for the retiring version — never delete it, to preserve the audit
   trail required by the Explanation Store's contestability commitment.
2. Route all `/predict` traffic to the new champion version.
3. Retain the retired version's explanation logs and fairness reports for
   the same one-year minimum retention period set in the Module 3 Data
   Governance Framework, since a customer could still contest a decision
   made under the old version within that window.
4. Notify the AI & Data Ethics Committee and update the Model Card.
