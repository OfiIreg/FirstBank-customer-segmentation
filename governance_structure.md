# Ethical Governance Structure
FirstBank Nigeria Limited — Cross-Sell Propensity Analytics Project

Formalizes the Ethics Oversight Body first named in the Module 1 Vision
Document into an ongoing operating structure, now that there is a live
system to govern.

## Roles and Authority

| Role | Held By (Module 1) | Ongoing Responsibility |
|---|---|---|
| Executive Sponsor | Head of Retail Banking | Owns the business outcome; approves scope changes |
| Business Owner | Chief Marketing Officer | Approves campaign use of model output |
| Technical Sponsor | Chief Data and Analytics Officer | Owns model quality, retraining cadence, MLflow registry |
| Compliance Authority | Data Protection Officer | Sign-off on every release; owns the Regulatory Compliance Report |
| Oversight Body | AI & Data Ethics Committee (chaired jointly by CDAO and DPO) | Reviews fairness results before each release; **can pause deployment** |

## Review Cadence

- **Per release**: Data Protection Officer sign-off (unchanged from Module 2).
- **Monthly**: fairness and drift monitoring review (Section 7 of the
  Final Report) — a standing agenda item, not an ad hoc check.
- **Quarterly**: full Ethical AI Framework re-read by the AI & Data Ethics
  Committee, confirming the framework still matches what the system
  actually does.

## Escalation Path

A finding from the monitoring system (Section 7.2) that crosses an alert
threshold routes automatically to the Data Protection Officer, who
convenes the AI & Data Ethics Committee within five business days. The
Committee has standing authority to suspend the model's use in active
campaigns pending review — this is not a recommendation, it is the same
authority granted in the Module 1 Vision Document, now exercised against
a live system rather than a planned one.

## What Changed From Module 1 to Now

Module 1 wrote this authority into a plan. The Final Project is where it
becomes real: the Incident Response Plan (Section 7.3 of the Final
Report) gives this committee an actual runbook to follow, rather than a
one-line commitment to "pause deployment pending review."
