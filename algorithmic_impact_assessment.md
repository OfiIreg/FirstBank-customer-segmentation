# Algorithmic Impact Assessment
FirstBank Nigeria Limited — Cross-Sell Propensity Model
BAN6800 Final Project

## 1. System Description

A logistic regression model that scores retail customers with a
probability of converting on a targeted cross-sell offer, used to
prioritize marketing outreach. Registered in MLflow as
`firstbank-cross-sell-propensity-model`, version 1, alias `champion`.

## 2. Who Is Affected

- **Directly**: FirstBank retail customers who may or may not receive a
  targeted marketing offer based on their score.
- **Indirectly**: relationship managers whose targeting workflow changes;
  marketing budget allocation decisions.

## 3. Severity and Reversibility of Impact

| Dimension | Assessment |
|---|---|
| Decision type | Marketing prioritization only — explicitly not credit, pricing, or account access (Model Card, Transparency Statement) |
| Reversibility | Fully reversible — a customer not targeted this cycle can be targeted next cycle; no permanent consequence |
| Severity if wrong | Low-to-moderate — a missed cross-sell opportunity or an unwanted marketing contact, not a denial of service |
| Scale | Up to FirstBank's full 42 million retail account base if scaled beyond the pilot |

Given full reversibility and marketing-only scope, this system sits at
**low-to-moderate impact tier** on a standard algorithmic impact scale —
still warranting the fairness audit and monitoring performed, but not the
heightened scrutiny a credit or hiring decision would require.

## 4. Risk of Disparate Impact

Assessed and found present pre-mitigation (Module 4): customers aged 60+
and customers with unspecified gender were under-scored relative to their
true engagement. This risk was **material enough to require mitigation**,
which was applied and verified (Section 6.5 of the Module 4 report).
Residual risk: the bias pattern was identified in a synthetic label: the
same audit must be re-run against real campaign-response data (Section 7.4
of this report) before the finding can be considered conclusive for real
customers.

## 5. Human Oversight

Per Module 1's Ethics Oversight Body, the AI & Data Ethics Committee
reviews fairness results before each major release and can pause
deployment. No prediction from this model triggers an automated customer-
facing action without a relationship manager or marketing campaign
manager choosing to act on it — the model ranks, it does not decide.

## 6. Assessment Outcome

**Proceed to limited pilot**, conditional on: (a) the acceptance criteria
in the Model Card being met on a rolling basis, (b) monthly fairness
monitoring being active before wider rollout (Section 7), and (c) a
retrain against real labeled data before any customer beyond the pilot
segment is scored.
