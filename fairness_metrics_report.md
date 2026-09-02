# Fairness Metrics Report
FirstBank Cross-Sell Propensity Model — BAN6800 Module 4

Fairness objectives and protected attributes carried forward from the
Module 2 Project Overview Report (Section 6.2) and the Module 1 Vision
Document's Fairness Objectives.

## Protected Attributes Evaluated
- **Gender** (Male, Female, Unspecified)
- **Age band** (60+ vs Under 60), per the Module 3 bias detection thresholds

## Metrics (test set, n=1,200)

| Metric | Gender | Age Band (60+ vs Under 60) |
|---|---|---|
| Demographic Parity Difference | 0.110 | 0.065 |
| Demographic Parity Ratio (Disparate Impact Ratio) | 0.00 | 0.385 |
| Equalized Odds Difference | 0.207 | 0.123 |

**Four-fifths rule check**: both attributes fall below the 0.80 disparate
impact ratio threshold before mitigation — flagged for corrective action.

## Root Cause

The disparity is not a modeling artifact; it traces to the historical
under-targeting pattern documented in the label-generation logic (Module 4
report, Section 1): older customers and customers with unspecified gender
were, by design, historically under-represented in positive outcomes,
mirroring a realistic risk named in the Module 1 Ethical AI Charter. The
model learned that pattern from the (synthetic) training data, which is
exactly why this audit exists.

## Mitigation

**Method**: Fairlearn `ThresholdOptimizer` (post-processing), constrained to
demographic parity, applying group-specific decision thresholds per
protected attribute.

| Metric | Age Band — Before | Age Band — After | Gender — Before | Gender — After |
|---|---|---|---|---|
| Disparate Impact Ratio | 0.385 | **0.828** | 0.00 | **0.966** |
| Equalized Odds Difference | 0.123 | 0.074 | 0.207 | 0.041 |
| Accuracy | 0.716 | 0.723 | 0.716 | 0.719 |

Both attributes clear the 0.80 threshold post-mitigation, with accuracy
essentially unchanged (age: +0.7pp, gender: +0.3pp) — the mitigation did not
trade meaningful predictive performance for fairness.

## Recommendation

Deploy with the ThresholdOptimizer post-processing step applied at
inference time, and monitor disparate impact ratio monthly (see Model Card,
Monitoring Signals) since group-specific thresholds can drift out of
compliance as the customer population shifts.
