# Ethical AI Framework
FirstBank Nigeria Limited — Cross-Sell Propensity Analytics Project
BAN6800 Final Project

This framework synthesizes the ethical commitments made across all five
prior modules into one governing document. It does not introduce new
principles; it reconciles and consolidates what was already committed to,
so there is a single place a regulator, auditor, or new team member can
read to understand FirstBank's ethical posture on this system.

## 1. Foundational Principles (Module 1)

The Module 1 Vision Document committed this project to three principles,
carried through every subsequent module without exception:

- **Fairness** — segments and predictions are evaluated for disparate
  impact before any commercial action is taken on them.
- **Transparency** — model logic must be explainable in business terms to
  relationship managers and, on request, to customers.
- **Accountability** — every automated decision affecting a customer's
  product access has a named human owner and an audit trail.

## 2. How Each Module Operationalized These Principles

| Module | Operationalization |
|---|---|
| 1 | Defined the Ethical AI Charter, Fairness Objectives, and Ethical Risk Register |
| 2 | Translated principles into a Data Privacy Plan, Fairness Metrics definitions (Demographic Parity, Equalized Odds), and a Regulatory Compliance Checklist |
| 3 | Built a Bias Detection Suite, a Data Anonymization Plan, and Privacy Audit Logging into the pipeline itself, not bolted on after |
| 4 | Measured Demographic Parity, Equalized Odds, and Disparate Impact Ratio against the trained model; applied Fairlearn ThresholdOptimizer mitigation where the four-fifths rule was failed; published a Model Card |
| 5 | Translated fairness findings into stakeholder-accessible language via the Ethical Compliance Dashboard; published a Transparency Statement |
| Final | Adds continuous monitoring (Section 7), an Explanation Store for individual-decision audit, and this consolidated framework |

## 3. What Was Found and Corrected

The fairness audit (Module 4) found the model's synthetic training label
carried a documented historical bias pattern affecting two groups:
customers aged 60+ (disparate impact ratio 0.385 pre-mitigation) and
customers recorded with unspecified gender (0.00 pre-mitigation). Both
were corrected via Fairlearn's ThresholdOptimizer to 0.828 and 0.966
respectively, clearing the 0.80 four-fifths threshold, with accuracy
essentially unchanged. This is documented, not hidden, in the Model Card,
the Fairness Metrics Report, and Slide 8 of the Module 5 presentation.

## 4. Where This Framework Falls Short (Documented Honestly)

Per the assignment's explicit instruction to document where mitigation
fell short: the correction was applied post-hoc, at the decision-threshold
level, not by removing the bias at its source (the synthetic label). A
real-data retrain (Section 7.4) is required before this model is used on
live customers, and the mitigation must be re-validated against real
labels, since a synthetic-label bias pattern and a real-world bias pattern
are not guaranteed to look identical.

## 5. Governance Structure

See `governance_structure.md` for the full ongoing oversight structure;
in summary, the AI & Data Ethics Committee (Module 1) retains authority to
pause any deployment pending review, and the Data Protection Officer signs
off on every release per the Module 2 stakeholder communication plan.
