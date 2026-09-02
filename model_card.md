# Model Card: FirstBank Cross-Sell Propensity Model

BAN6800, Module 4 | Champion model registered in MLflow as
`firstbank-cross-sell-propensity-model`, alias `champion`, version 1

## Model Details
- **Model type**: Logistic Regression (L2-regularized), tuned via 5-fold
  stratified cross-validated randomized search
- **Selected because**: best test ROC-AUC (0.702) among three candidates
  (Logistic Regression, Random Forest, Gradient Boosting) and the most
  interpretable, which matters directly for the explainability requirements
  in this module and the transparency commitment in the Module 1 Ethical AI
  Charter
- **Training framework**: scikit-learn 1.8, tracked in MLflow
- **Input**: 31 features (15 numeric transaction/demographic features,
  16 one-hot encoded location indicators); gender is intentionally excluded
  as a model input (see Fairness section)
- **Output**: probability of cross-sell conversion in the next campaign
  cycle; decision threshold 0.5 by default

## Intended Use
- **Primary use**: rank FirstBank retail customers by propensity to convert
  on a targeted cross-sell offer, to focus marketing spend and relationship
  manager attention, per the Module 1 Vision Document's goal of a 15%
  cross-sell conversion uplift.
- **Out of scope**: this model must not be used for credit decisions, pricing,
  or any decision that changes a customer's access to a product or service.
  It is a marketing prioritization tool only.

## Training Data
- Module 3's `customer_analytics_table.csv` (6,000 customers, synthetic
  proxy for FirstBank's core banking data), with a synthesized
  `cross_sell_conversion` label (see Module 4 report, Section 1, for the
  documented label-generation rule and why it was necessary).
- 80/20 stratified train/test split (4,800 / 1,200), 5-fold CV for tuning.

## Performance (held-out test set, n=1,200)
| Metric | Baseline | Logistic Regression (champion) |
|---|---|---|
| Accuracy | 0.696 | 0.716 |
| Precision | 0.000 | 0.597 |
| Recall | 0.000 | 0.203 |
| F1 | 0.000 | 0.303 |
| ROC-AUC | 0.500 | 0.702 |

## Explainability
- **Global**: SHAP identifies `frequency`, `recency_days`, `monetary_total`,
  and `account_balance` as the dominant drivers, consistent with the
  business logic used to define the label.
- **Local**: SHAP waterfall and LIME explanations agree directionally for
  individual predictions (see report Section 6).
- **Counterfactual**: for a sampled non-converter, a ~90% increase in
  transaction frequency alone was sufficient to flip the predicted decision.

## Fairness
- **Protected attributes evaluated**: gender, age (age band: 60+ vs under 60).
- **Pre-mitigation**: disparate impact ratio 0.00 (gender), 0.39 (age band) —
  both fail the 0.80 four-fifths rule.
- **Mitigation applied**: Fairlearn `ThresholdOptimizer` (post-processing,
  demographic parity constraint, group-specific thresholds).
- **Post-mitigation**: disparate impact ratio 0.97 (gender), 0.83 (age band)
  — both clear the 0.80 threshold, with accuracy slightly improved
  (0.716 → 0.726) and equalized-odds difference reduced in both cases.
- The bias originates in the synthesized label (documented in the training
  data description) and mirrors the real risk named in the Module 1 Ethical
  AI Charter: historically under-targeted groups being further
  under-targeted by a naive model. This model card documents that risk
  rather than hiding it.

## Limitations
- Trained on a synthetic proxy dataset, not FirstBank's real core banking
  data; absolute performance numbers will not transfer directly and the
  model must be retrained once real, labeled campaign-response data is
  available.
- Recall is modest (0.20 pre-mitigation), meaning the model misses many
  true converters; acceptable for a "focus marketing spend" use case but
  not for a use case requiring high recall.
- The label itself is synthetic; fairness results describe how the model
  responds to the injected pattern, not a finding about real FirstBank
  customers.

## Operational Risks (from the Module 1 Risk Register)
- Schema drift if the real core banking extract differs from the proxy
  schema (flagged in Module 1's RAID log).
- Model staleness: retrain on a monthly cadence to match the Module 1
  segment-refresh goal.

## Monitoring Signals Post-Deployment
- Track selection rate and disparate impact ratio by gender and age band
  monthly; alert if disparate impact ratio drops below 0.80 again.
- Track prediction drift (mean predicted probability) against the training
  baseline; investigate if it moves more than 5 percentage points.
- Track actual campaign conversion rate against predicted rate to catch
  model decay early.

## Acceptance Criteria for Promotion to Production
- ROC-AUC ≥ 0.70 on a rolling 90-day holdout.
- Disparate impact ratio ≥ 0.80 for both gender and age band, post-mitigation.
- No unexplained decision flips above 5% under the sensitivity test in
  Module 4 Section 7.
