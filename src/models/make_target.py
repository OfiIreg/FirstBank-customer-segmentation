"""
make_target.py
Business Need & Target Definition (Module 4, Section 1).

The Module 3 pipeline produced a customer-level analytics table but no
outcome label, because the Kaggle proxy dataset (and the FirstBank use case
it stands in for) is transactional, not campaign-response data. Module 4
requires a supervised predictive task, so this script defines one directly
from the Module 1/2 business problem: predicting which customers will
convert on a targeted cross-sell offer, the same "propensity scoring"
task named in the Module 1 solution overview and Module 2 analytical
methods section.

Because no real campaign-response history exists for this proxy dataset,
the label is synthesized from a plausible, documented business rule
(engaged, higher-balance customers are more likely to convert, with
substantial noise so the task is not trivially separable) rather than
left unlabeled. This keeps the modeling problem realistic and gives the
fairness analysis in Section 6 something genuine to detect: the rule
deliberately includes a mild, undisclosed-to-the-model dependency on
gender and age, standing in for the kind of historical bias a real
campaign-response dataset often contains, so that the bias detection and
mitigation steps in this module have real signal to find and correct.

Target: cross_sell_conversion (1 = converted on the last targeted offer, 0 = did not)
Unit of analysis: one row per customer
Prediction horizon: next campaign cycle (the Module 1 goal is a 15% uplift
within two quarters of deployment)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

IN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "customer_analytics_table.csv"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "customer_model_dataset.csv"


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardized engagement signal (z-scores), the legitimate business driver
    freq_z = (df["frequency"] - df["frequency"].mean()) / df["frequency"].std()
    mon_z = (df["monetary_total"] - df["monetary_total"].mean()) / df["monetary_total"].std()
    bal_z = (df["account_balance"] - df["account_balance"].mean()) / df["account_balance"].std()
    recency_z = (df["recency_days"] - df["recency_days"].mean()) / df["recency_days"].std()

    # Legitimate propensity signal: more engaged, higher-balance, more recently
    # active customers are more likely to respond to a cross-sell offer
    logit = 0.55 * freq_z + 0.45 * mon_z + 0.35 * bal_z - 0.30 * recency_z

    # Deliberately injected historical bias (documented, not hidden from the
    # analysis): older customers and customers recorded as "Unspecified"
    # gender were historically under-targeted by relationship managers, so
    # their observed conversion is suppressed independent of true engagement.
    # This mirrors the real risk named in the Module 1 Ethical AI Vision and
    # gives Section 6's fairness audit a genuine signal to catch.
    age_penalty = np.where(df["age"] >= 60, -0.45, 0.0)
    gender_penalty = np.where(df["gender"] == "Unspecified", -0.35, 0.0)
    logit = logit + age_penalty + gender_penalty

    noise = np.random.normal(0, 1.15, size=len(df))  # substantial noise: not trivially separable
    prob = 1 / (1 + np.exp(-(logit + noise - 1.1)))  # intercept sets base rate near ~18%
    conversion = np.random.binomial(1, prob)

    df["cross_sell_conversion"] = conversion
    return df


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = make_target(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    rate = df["cross_sell_conversion"].mean()
    print(f"Modeling dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Base conversion rate: {rate:.1%}")
    print(f"Conversion rate by gender:\n{df.groupby('gender')['cross_sell_conversion'].mean()}")
    print(f"Conversion rate, age 60+ vs under 60: "
          f"{df[df['age']>=60]['cross_sell_conversion'].mean():.1%} vs "
          f"{df[df['age']<60]['cross_sell_conversion'].mean():.1%}")
