"""
generate_raw_data.py
Generates a synthetic proxy dataset for the FirstBank Nigeria Limited customer
segmentation project (BAN6800, Modules 1-3).

Per the Module 3 assignment brief: "you may supplement it with synthetic data
containing noise to match the characteristics of the original dataset" when the
identified real dataset (Kaggle: Bank Customer Segmentation, Bansal, 2021) cannot
be pulled into this environment. This script reproduces that dataset's schema
(customer_id, age, gender, location, account_balance, txn_date, txn_amount,
txn_category) as defined in the Module 2 Data Dictionary, at the same order of
magnitude, and deliberately injects the kinds of data quality problems a real
core-banking extract would contain: missing values, duplicate records, invalid
categories, and outliers. This messiness is what the pipeline in this module is
built to detect and correct.
"""
import numpy as np
import pandas as pd
from faker import Faker
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_CUSTOMERS = 6000
TXN_PER_CUSTOMER_MEAN = 6

NIGERIAN_STATES = [
    "Lagos", "Kano", "Rivers", "Oyo", "Kaduna", "Ogun", "Enugu", "Delta",
    "Anambra", "Edo", "Plateau", "Abuja FCT", "Cross River", "Imo", "Kwara",
]
GENDERS = ["Male", "Female"]
TXN_CATEGORIES = [
    "POS Purchase", "ATM Withdrawal", "Airtime/Data", "Bill Payment",
    "Transfer", "Salary Credit", "Loan Repayment", "Savings Deposit",
]
# a few "dirty" category variants that a real core-banking export would contain
DIRTY_CATEGORY_VARIANTS = ["pos purchase", "ATM-Withdrawal", "TRANSFER", "n/a", ""]

def make_customers(n):
    rows = []
    for i in range(n):
        cust_id = f"FB{100000 + i}"
        age = int(np.clip(np.random.normal(38, 13), 18, 85))
        gender = random.choice(GENDERS)
        location = random.choice(NIGERIAN_STATES)
        base_balance = float(np.clip(np.random.lognormal(mean=10.5, sigma=1.1), 500, 5_000_000))
        rows.append((cust_id, age, gender, location, round(base_balance, 2)))
    return pd.DataFrame(rows, columns=["customer_id", "age", "gender", "location", "account_balance"])

def make_transactions(customers_df):
    rows = []
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-12-31")
    for _, cust in customers_df.iterrows():
        n_txn = max(1, int(np.random.poisson(TXN_PER_CUSTOMER_MEAN)))
        for _ in range(n_txn):
            txn_date = start + (end - start) * random.random()
            category = random.choice(TXN_CATEGORIES)
            amount = float(np.clip(np.random.lognormal(mean=8.5, sigma=1.3), 100, 2_000_000))
            rows.append((cust["customer_id"], txn_date.date().isoformat(), round(amount, 2), category))
    return pd.DataFrame(rows, columns=["customer_id", "txn_date", "txn_amount", "txn_category"])

def inject_messiness(customers_df, txns_df):
    customers_df = customers_df.copy()
    txns_df = txns_df.copy()

    # 1. Missing values (2-4% across a few fields), simulating incomplete KYC records
    for col, frac in [("age", 0.03), ("gender", 0.02), ("location", 0.025), ("account_balance", 0.015)]:
        idx = customers_df.sample(frac=frac, random_state=SEED).index
        customers_df.loc[idx, col] = np.nan

    idx = txns_df.sample(frac=0.02, random_state=SEED).index
    txns_df.loc[idx, "txn_amount"] = np.nan

    # 2. Duplicate records (~1.5% of customers double-registered, common after branch merges)
    dupes = customers_df.sample(frac=0.015, random_state=SEED + 1)
    customers_df = pd.concat([customers_df, dupes], ignore_index=True)

    dupe_txns = txns_df.sample(frac=0.01, random_state=SEED + 1)
    txns_df = pd.concat([txns_df, dupe_txns], ignore_index=True)

    # 3. Inconsistent / dirty category labels
    dirty_idx = txns_df.sample(frac=0.04, random_state=SEED + 2).index
    txns_df.loc[dirty_idx, "txn_category"] = [random.choice(DIRTY_CATEGORY_VARIANTS) for _ in dirty_idx]

    # 4. Outliers / implausible values (data entry errors)
    out_idx = txns_df.sample(frac=0.005, random_state=SEED + 3).index
    txns_df.loc[out_idx, "txn_amount"] = txns_df.loc[out_idx, "txn_amount"] * 1000

    neg_idx = customers_df.sample(frac=0.003, random_state=SEED + 4).index
    customers_df.loc[neg_idx, "age"] = -1  # invalid age, simulating a bad data entry

    # 5. Shuffle row order so it isn't suspiciously sorted
    customers_df = customers_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    txns_df = txns_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return customers_df, txns_df

if __name__ == "__main__":
    customers = make_customers(N_CUSTOMERS)
    txns = make_transactions(customers)
    customers, txns = inject_messiness(customers, txns)

    customers.to_csv("/home/claude/m3/repo/data/raw/customers_raw.csv", index=False)
    txns.to_csv("/home/claude/m3/repo/data/raw/transactions_raw.csv", index=False)

    print(f"customers_raw.csv: {len(customers)} rows")
    print(f"transactions_raw.csv: {len(txns)} rows")
    print(f"Total raw records: {len(customers) + len(txns)}")
