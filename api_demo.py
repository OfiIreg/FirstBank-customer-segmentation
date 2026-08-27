"""
api_demo.py
API Demo (Module 5 technical requirement): demonstrates the Module 4
FastAPI /predict endpoint in action against several realistic customer
profiles, printing plain-language interpretations alongside the raw
response, so a stakeholder (not just a developer) can follow along.

Usage:
    1. In one terminal, start the API:
         cd ../../Module4/src/api && uvicorn main:app --port 8000
    2. In another terminal, run this demo:
         python api_demo.py
"""
import requests
import json

API_URL = "http://localhost:8000/predict"

CUSTOMERS = [
    {
        "label": "High-engagement customer (frequent, recent, high balance)",
        "payload": {
            "age": 38, "account_balance": 250000, "recency_days": 14, "frequency": 8,
            "monetary_total": 180000, "monetary_avg": 22500,
            "share_ATM Withdrawal": 0.1, "share_Airtime/Data": 0.05, "share_Bill Payment": 0.1,
            "share_Loan Repayment": 0.0, "share_Other/Unclassified": 0.0, "share_POS Purchase": 0.3,
            "share_Salary Credit": 0.2, "share_Savings Deposit": 0.2, "share_Transfer": 0.05,
            "location": "Lagos",
        },
    },
    {
        "label": "Low-engagement customer (infrequent, dormant, low balance)",
        "payload": {
            "age": 68, "account_balance": 15000, "recency_days": 120, "frequency": 1,
            "monetary_total": 8000, "monetary_avg": 8000,
            "share_ATM Withdrawal": 0.5, "share_Airtime/Data": 0.0, "share_Bill Payment": 0.0,
            "share_Loan Repayment": 0.0, "share_Other/Unclassified": 0.0, "share_POS Purchase": 0.0,
            "share_Salary Credit": 0.0, "share_Savings Deposit": 0.5, "share_Transfer": 0.0,
            "location": "Kano",
        },
    },
    {
        "label": "Moderate-engagement customer (borderline)",
        "payload": {
            "age": 34, "account_balance": 90000, "recency_days": 40, "frequency": 4,
            "monetary_total": 60000, "monetary_avg": 15000,
            "share_ATM Withdrawal": 0.2, "share_Airtime/Data": 0.1, "share_Bill Payment": 0.15,
            "share_Loan Repayment": 0.0, "share_Other/Unclassified": 0.0, "share_POS Purchase": 0.2,
            "share_Salary Credit": 0.15, "share_Savings Deposit": 0.1, "share_Transfer": 0.1,
            "location": "Rivers",
        },
    },
]


def explain(prob, label):
    if label == 1:
        return f"-> Model recommends TARGETING this customer for the cross-sell campaign (P={prob:.0%})."
    return f"-> Model does NOT recommend targeting this customer right now (P={prob:.0%})."


if __name__ == "__main__":
    print("FirstBank Cross-Sell Propensity API Demo\n" + "=" * 50)
    for c in CUSTOMERS:
        print(f"\n{c['label']}")
        try:
            resp = requests.post(API_URL, json=c["payload"], timeout=5)
            resp.raise_for_status()
            result = resp.json()
            print("Request:", json.dumps(c["payload"], indent=None)[:100] + "...")
            print("Response:", json.dumps(result, indent=2))
            print(explain(result["predicted_conversion_probability"], result["predicted_label"]))
        except requests.exceptions.ConnectionError:
            print("Could not connect to the API. Start it first with:")
            print("  cd ../../Module4/src/api && uvicorn main:app --port 8000")
            break
