# FirstBank Cross-Sell Insights Dashboard

BAN6800 Module 5 — Streamlit dashboard for stakeholder exploration of the
Module 4 cross-sell propensity model.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (7 tabs: Overview, Key Insights, Drivers, Example Customers, What-If Explorer, Fairness & Ethics, Limitations) |
| `model_bundle.joblib` | The serialized champion model + preprocessor from Module 4 |
| `example_predictions.csv` | 5 curated, anonymized example customers |
| `sample_customers.csv` | 150-row sample of the test set |
| `fairness_results.json` | Module 4 fairness audit + mitigation results |
| `model_results.json` | Module 4 model performance metrics |
| `shap_feature_importance.csv` | Module 4 SHAP global feature ranking |
| `api_demo.py` | Standalone script demonstrating the Module 4 FastAPI `https://firstbank-cross-sell-api.onrender.com)` endpoint |
| `requirements.txt` | Minimal dependencies for the dashboard only |

This folder was verified locally: `streamlit run app.py` starts cleanly
and responds with HTTP 200 (confirmed via `/_stcore/health` and a direct
request to `/`).

## Deploy to Streamlit Community Cloud (free, ~5 minutes)

1. Push this `dashboard/` folder to your GitHub repository (the same repo
   from Module 2, e.g. under a `dashboards/segment-explorer/` or
   `module5/dashboard/` path — update the path in step 3 to match).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **New app**, select your repository and branch, and set the
   **Main file path** to `dashboard/app.py` (or wherever you placed it).
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL like
   `https://your-app-name.streamlit.app`.
5. Put that URL on the title slide and the "Live Dashboard" slide of the
   presentation, and in your GitHub repo's README.

## Deploy to Hugging Face Spaces (alternative)

1. Create a new Space at https://huggingface.co/new-space, choosing the
   **Streamlit** SDK.
2. Upload this folder's contents (or `git push` to the Space's own git
   remote, which HF gives you on creation).
3. The Space builds automatically and gives you a URL like
   `https://huggingface.co/spaces/your-username/firstbank-dashboard`.

## Run locally first (recommended before deploying)

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.
