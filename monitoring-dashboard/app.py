"""
app.py
FirstBank Monitoring Dashboard — Final Project, Section 7.

Displays the output of monitoring/run_drift_check.py: data drift status
and fairness drift status for the cross-sell propensity model, so the
AI & Data Ethics Committee and Chief Data and Analytics Officer named in
the Module 1 governance structure have a live view rather than only a
JSON file in a repo.

Run locally with: streamlit run app.py
"""
import streamlit as st
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="FirstBank Monitoring", page_icon="\U0001F4CA", layout="wide")

NAVY = "#1F3864"
GOLD = "#C9A227"
GREEN = "#1A7F37"
RED = "#C0392B"
GREY = "#595959"

HERE = Path(__file__).parent

with open(HERE / "monitoring_result.json") as f:
    monitoring = json.load(f)
with open(HERE / "fairness_results.json") as f:
    fairness = json.load(f)

st.markdown(f"""
<div style="background-color:{NAVY}; padding:22px 28px; border-radius:8px; margin-bottom:20px;">
  <div style="color:{GOLD}; font-size:13px; font-weight:700; letter-spacing:1.5px;">FIRSTBANK NIGERIA LIMITED</div>
  <div style="color:white; font-size:26px; font-weight:700; margin-top:4px;">Model Monitoring Dashboard</div>
  <div style="color:#B7C3DC; font-size:13.5px; margin-top:4px;">Final Project &nbsp;|&nbsp; Continuous Fairness & Drift Monitoring &nbsp;|&nbsp; Cross-Sell Propensity Model</div>
</div>
""", unsafe_allow_html=True)

ts = datetime.fromisoformat(monitoring["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Last check: {ts}  \u2014  produced by monitoring/run_drift_check.py, scheduled weekly via GitHub Actions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Drift")
    dd = monitoring["data_drift"]
    if dd["status"] == "OK":
        st.success(f"OK \u2014 {dd['columns_drifted']} of {dd['columns_checked']} monitored columns drifted "
                   f"({dd['share_drifted']:.0%})")
    else:
        st.error(f"ALERT \u2014 {dd['columns_drifted']} of {dd['columns_checked']} monitored columns drifted "
                 f"({dd['share_drifted']:.0%})")
    st.caption("Compares the current scoring population's feature distributions against the training reference "
               "using Evidently AI. Flags if more than 30% of columns drift.")

with col2:
    st.subheader("Fairness Drift")
    fd = monitoring["fairness_drift"]
    if fd["status"] == "OK":
        st.success("OK \u2014 both protected attributes remain above the 0.80 fairness threshold")
    else:
        st.warning("ALERT \u2014 re-check required (see details below)")
        for a in fd["alerts"]:
            st.write(f"\u26a0\ufe0f {a}")
    st.caption("Re-checks Demographic Parity / Disparate Impact Ratio for gender and age band. "
               "This alert does not block deployment automatically \u2014 it routes to the AI & Data "
               "Ethics Committee per the Incident Response Plan (docs/monitoring_plan.md).")

st.divider()
st.subheader("Current Fairness Snapshot")

c1, c2 = st.columns(2)
for col, attr, label in [(c1, "age_band", "Age Band (60+)"), (c2, "gender", "Gender")]:
    with col:
        mit = fairness.get("mitigation", {}).get(attr, {})
        before = mit.get("disparate_impact_ratio_before", 0)
        after = mit.get("disparate_impact_ratio_after", 0)
        st.metric(f"{label} \u2014 Disparate Impact Ratio", f"{after:.2f}",
                  delta=f"+{after - before:.2f} vs pre-mitigation", delta_color="normal")
        st.progress(min(after, 1.0))
        status = "\u2705 Above 0.80 threshold" if after >= 0.80 else "\u26a0\ufe0f Below 0.80 threshold"
        st.caption(status)

st.info(
    "**Honest status:** Fairlearn's ThresholdOptimizer mitigation substantially narrowed both gaps "
    "(gender: 0.00\u21920.79, age band: 0.39\u21920.75) but neither fully cleared the 0.80 four-fifths "
    "threshold. This is documented transparently in the Final Report, Section 8, per the assignment's "
    "explicit requirement to document where mitigation fell short rather than only where it succeeded. "
    "A follow-up mitigation pass is a stated precondition before this model scores customers beyond "
    "the pilot segment."
)

st.divider()
st.caption("FirstBank Nigeria Limited \u2014 BAN6800 Final Project. This dashboard reflects the most recent "
           "scheduled monitoring run; see the GitHub Actions tab for live run history.")
