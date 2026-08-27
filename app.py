"""
app.py
FirstBank Cross-Sell Propensity Dashboard — BAN6800 Module 5

An interactive, stakeholder-facing dashboard translating the Module 4
predictive model into business-usable insights: plain-language model
explanation, what-if analysis, example predictions, and an ethical
compliance view of the fairness audit and mitigation from Module 4.

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by connecting this folder's GitHub
repo path and pointing the app file to dashboard/app.py.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="FirstBank Cross-Sell Insights",
    page_icon="\U0001F3E6",
    layout="wide",
)

NAVY = "#1F3864"
GOLD = "#C9A227"
GREEN = "#1A7F37"
RED = "#C0392B"
GREY = "#595959"

HERE = Path(__file__).parent


@st.cache_resource
def load_bundle():
    return joblib.load(HERE / "model_bundle.joblib")


@st.cache_data
def load_data():
    examples = pd.read_csv(HERE / "example_predictions.csv")
    sample = pd.read_csv(HERE / "sample_customers.csv")
    with open(HERE / "fairness_results.json") as f:
        fairness = json.load(f)
    with open(HERE / "model_results.json") as f:
        model_results = json.load(f)
    shap_importance = pd.read_csv(HERE / "shap_feature_importance.csv")
    return examples, sample, fairness, model_results, shap_importance


bundle = load_bundle()
examples, sample, fairness, model_results, shap_importance = load_data()

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style="background-color:{NAVY}; padding:22px 28px; border-radius:8px; margin-bottom:6px;">
  <div style="color:{GOLD}; font-size:13px; font-weight:700; letter-spacing:1.5px;">FIRSTBANK NIGERIA LIMITED</div>
  <div style="color:white; font-size:26px; font-weight:700; margin-top:4px;">Cross-Sell Propensity Insights Dashboard</div>
  <div style="color:#B7C3DC; font-size:13.5px; margin-top:4px;">BAN6800 Module 5 &nbsp;|&nbsp; For marketing and retail banking stakeholders &nbsp;|&nbsp; Built on the Module 4 model</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Overview", "Key Insights", "Who Converts? (What Drives Predictions)",
    "Example Customers", "What-If Explorer", "Fairness & Ethics", "Limitations & Next Steps",
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("The Business Problem")
        st.write(
            "FirstBank serves over 42 million retail customer accounts, but relationship "
            "managers and marketing teams currently target cross-sell offers using broad "
            "demographic guesses (age, branch location) instead of how customers actually "
            "bank. This leads to generic offers, wasted marketing spend, and missed "
            "opportunities to deepen relationships with the right customers."
        )
        st.subheader("What This Model Does")
        st.write(
            "The model scores every customer with a probability of responding to a "
            "targeted cross-sell offer, based on how they actually use their account "
            "(transaction frequency, recent activity, balances) rather than who they are "
            "demographically. Marketing can then focus offers on customers most likely to convert."
        )
        st.subheader("Who This Dashboard Is For")
        st.write(
            "Marketing managers deciding which customers to target this campaign cycle; "
            "retail banking leadership tracking progress toward the conversion uplift "
            "goal; and the Data Protection Officer and AI & Data Ethics Committee "
            "reviewing fairness before any campaign launches."
        )
    with col2:
        st.metric("Customers Scored (test set)", "1,200")
        st.metric("Baseline Conversion Rate", "30.4%")
        st.metric("Model Discrimination (ROC-AUC)", "0.70", help="0.50 = no better than guessing; 1.00 = perfect")
        st.metric("Business Goal (Module 1)", "+15% cross-sell conversion")

# ============================================================
# TAB 2: KEY INSIGHTS
# ============================================================
with tabs[1]:
    st.subheader("Plain-Language Model Explanation")
    st.info(
        "Think of the model as a **ranking tool**, not a yes/no gate. For every customer, "
        "it produces a score between 0 and 1 (like a probability) showing how likely they "
        "are to respond to a cross-sell offer. Marketing can sort the customer list by "
        "this score and focus effort on the top of the list, instead of contacting everyone equally."
    )

    c1, c2, c3, c4 = st.columns(4)
    m = model_results["logistic_regression"]["metrics"]
    b = model_results["baseline"]["metrics"]
    c1.metric("Of customers flagged 'likely to convert'", f"{m['precision']:.0%}", "actually convert", help="Precision")
    c2.metric("Of all real converters", f"{m['recall']:.0%}", "the model correctly flags", help="Recall")
    c3.metric("Improvement vs. guessing", f"+{(m['roc_auc']-0.5)*100:.0f} pts", "ROC-AUC over a coin flip")
    c4.metric("vs. targeting everyone equally", "Baseline catches 0%", "of converters precisely")

    st.subheader("How the Model Compares to Doing Nothing Different")
    fig = go.Figure()
    models_plot = ["No Model\n(target everyone)", "FirstBank Model\n(targeted)"]
    precision_vals = [0.304, m["precision"]]  # baseline = just the overall conversion rate if you "targeted everyone"
    fig.add_trace(go.Bar(x=models_plot, y=precision_vals, marker_color=[GREY, GOLD],
                          text=[f"{v:.0%}" for v in precision_vals], textposition="outside"))
    fig.update_layout(
        title="Chance a Targeted Customer Actually Converts",
        yaxis_title="Conversion rate among those targeted", yaxis_tickformat=".0%",
        height=380, showlegend=False, plot_bgcolor="white",
        title_font=dict(color=NAVY, size=16),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Targeting customers using the model nearly doubles the conversion rate among "
        "those contacted, compared to targeting customers with no model at all."
    )

# ============================================================
# TAB 3: WHAT DRIVES PREDICTIONS
# ============================================================
with tabs[2]:
    st.subheader("What Drives Predictions? (In Business Terms)")
    st.write("The model looks at **how engaged a customer already is** with their account, not who they are demographically.")

    driver_labels = {
        "frequency": "How often they transact",
        "recency_days": "How recently they last transacted",
        "monetary_total": "Total amount they've transacted",
        "account_balance": "Their account balance",
        "monetary_avg": "Their average transaction size",
        "share_Salary Credit": "Whether salary is paid into the account",
        "share_ATM Withdrawal": "Reliance on ATM withdrawals",
        "share_Airtime/Data": "Airtime/data purchase habits",
        "location_Lagos": "Being based in Lagos",
        "share_Transfer": "Use of bank transfers",
    }
    top = shap_importance.head(10).copy()
    top["business_label"] = top["feature"].map(driver_labels).fillna(top["feature"])
    top = top.sort_values("mean_abs_shap")

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"], y=top["business_label"], orientation="h",
        marker_color=NAVY,
    ))
    fig.update_layout(
        title="Top Factors the Model Weighs Most Heavily",
        xaxis_title="Relative influence on the prediction", height=430,
        plot_bgcolor="white", title_font=dict(color=NAVY, size=16),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "**In plain terms:** customers who transact often, recently, and with higher "
        "balances are scored as more likely to convert. Where a customer lives and how "
        "they spend (airtime vs. bills vs. transfers) matter far less. The model is "
        "reading *engagement*, not demographics."
    )

# ============================================================
# TAB 4: EXAMPLE CUSTOMERS
# ============================================================
with tabs[3]:
    st.subheader("Example Predictions, Explained in Plain Language")
    st.write("Three real (anonymized) customers from the model's test set, showing how the score translates to a business decision.")

    for _, row in examples.head(3).iterrows():
        prob = row["predicted_probability"]
        label = "Likely to Convert" if row["predicted_label"] == 1 else "Unlikely to Convert"
        color = GREEN if row["predicted_label"] == 1 else RED

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div style="background:{color}15; border-left:5px solid {color}; padding:14px; border-radius:6px;">
                <div style="font-size:12px; color:{GREY};">Customer {row['customer_id']}</div>
                <div style="font-size:24px; font-weight:700; color:{color};">{prob:.0%}</div>
                <div style="font-size:13px; color:{color}; font-weight:600;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if row["predicted_label"] == 1:
                    explanation = (
                        f"This customer transacts about {row['frequency']:.0f} times with a total "
                        f"transaction volume of \u20a6{row['monetary_total']:,.0f} in the observed window, "
                        f"and was last active {row['recency_days']:.0f} days ago. That level of engagement "
                        f"is why the model flags them as a strong cross-sell candidate."
                    )
                else:
                    explanation = (
                        f"This customer transacted only {row['frequency']:.0f} time(s), with a total "
                        f"volume of \u20a6{row['monetary_total']:,.0f}, and was last active "
                        f"{row['recency_days']:.0f} days ago. Low recent engagement is why the model "
                        f"does not flag them for this campaign right now."
                    )
                st.write(explanation)
            st.divider()

# ============================================================
# TAB 5: WHAT-IF EXPLORER
# ============================================================
with tabs[4]:
    st.subheader("What-If Analysis")
    st.write("Adjust a customer's engagement level below and see how the model's prediction changes in real time.")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        frequency = st.slider("Transactions in the period", 0, 40, 8)
        recency = st.slider("Days since last transaction", 0, 200, 20)
        monetary_total = st.slider("Total transaction volume (\u20a6)", 0, 2_000_000, 150_000, step=10_000)
        account_balance = st.slider("Account balance (\u20a6)", 0, 3_000_000, 200_000, step=10_000)
        location = st.selectbox("Location", [
            "Lagos", "Kano", "Rivers", "Oyo", "Kaduna", "Ogun", "Enugu", "Delta",
            "Anambra", "Edo", "Plateau", "Abuja FCT", "Cross River", "Imo", "Kwara",
        ])

        row = {
            "age": 40, "account_balance": account_balance, "recency_days": recency,
            "frequency": frequency, "monetary_total": monetary_total,
            "monetary_avg": monetary_total / max(frequency, 1),
            "share_ATM Withdrawal": 0.15, "share_Airtime/Data": 0.05, "share_Bill Payment": 0.1,
            "share_Loan Repayment": 0.0, "share_Other/Unclassified": 0.0, "share_POS Purchase": 0.25,
            "share_Salary Credit": 0.2, "share_Savings Deposit": 0.2, "share_Transfer": 0.05,
            "location": location,
        }
        X = pd.DataFrame([row])[bundle["numeric_features"] + bundle["categorical_features"]]
        X_t = bundle["preprocessor"].transform(X)
        proba = bundle["model"].predict_proba(X_t)[0, 1]

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            title={"text": "Predicted Conversion Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": GOLD},
                "steps": [
                    {"range": [0, 50], "color": "#FDECEA"},
                    {"range": [50, 100], "color": "#E6F4EA"},
                ],
                "threshold": {"line": {"color": RED, "width": 3}, "value": 50},
            },
        ))
        fig.update_layout(height=340, font=dict(color=NAVY))
        st.plotly_chart(fig, use_container_width=True)

        decision = "Would be TARGETED for this campaign" if proba >= 0.5 else "Would NOT be targeted for this campaign"
        st.markdown(f"**Decision at default threshold:** {decision}")
        st.caption(
            "Try raising 'Transactions in the period' or lowering 'Days since last transaction' "
            "and watch the score move \u2014 this is the same lever the counterfactual analysis "
            "in the Module 4 report identified as the most actionable one."
        )

# ============================================================
# TAB 6: FAIRNESS & ETHICS
# ============================================================
with tabs[5]:
    st.subheader("Ethical Compliance Dashboard")
    st.write(
        "Before this model can be used on real campaigns, it must be checked for fairness "
        "across customer groups, per the Ethical AI Charter (Module 1) and Fairness "
        "Objectives (Module 2)."
    )

    mit = fairness["mitigation"]
    fig = go.Figure()
    for attr, label, xpos in [("age_band", "Age Band", 0), ("gender", "Gender", 1)]:
        before = mit[attr]["disparate_impact_ratio_before"]
        after = mit[attr]["disparate_impact_ratio_after"]
        fig.add_trace(go.Bar(name=f"{label} \u2014 Before", x=[label], y=[before], marker_color=RED,
                              showlegend=(xpos == 0), legendgroup="before", offsetgroup=0))
        fig.add_trace(go.Bar(name=f"{label} \u2014 After", x=[label], y=[after], marker_color=GREEN,
                              showlegend=(xpos == 0), legendgroup="after", offsetgroup=1))
    fig.add_hline(y=0.8, line_dash="dash", line_color=GOLD, annotation_text="Fairness threshold (0.80)")
    fig.update_layout(
        title="Fairness Score by Customer Group, Before and After Correction",
        yaxis_title="Fairness score (1.00 = perfectly equal treatment)", barmode="group",
        height=420, plot_bgcolor="white", title_font=dict(color=NAVY, size=16),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What We Found")
        st.warning(
            "Before correction, the model under-scored two groups relative to their true "
            "engagement: customers aged 60 and over, and customers with unspecified gender "
            "in our records. This mirrors a real risk named in FirstBank's Ethical AI "
            "Charter \u2014 that historically under-targeted groups get further overlooked by "
            "an automated tool."
        )
    with col2:
        st.markdown("#### What We Did About It")
        st.success(
            "We applied a bias correction technique that adjusts the model's decision "
            "threshold separately for each group, so no group is systematically "
            "under-scored. After correction, both groups clear FirstBank's fairness bar, "
            "**and overall accuracy did not get worse** \u2014 fairness and performance were "
            "not a trade-off here."
        )

    st.markdown("#### Transparency Statement")
    st.info(
        "This model recommends who to prioritize for marketing outreach. **It does not, "
        "and must not, decide loan approval, pricing, or account access for any customer.** "
        "It was audited for fairness across gender and age, corrected where bias was found, "
        "and will be re-checked every quarter. Any customer or staff member can request a "
        "review of why a specific customer received a given score."
    )

# ============================================================
# TAB 7: LIMITATIONS & NEXT STEPS
# ============================================================
with tabs[6]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What This Model Can and Cannot Do")
        st.markdown("""
**Can do:**
- Rank customers by likelihood of responding to a cross-sell offer
- Flag which behavioral signals matter most
- Be re-audited for fairness on a regular schedule

**Cannot do:**
- Guarantee any individual customer's response
- Be used for credit, pricing, or account-access decisions
- Replace a relationship manager's judgment on sensitive accounts
        """)
        st.subheader("Known Limitations")
        st.markdown("""
- Built and validated on a representative test dataset for this stage of the project; real campaign-response data will be used to retrain before full production use
- Catches roughly 1 in 5 true converters at the default threshold; better suited to prioritizing outreach than an exhaustive list
- Requires quarterly fairness re-checks, since customer behavior patterns shift over time
        """)
    with col2:
        st.subheader("Next Steps")
        st.markdown("""
- Pilot the model on a limited campaign segment and compare actual vs. predicted conversion
- Monitor the fairness dashboard above monthly, not just at launch
- Feed real campaign outcomes back into the model to retrain on FirstBank's own response history
- Extend the same approach to the underserved segments identified in Module 1 for financial-inclusion outreach
        """)

st.divider()
st.caption(
    "FirstBank Nigeria Limited Customer Segmentation Project \u2014 BAN6800 Modules 1-5. "
    "Model: Logistic Regression, registered in MLflow as firstbank-cross-sell-propensity-model. "
    "This dashboard presents findings for stakeholder discussion; it does not constitute a formal recommendation."
)
