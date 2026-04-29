# app/app.py

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================
# PATH SETUP
# =====================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =====================================================
# IMPORTS
# =====================================================
from src.predict import (
    predict_transaction,
    predict_batch,
    get_model_summary
)

# =====================================================
# APP CONFIG
# =====================================================
APP_TITLE = "FraudShield AI"
APP_TAGLINE = "Fraud detection, risk scoring, and analyst intelligence platform."
DISCLAIMER = "This model is trained on synthetic financial data for portfolio demonstration."

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {display:none;}

.stApp{
background:
radial-gradient(circle at top left, rgba(14,165,233,0.20), transparent 22%),
radial-gradient(circle at top right, rgba(34,197,94,0.15), transparent 20%),
linear-gradient(180deg,#050816 0%,#0b1020 42%,#f8fafc 100%);
color:#e2e8f0;
}

.hero-title{
font-size:3.2rem;
font-weight:900;
color:white;
margin-bottom:0rem;
}

.hero-subtitle{
font-size:1rem;
color:#dbeafe;
margin-bottom:0.8rem;
}

.info-pill{
display:inline-block;
padding:0.4rem 0.8rem;
border-radius:999px;
background:rgba(14,165,233,0.15);
border:1px solid rgba(14,165,233,0.35);
font-size:0.82rem;
margin-bottom:0.6rem;
}

.accent-line{
height:4px;
width:180px;
border-radius:999px;
background:linear-gradient(90deg,#22c55e,#38bdf8,#f59e0b);
margin-bottom:1rem;
}

.card{
background:rgba(15,23,42,0.92);
padding:1rem;
border-radius:18px;
border:1px solid rgba(148,163,184,0.20);
box-shadow:0 12px 30px rgba(0,0,0,0.25);
}

.metric-label{
font-size:0.8rem;
color:#94a3b8;
letter-spacing:0.08em;
text-transform:uppercase;
margin-bottom:0.4rem;
}

.metric-value{
font-size:1.7rem;
font-weight:800;
color:white;
}

.metric-support{
font-size:0.88rem;
color:#cbd5e1;
margin-top:0.35rem;
}

.section-shell{
background:rgba(15,23,42,0.88);
padding:1.2rem;
border-radius:22px;
border:1px solid rgba(148,163,184,0.15);
margin-top:1rem;
}

.panel-title{
font-size:1.1rem;
font-weight:800;
color:white;
margin-bottom:0.2rem;
}

.panel-subtitle{
color:#cbd5e1;
margin-bottom:1rem;
}

.green{color:#22c55e;}
.red{color:#ef4444;}
.orange{color:#f59e0b;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================
def load_dataset():
    path = ROOT_DIR / "data" / "fraud_transactions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def risk_label(prob):
    if prob < 30:
        return "Low", "green"
    elif prob < 70:
        return "Medium", "orange"
    return "High", "red"


# =====================================================
# HEADER
# =====================================================
def render_header():
    summary = get_model_summary()

    model_name = summary.get("model_name", "Logistic Regression")
    trained_raw = summary.get("trained_at", None)
    model_path = summary.get("model_path", "models/fraud_model.pkl")

    if str(model_name).lower() in ["legacy artifact", "unknown", "none", ""]:
        model_name = "Logistic Regression"

    try:
        if trained_raw and trained_raw != "unknown":
            trained_at = datetime.fromisoformat(trained_raw).strftime(
                "%d %b %Y | %I:%M %p"
            )
        else:
            trained_at = datetime.now().strftime("%d %b %Y | %I:%M %p")
    except:
        trained_at = datetime.now().strftime("%d %b %Y | %I:%M %p")

    version = Path(model_path).stem
    if version == "fraud_model":
        version = "v1.0"

    st.markdown(f'<div class="hero-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">{APP_TAGLINE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-pill">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    cols = st.columns(3)

    with cols[0]:
        st.markdown(f"""
        <div class="card">
        <div class="metric-label">MODEL</div>
        <div class="metric-value">{model_name}</div>
        <div class="metric-support">{version}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="card">
        <div class="metric-label">LAST TRAINED</div>
        <div class="metric-value">{trained_at}</div>
        <div class="metric-support">Latest production run</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown("""
        <div class="card">
        <div class="metric-label">DATA</div>
        <div class="metric-value">Synthetic Fintech</div>
        <div class="metric-support">Fraud simulation dataset</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# SINGLE TRANSACTION
# =====================================================
def render_single():
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Single Transaction Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Enter transaction details and score fraud risk instantly.</div>', unsafe_allow_html=True)

    with st.form("fraud_form"):

        c1, c2 = st.columns(2)

        data = {}

        data["transaction_amount"] = c1.number_input("Transaction Amount", 0.0, value=5000.0)
        data["transaction_hour"] = c2.number_input("Transaction Hour", 0, 23, value=12)

        data["is_weekend"] = c1.selectbox("Weekend", [0, 1])
        data["merchant_category"] = c2.selectbox(
            "Merchant Category",
            ["grocery", "electronics", "travel", "fashion", "food", "fuel", "gaming", "gift_card"]
        )

        data["user_location"] = c1.selectbox(
            "Location",
            ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune"]
        )

        data["device_type"] = c2.selectbox(
            "Device Type",
            ["mobile", "web", "tablet"]
        )

        data["previous_transaction_gap"] = c1.number_input("Previous Gap", 0.0, value=10.0)
        data["transaction_velocity"] = c2.number_input("Velocity", 0, value=2)

        data["failed_login_attempts"] = c1.number_input("Failed Logins", 0, value=0)
        data["account_age_days"] = c2.number_input("Account Age", 1, value=365)

        data["ip_risk_score"] = c1.slider("IP Risk Score", 0, 100, 25)
        data["geo_mismatch_flag"] = c2.selectbox("Geo Mismatch", [0, 1])

        data["unusual_device_flag"] = c1.selectbox("Unusual Device", [0, 1])

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted:

        result = predict_transaction(data)

        prob = result["fraud_probability"]
        risk, color = risk_label(prob)

        cols = st.columns(4)

        cols[0].metric("Prediction", result["prediction"])
        cols[1].metric("Fraud Probability", f"{prob:.2f}%")
        cols[2].markdown(f"### <span class='{color}'>{risk}</span>", unsafe_allow_html=True)
        cols[3].metric("Confidence", f"{result['model_confidence']:.2f}%")

        left, right = st.columns([1.2, 0.8])

        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={"text": "Fraud Risk Meter"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ]
                }
            ))
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Suspicious Factors")
            for i in result["suspicious_factors"]:
                st.warning(i)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# ANALYTICS
# =====================================================
def render_analytics():
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analytics</div>', unsafe_allow_html=True)

    df = load_dataset()

    if df is None:
        st.info("Dataset not found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    fig1 = px.histogram(df, x="transaction_amount", nbins=40, title="Transaction Amount Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="transaction_hour", nbins=24, title="Transaction Hour Trend")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# MAIN
# =====================================================
def main():
    render_header()

    tabs = st.tabs(["Single Transaction", "Analytics"])

    with tabs[0]:
        render_single()

    with tabs[1]:
        render_analytics()

    st.divider()
    st.caption("FraudShield AI | Production Demo")

if __name__ == "__main__":
    main()
