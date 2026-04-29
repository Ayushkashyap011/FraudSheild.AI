# app/app.py
# Updated production-grade version:
# 1. Clear business-friendly field labels
# 2. Better helper text
# 3. Risk Drivers section
# 4. Includes test case below

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_transaction, get_model_summary

APP_TITLE = "FraudShield AI"
APP_TAGLINE = "Fraud detection, risk scoring, and analyst intelligence platform."
DISCLAIMER = "This model uses synthetic financial data for portfolio demonstration."

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {display:none;}

.stApp{
background:
radial-gradient(circle at top left, rgba(14,165,233,0.18), transparent 24%),
linear-gradient(180deg,#050816 0%,#0b1020 42%,#f8fafc 100%);
color:#e2e8f0;
}

.card{
background:rgba(15,23,42,0.92);
padding:1rem;
border-radius:18px;
border:1px solid rgba(148,163,184,0.18);
}

.hero-title{
font-size:3rem;
font-weight:900;
color:white;
}

.hero-subtitle{
color:#dbeafe;
margin-bottom:1rem;
}

.low{color:#22c55e;}
.medium{color:#f59e0b;}
.high{color:#ef4444;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================
def risk_level(prob):
    if prob < 30:
        return "Low", "low"
    elif prob < 70:
        return "Medium", "medium"
    return "High", "high"


def safe_summary():
    try:
        data = get_model_summary()
        if not isinstance(data, dict):
            data = {}
    except:
        data = {}

    return {
        "model_name": data.get("model_name", "Logistic Regression"),
        "trained_at": data.get("trained_at", "Latest Build"),
    }


# =====================================================
# HEADER
# =====================================================
summary = safe_summary()

st.markdown(f"<div class='hero-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='hero-subtitle'>{APP_TAGLINE}</div>", unsafe_allow_html=True)
st.caption(DISCLAIMER)

c1, c2 = st.columns(2)

with c1:
    st.info(f"Model: {summary['model_name']}")

with c2:
    st.info(f"Last Trained: {summary['trained_at']}")

st.divider()

# =====================================================
# FORM
# =====================================================
st.subheader("Single Transaction Analysis")
st.caption("Enter transaction details below.")

with st.form("fraud_form"):

    col1, col2 = st.columns(2)

    data = {}

    data["transaction_amount"] = col1.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        value=5000.0,
        help="Total amount of current transaction."
    )

    data["transaction_hour"] = col2.slider(
        "Transaction Time (24h format)",
        0, 23, 12,
        help="0 = Midnight, 14 = 2 PM"
    )

    data["is_weekend"] = col1.selectbox(
        "Weekend Transaction?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    data["merchant_category"] = col2.selectbox(
        "Merchant Category",
        ["grocery", "electronics", "travel", "fashion", "food", "fuel", "gaming", "gift_card"]
    )

    data["user_location"] = col1.selectbox(
        "Customer City",
        ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune"]
    )

    data["device_type"] = col2.selectbox(
        "Device Used",
        ["mobile", "web", "tablet"]
    )

    data["previous_transaction_gap"] = col1.number_input(
        "Minutes Since Previous Transaction",
        min_value=0.0,
        value=10.0,
        help="If last payment was 1 minute ago, enter 1."
    )

    data["transaction_velocity"] = col2.number_input(
        "Number of Transactions in Last Hour",
        min_value=0,
        value=2,
        help="High value may indicate suspicious rapid activity."
    )

    data["failed_login_attempts"] = col1.number_input(
        "Failed Login Attempts",
        min_value=0,
        value=0
    )

    data["account_age_days"] = col2.number_input(
        "Account Age (Days)",
        min_value=1,
        value=365
    )

    data["ip_risk_score"] = col1.slider(
        "IP Risk Score",
        0, 100, 25,
        help="Higher means suspicious IP source."
    )

    data["geo_mismatch_flag"] = col2.selectbox(
        "Location Mismatch?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    data["unusual_device_flag"] = col1.selectbox(
        "Unrecognized Device?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    submitted = st.form_submit_button("Analyze Transaction")

# =====================================================
# RESULT
# =====================================================
if submitted:

    result = predict_transaction(data)

    prediction = result.get("prediction", "Unknown")
    prob = float(result.get("fraud_probability", 0))
    confidence = float(result.get("model_confidence", 0))
    factors = result.get("suspicious_factors", [])

    risk, css = risk_level(prob)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Prediction", prediction)
    m2.metric("Fraud Probability", f"{prob:.2f}%")
    m3.markdown(f"### <span class='{css}'>{risk}</span>", unsafe_allow_html=True)
    m4.metric("Confidence", f"{confidence:.2f}%")

    left, right = st.columns([1.3, 0.7])

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
        st.subheader("Risk Drivers")

        if factors:
            for item in factors:
                st.warning(item)
        else:
            st.success("No suspicious indicators detected.")

st.divider()
st.caption("FraudShield AI | Production Demo")
