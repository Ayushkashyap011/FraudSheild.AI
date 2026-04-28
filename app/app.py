# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import random

# ==================================================
# IMPORT PREDICTION ENGINE
# ==================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from predict import predict_transaction

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PREMIUM CSS
# ==================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #00F5B0;
    margin-bottom: 0;
}

.sub-title {
    color: #A9A9A9;
    font-size: 16px;
    margin-top: -10px;
}

.card {
    background: #111111;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #262626;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.25);
}

.green {
    color: #00FF99;
    font-weight: bold;
}

.red {
    color: #FF4B4B;
    font-weight: bold;
}

.orange {
    color: #FFA500;
    font-weight: bold;
}

.small-text {
    color: #BFBFBF;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown('<p class="main-title"> FraudShield AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Premium Fraud Detection & Real-Time Risk Intelligence Platform</p>',
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header(" Transaction Inputs")

transaction_amount = st.sidebar.number_input("Amount ₹", 1.0, value=5000.0)
transaction_hour = st.sidebar.slider("Hour", 0, 23, 12)
is_weekend = st.sidebar.selectbox("Weekend?", [0, 1])

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    ["grocery", "electronics", "travel", "fashion",
     "food", "fuel", "gaming", "gift_card"]
)

user_location = st.sidebar.selectbox(
    "User Location",
    ["Mumbai", "Delhi", "Bangalore", "Hyderabad",
     "Chennai", "Pune", "Kolkata", "Bhubaneswar"]
)

device_type = st.sidebar.selectbox("Device Type", ["mobile", "web", "tablet"])
previous_transaction_gap = st.sidebar.number_input("Prev Gap (mins)", 0.0, value=15.0)
transaction_velocity = st.sidebar.slider("Txns Last Hour", 0, 20, 2)
failed_login_attempts = st.sidebar.slider("Failed Logins", 0, 10, 0)
account_age_days = st.sidebar.number_input("Account Age Days", 1, value=365)
ip_risk_score = st.sidebar.slider("IP Risk Score", 0, 100, 20)
geo_mismatch_flag = st.sidebar.selectbox("Geo Mismatch?", [0, 1])
unusual_device_flag = st.sidebar.selectbox("Unusual Device?", [0, 1])

analyze = st.sidebar.button(" Analyze Transaction")

# ==================================================
# DEFAULT SCREEN
# ==================================================
if not analyze:
    st.info("Use sidebar controls and click **Analyze Transaction**.")
    st.stop()

# ==================================================
# INPUT OBJECT
# ==================================================
input_data = {
    "transaction_amount": transaction_amount,
    "transaction_hour": transaction_hour,
    "is_weekend": is_weekend,
    "merchant_category": merchant_category,
    "user_location": user_location,
    "device_type": device_type,
    "previous_transaction_gap": previous_transaction_gap,
    "transaction_velocity": transaction_velocity,
    "failed_login_attempts": failed_login_attempts,
    "account_age_days": account_age_days,
    "ip_risk_score": ip_risk_score,
    "geo_mismatch_flag": geo_mismatch_flag,
    "unusual_device_flag": unusual_device_flag
}

# ==================================================
# PREDICTION
# ==================================================
result = predict_transaction(input_data)

prediction = result["prediction"]
prob = result["fraud_probability"]
risk = result["risk_level"]
confidence = result["model_confidence"]

# ==================================================
# RISK COLOR
# ==================================================
if risk == "Low":
    risk_color = "green"
elif risk == "Medium":
    risk_color = "orange"
else:
    risk_color = "red"

pred_color = "red" if prediction == "Fraud" else "green"

# ==================================================
# TOP KPI CARDS
# ==================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="card"><h4>Prediction</h4><h2 class="{pred_color}">{prediction}</h2></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f'<div class="card"><h4>Fraud Probability</h4><h2>{prob}%</h2></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f'<div class="card"><h4>Risk Level</h4><h2 class="{risk_color}">{risk}</h2></div>',
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f'<div class="card"><h4>Confidence</h4><h2>{confidence}%</h2></div>',
        unsafe_allow_html=True
    )

st.markdown("")

# ==================================================
# GAUGE CHART
# ==================================================
left, right = st.columns([2, 1])

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Fraud Risk Meter"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "#00FF99"},
                {'range': [30, 70], 'color': "#FFA500"},
                {'range': [70, 100], 'color': "#FF4B4B"}
            ]
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# SUSPICIOUS FACTORS
# ==================================================
with right:
    st.markdown("###  Suspicious Factors")

    for item in result["suspicious_factors"]:
        st.warning(item)

# ==================================================
# FRAUD TREND CHARTS
# ==================================================
st.markdown("##  Fraud Analytics")

c1, c2 = st.columns(2)

with c1:
    trend_df = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Fraud Cases": [14, 19, 12, 21, 18, 25, 17]
    })

    fig2 = px.line(
        trend_df,
        x="Day",
        y="Fraud Cases",
        markers=True,
        title="Weekly Fraud Activity"
    )
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    device_df = pd.DataFrame({
        "Device": ["Mobile", "Web", "Tablet"],
        "Fraud %": [52, 38, 10]
    })

    fig3 = px.pie(
        device_df,
        names="Device",
        values="Fraud %",
        title="Fraud by Device Type"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ==================================================
# MOCK SHAP EXPLAINABILITY
# ==================================================
st.markdown("##  Model Explanation")

explain_data = pd.DataFrame({
    "Feature": [
        "IP Risk Score",
        "Failed Logins",
        "Transaction Amount",
        "Account Age",
        "Geo Mismatch"
    ],
    "Impact": [
        random.randint(5, 20),
        random.randint(4, 15),
        random.randint(3, 12),
        -random.randint(2, 8),
        random.randint(4, 12)
    ]
})

fig4 = px.bar(
    explain_data,
    x="Impact",
    y="Feature",
    orientation="h",
    title="Top Drivers Behind Prediction"
)

st.plotly_chart(fig4, use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Built by Kashyap  | FraudShield AI | End-to-End ML Product")