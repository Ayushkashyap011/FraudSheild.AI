"""Streamlit application for FraudShield AI."""

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

import config
from src.predict import get_model_summary, predict_batch, predict_transaction


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.22), transparent 24%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 22%),
            linear-gradient(180deg, #050816 0%, #0a1020 38%, #f7fafc 100%);
        color: #e2e8f0;
    }
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li,
    .main [data-testid="stMarkdownContainer"] span {
        color: #dbeafe;
    }
    .hero-title {
        font-size: 3.15rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin-bottom: 0.18rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        color: rgba(226, 232, 240, 0.88);
        font-size: 1.04rem;
        margin-bottom: 0.85rem;
        max-width: 52rem;
    }
    .info-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.18);
        color: #e0f2fe;
        border: 1px solid rgba(125, 211, 252, 0.28);
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }
    .panel-title {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .panel-subtitle {
        color: rgba(226, 232, 240, 0.8);
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.90);
        box-shadow: 0 16px 42px rgba(2, 6, 23, 0.24);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1;
        color: #f8fafc;
        word-break: normal;
        overflow-wrap: anywhere;
    }
    .metric-value.compact {
        font-size: 1.45rem;
        line-height: 1.08;
    }
    .metric-support {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }
    .badge-low { color: #052e16; background: #86efac; }
    .badge-medium { color: #713f12; background: #fde68a; }
    .badge-high { color: #7f1d1d; background: #fca5a5; }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .section-shell {
        background: rgba(15, 23, 42, 0.84);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.18);
    }
    .accent-line {
        height: 4px;
        width: 160px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #38bdf8 48%, #f59e0b 100%);
        margin: 0.75rem 0 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _load_dataset() -> pd.DataFrame | None:
    if config.DATA_PATH.exists():
        return pd.read_csv(config.DATA_PATH)
    return None


def _risk_badge(probability: float) -> tuple[str, str]:
    if probability < config.RISK_LEVELS["Low"]:
        return "badge-low", "Low"
    if probability < config.RISK_LEVELS["Medium"]:
        return "badge-medium", "Medium"
    return "badge-high", "High"


def _load_example(example_name: str) -> None:
    st.session_state["transaction_form_values"] = config.MODEL_EXAMPLES[example_name].copy()


def _read_form_values() -> dict[str, object]:
    return st.session_state.get("transaction_form_values", config.MODEL_EXAMPLES["legit"].copy())


def _ensure_form_state() -> None:
    if "transaction_form_values" not in st.session_state:
        st.session_state["transaction_form_values"] = config.MODEL_EXAMPLES["legit"].copy()


def _log_feedback(feedback_label: str, result: dict[str, object], input_data: dict[str, object], threshold: float) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feedback": feedback_label,
        "model_name": result.get("model_name"),
        "trained_at": result.get("trained_at"),
        "threshold": threshold,
        "prediction": result.get("prediction"),
        "fraud_probability": result.get("fraud_probability"),
        "risk_level": result.get("risk_level"),
        **input_data,
    }

    feedback_frame = pd.DataFrame([row])
    file_exists = config.FEEDBACK_PATH.exists()
    feedback_frame.to_csv(config.FEEDBACK_PATH, mode="a", header=not file_exists, index=False)


def _color_prediction_frame(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_rows(row: pd.Series) -> list[str]:
        color = "#7f1d1d" if row.get("prediction") == "Fraud" else "#14532d"
        return [f"background-color: {color}; color: #f8fafc"] * len(row)

    return frame.style.apply(color_rows, axis=1).format(
        {
            "fraud_probability": "{:.4f}",
            "fraud_probability_pct": "{:.2f}",
            "threshold": "{:.2f}",
        }
    )


def _render_header() -> None:
    summary = get_model_summary()
    trained_at = datetime.fromisoformat(summary["trained_at"]).strftime("%Y-%m-%d %H:%M:%S") if summary.get("trained_at") not in (None, "unknown") else "unknown"
    version_label = Path(summary["model_path"]).stem.replace("fraud_model_", "v")
    st.markdown(f'<div class="hero-title">{config.APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">{config.APP_TAGLINE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-pill">{config.DISCLAIMER}</div>', unsafe_allow_html=True)
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    model_cols = st.columns(4)
    with model_cols[0]:
        st.markdown(
            f'<div class="card"><div class="metric-label">Model</div><div class="metric-value compact">{summary["model_name"]}</div><div class="metric-support">Best on validation split</div></div>',
            unsafe_allow_html=True,
        )
    with model_cols[1]:
        st.markdown(
            f'<div class="card"><div class="metric-label">Last trained</div><div class="metric-value compact">{trained_at}</div><div class="metric-support">Latest training run</div></div>',
            unsafe_allow_html=True,
        )
    with model_cols[2]:
        st.markdown(
            '<div class="card"><div class="metric-label">Data</div><div class="metric-value compact">Demo data</div><div class="metric-support">Portfolio-safe sample transactions</div></div>',
            unsafe_allow_html=True,
        )


def _render_transaction_form() -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Single transaction analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Score one transaction, review SHAP drivers, and capture analyst feedback.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Load fraud example", on_click=_load_example, args=("fraud",), width="stretch")
    with col2:
        st.button("Load legit example", on_click=_load_example, args=("legit",), width="stretch")

    defaults = _read_form_values()

    with st.form("single_transaction_form"):
        grid = st.columns(2)
        values = {}

        values["transaction_amount"] = grid[0].number_input("Transaction amount", min_value=0.0, value=float(defaults["transaction_amount"]), step=50.0)
        values["transaction_hour"] = grid[1].number_input("Transaction hour", min_value=0, max_value=23, value=int(defaults["transaction_hour"]), step=1)
        values["is_weekend"] = grid[0].selectbox("Weekend transaction", [0, 1], index=[0, 1].index(int(defaults["is_weekend"])))
        merchant_options = ["grocery", "electronics", "travel", "fashion", "food", "fuel", "gaming", "gift_card"]
        values["merchant_category"] = grid[1].selectbox("Merchant category", merchant_options, index=merchant_options.index(defaults["merchant_category"]))
        location_options = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Bhubaneswar"]
        values["user_location"] = grid[0].selectbox("User location", location_options, index=location_options.index(defaults["user_location"]))
        values["device_type"] = grid[1].selectbox("Device type", ["mobile", "web", "tablet"], index=["mobile", "web", "tablet"].index(defaults["device_type"]))
        values["previous_transaction_gap"] = grid[0].number_input("Previous transaction gap (minutes)", min_value=0.0, value=float(defaults["previous_transaction_gap"]), step=1.0)
        values["transaction_velocity"] = grid[1].number_input("Transaction velocity", min_value=0, value=int(defaults["transaction_velocity"]), step=1)
        values["failed_login_attempts"] = grid[0].number_input("Failed login attempts", min_value=0, value=int(defaults["failed_login_attempts"]), step=1)
        values["account_age_days"] = grid[1].number_input("Account age (days)", min_value=1, value=int(defaults["account_age_days"]), step=1)
        values["ip_risk_score"] = grid[0].slider("IP risk score", min_value=0.0, max_value=100.0, value=float(defaults["ip_risk_score"]), step=1.0)
        values["geo_mismatch_flag"] = grid[1].selectbox("Geo mismatch flag", [0, 1], index=[0, 1].index(int(defaults["geo_mismatch_flag"])))
        values["unusual_device_flag"] = grid[0].selectbox("Unusual device flag", [0, 1], index=[0, 1].index(int(defaults["unusual_device_flag"])))

        submitted = st.form_submit_button("Analyze transaction", width="stretch")

    if submitted:
        st.session_state["transaction_form_values"] = values.copy()
        result = predict_transaction(values)
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = values

    if "last_result" not in st.session_state:
        st.info("Load an example or enter transaction details, then run the analysis.")
        return

    result = st.session_state["last_result"]
    input_data = st.session_state["last_inputs"]
    probability = float(result["fraud_probability"])
    badge_class, badge_text = _risk_badge(probability / 100.0)

    metrics = st.columns(4)
    with metrics[0]:
        st.markdown(
            f'<div class="card"><div class="metric-label">Prediction</div><div class="metric-value">{result["prediction"]}</div><div class="metric-support">Current operating rule</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[1]:
        st.markdown(
            f'<div class="card"><div class="metric-label">Fraud probability</div><div class="metric-value">{probability:.2f}%</div><div class="metric-support">Model confidence score</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[2]:
        st.markdown(
            f'<div class="card"><div class="metric-label">Risk level</div><div class="metric-value"><span class="badge {badge_class}">{badge_text}</span></div><div class="metric-support">Business banding</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[3]:
        st.markdown(
            f'<div class="card"><div class="metric-label">Model confidence</div><div class="metric-value">{float(result["model_confidence"]):.2f}%</div><div class="metric-support">Higher means stronger signal</div></div>',
            unsafe_allow_html=True,
        )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            title={"text": "Fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626"},
                "steps": [
                    {"range": [0, 33], "color": "#dcfce7"},
                    {"range": [33, 67], "color": "#fef3c7"},
                    {"range": [67, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))

    left, right = st.columns([1.2, 0.8])
    with left:
        st.plotly_chart(gauge, width="stretch")
    with right:
        st.markdown('<div class="panel-title">SHAP drivers</div>', unsafe_allow_html=True)
        shap_table = pd.DataFrame(result["shap_explanations"])
        if shap_table.empty:
            st.info("No SHAP explanation available for this model.")
        else:
            chart_frame = shap_table.set_index("feature")[["shap_value"]].sort_values("shap_value")
            st.bar_chart(chart_frame)
            st.dataframe(
                shap_table[["feature", "shap_value", "direction"]],
                width="stretch",
                hide_index=True,
            )

    st.markdown('<div class="panel-title">Analyst feedback</div>', unsafe_allow_html=True)
    fb_left, fb_right = st.columns(2)
    if fb_left.button("👍 Helpful", width="stretch"):
        _log_feedback("up", result, input_data, float(result.get("threshold", config.DEFAULT_THRESHOLD)))
        st.success("Feedback saved to feedback.csv")
    if fb_right.button("👎 Needs review", width="stretch"):
        _log_feedback("down", result, input_data, float(result.get("threshold", config.DEFAULT_THRESHOLD)))
        st.warning("Feedback saved to feedback.csv")
    st.markdown('</div>', unsafe_allow_html=True)


def _render_batch_tab() -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Batch scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Upload a CSV, score all rows, and download the color-coded results.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV with the feature columns", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to score multiple transactions at once.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    batch_df = pd.read_csv(uploaded)
    scored = predict_batch(batch_df)
    styled = _color_prediction_frame(scored)
    st.dataframe(styled, width="stretch", hide_index=True)

    csv_bytes = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored results",
        data=csv_bytes,
        file_name="fraudshield_batch_results.csv",
        mime="text/csv",
        width="stretch",
    )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_analytics_tab() -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Score distributions, predicted fraud rate, and global model importance.</div>', unsafe_allow_html=True)
    dataset = _load_dataset()
    if dataset is None:
        st.info("Place data/fraud_transactions.csv locally to unlock the analytics view.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    scored = predict_batch(dataset)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            scored,
            x="fraud_probability",
            color="prediction",
            nbins=30,
            title="Score distribution",
            color_discrete_map={"Fraud": "#dc2626", "Legit": "#16a34a"},
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fraud_rate = float((scored["prediction"] == "Fraud").mean() * 100)
        st.metric("Predicted fraud rate", f"{fraud_rate:.2f}%")
        rate_fig = px.histogram(
            scored,
            x="risk_level",
            category_orders={"risk_level": ["Low", "Medium", "High"]},
            title="Risk band distribution",
        )
        st.plotly_chart(rate_fig, width="stretch")

    summary = get_model_summary()
    feature_importance = summary.get("feature_importance")
    if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
        st.markdown('<div class="panel-title">Global feature importance</div>', unsafe_allow_html=True)
        top_importance = feature_importance.head(12).set_index("feature")[["importance"]]
        st.bar_chart(top_importance)
    else:
        st.info("No feature importance was stored with the model artifact.")
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    _ensure_form_state()
    _render_header()
    tabs = st.tabs(["Single transaction", "Batch prediction", "Analytics"])
    with tabs[0]:
        _render_transaction_form()
    with tabs[1]:
        _render_batch_tab()
    with tabs[2]:
        _render_analytics_tab()

    st.divider()
    st.caption("FraudShield AI | Synthetic portfolio project | SHAP-backed explainability")


if __name__ == "__main__":
    main()