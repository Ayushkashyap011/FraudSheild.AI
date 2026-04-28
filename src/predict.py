# src/predict.py

import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = Path("models/fraud_model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)


# ==========================================
# RISK LABEL FUNCTION
# ==========================================
def get_risk_level(prob):
    """
    Convert probability to business-friendly risk band
    """
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    return "High"


# ==========================================
# SUSPICIOUS FACTORS
# ==========================================
def detect_flags(row):
    """
    Human-readable fraud reasons
    """
    flags = []

    if row["transaction_amount"] > 15000:
        flags.append("High transaction amount")

    if row["transaction_hour"] >= 0 and row["transaction_hour"] <= 5:
        flags.append("Late-night transaction")

    if row["failed_login_attempts"] >= 3:
        flags.append("Multiple failed logins")

    if row["transaction_velocity"] >= 5:
        flags.append("Too many recent transactions")

    if row["account_age_days"] < 30:
        flags.append("Very new account")

    if row["ip_risk_score"] > 70:
        flags.append("Risky IP address")

    if row["geo_mismatch_flag"] == 1:
        flags.append("Geo-location mismatch")

    if row["unusual_device_flag"] == 1:
        flags.append("Unusual device detected")

    if row["merchant_category"] in ["electronics", "gaming", "gift_card"]:
        flags.append("High-risk merchant category")

    if len(flags) == 0:
        flags.append("No major suspicious indicators")

    return flags


# ==========================================
# PREDICT SINGLE TRANSACTION
# ==========================================
def predict_transaction(input_data: dict):
    """
    Input = dictionary of one transaction
    Output = prediction + probability + reasons
    """

    df = pd.DataFrame([input_data])

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    label = "Fraud" if pred == 1 else "Genuine"
    risk = get_risk_level(prob)
    reasons = detect_flags(input_data)

    result = {
        "prediction": label,
        "fraud_probability": round(prob * 100, 2),
        "risk_level": risk,
        "suspicious_factors": reasons,
        "model_confidence": round(max(prob, 1 - prob) * 100, 2)
    }

    return result


# ==========================================
# DEMO TEST
# ==========================================
if __name__ == "__main__":

    sample_transaction = {
        "transaction_amount": 32000,
        "transaction_hour": 2,
        "is_weekend": 1,
        "merchant_category": "electronics",
        "user_location": "Mumbai",
        "device_type": "web",
        "previous_transaction_gap": 1.5,
        "transaction_velocity": 7,
        "failed_login_attempts": 4,
        "account_age_days": 8,
        "ip_risk_score": 88,
        "geo_mismatch_flag": 1,
        "unusual_device_flag": 1
    }

    result = predict_transaction(sample_transaction)

    print("=" * 60)
    print("FraudShield AI Prediction")
    print("=" * 60)
    print("Prediction        :", result["prediction"])
    print("Fraud Probability :", result["fraud_probability"], "%")
    print("Risk Level        :", result["risk_level"])
    print("Model Confidence  :", result["model_confidence"], "%")

    print("\nSuspicious Factors:")
    for item in result["suspicious_factors"]:
        print("-", item)
