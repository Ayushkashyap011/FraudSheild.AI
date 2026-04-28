# src/generate_data.py

import pandas as pd
import numpy as np
import random
from pathlib import Path

# =====================================================
# FraudShield AI - Synthetic Dataset Generator
# Creates realistic fintech transaction fraud dataset
# =====================================================

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
N_ROWS = 50000

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad",
    "Chennai", "Pune", "Kolkata", "Bhubaneswar"
]

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel",
    "fashion", "food", "fuel", "gaming", "gift_card"
]

DEVICE_TYPES = [
    "mobile", "web", "tablet"
]

# -----------------------------
# HELPERS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def weighted_choice(options, probs):
    return np.random.choice(options, p=probs)


# -----------------------------
# GENERATE FEATURES
# -----------------------------
def generate_base_data(n_rows):
    data = pd.DataFrame()

    # Amounts (₹)
    data["transaction_amount"] = np.round(
        np.random.gamma(shape=2.2, scale=1800, size=n_rows), 2
    )

    # Time of day
    data["transaction_hour"] = np.random.randint(0, 24, n_rows)

    # Weekend flag
    data["is_weekend"] = np.random.choice([0, 1], size=n_rows, p=[0.72, 0.28])

    # Merchant categories
    data["merchant_category"] = np.random.choice(
        MERCHANT_CATEGORIES,
        size=n_rows,
        p=[0.24, 0.14, 0.08, 0.12, 0.20, 0.10, 0.07, 0.05]
    )

    # User city
    data["user_location"] = np.random.choice(CITIES, size=n_rows)

    # Device type
    data["device_type"] = np.random.choice(
        DEVICE_TYPES,
        size=n_rows,
        p=[0.72, 0.22, 0.06]
    )

    # Gap since previous transaction (minutes)
    data["previous_transaction_gap"] = np.round(
        np.random.exponential(scale=120, size=n_rows), 2
    )

    # Transactions in last hour
    data["transaction_velocity"] = np.random.poisson(lam=2.2, size=n_rows)

    # Failed login attempts
    data["failed_login_attempts"] = np.random.poisson(lam=0.4, size=n_rows)

    # Account age
    data["account_age_days"] = np.random.randint(1, 3650, n_rows)

    # IP risk score
    data["ip_risk_score"] = np.round(
        np.random.beta(a=2, b=5, size=n_rows) * 100, 2
    )

    # Geo mismatch
    data["geo_mismatch_flag"] = np.random.choice(
        [0, 1], size=n_rows, p=[0.92, 0.08]
    )

    # New device usage
    data["unusual_device_flag"] = np.random.choice(
        [0, 1], size=n_rows, p=[0.90, 0.10]
    )

    return data


# -----------------------------
# FRAUD LOGIC
# -----------------------------
def generate_target(df):
    score = np.zeros(len(df))

    # High amount suspicious
    score += np.where(df["transaction_amount"] > 12000, 1.3, 0)
    score += np.where(df["transaction_amount"] > 30000, 1.8, 0)

    # Midnight transactions
    score += np.where(
        (df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5),
        1.2,
        0
    )

    # Rapid multiple transactions
    score += np.where(df["transaction_velocity"] >= 5, 1.4, 0)

    # Very low time gap = suspicious burst
    score += np.where(df["previous_transaction_gap"] < 3, 1.1, 0)

    # Failed logins
    score += df["failed_login_attempts"] * 0.55

    # New account risky
    score += np.where(df["account_age_days"] < 30, 1.6, 0)
    score += np.where(df["account_age_days"] < 7, 1.0, 0)

    # Risky IP
    score += (df["ip_risk_score"] / 100) * 2.2

    # Geo mismatch
    score += df["geo_mismatch_flag"] * 1.8

    # Unusual device
    score += df["unusual_device_flag"] * 1.4

    # High-risk merchant categories
    risky_merchants = ["electronics", "gaming", "gift_card", "travel"]
    score += np.where(df["merchant_category"].isin(risky_merchants), 1.0, 0)

    # Web fraud slightly higher than mobile
    score += np.where(df["device_type"] == "web", 0.5, 0)

    # Convert to probability
    probability = sigmoid(score - 6.8)

    # Final fraud labels
    fraud = np.random.binomial(1, probability)

    return fraud


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Generating synthetic fraud dataset...")

    df = generate_base_data(N_ROWS)
    df["is_fraud"] = generate_target(df)

    # Create folder if not exists
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fraud_transactions.csv"
    df.to_csv(output_path, index=False)

    fraud_rate = round(df["is_fraud"].mean() * 100, 2)

    print("=" * 50)
    print("Dataset Created Successfully")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Fraud Rate: {fraud_rate}%")
    print(f"Saved To: {output_path}")
    print("=" * 50)

    print("\nSample Data:")
    print(df.head())


if __name__ == "__main__":
    main()
    