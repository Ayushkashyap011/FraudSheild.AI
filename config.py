"""Central configuration for FraudShield AI.

Keep project constants here so the training, prediction, and Streamlit
application share the same feature list, thresholds, and paths.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
APP_DIR = ROOT_DIR / "app"
SRC_DIR = ROOT_DIR / "src"

DATA_PATH = DATA_DIR / "fraud_transactions.csv"
FEEDBACK_PATH = ROOT_DIR / "feedback.csv"
MODEL_PREFIX = "fraud_model"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
SHAP_BACKGROUND_SIZE = 75
TOP_SHAP_FEATURES = 3

TARGET_COLUMN = "is_fraud"
FEATURE_COLUMNS = [
    "transaction_amount",
    "transaction_hour",
    "is_weekend",
    "merchant_category",
    "user_location",
    "device_type",
    "previous_transaction_gap",
    "transaction_velocity",
    "failed_login_attempts",
    "account_age_days",
    "ip_risk_score",
    "geo_mismatch_flag",
    "unusual_device_flag",
]

NUMERIC_FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "is_weekend",
    "previous_transaction_gap",
    "transaction_velocity",
    "failed_login_attempts",
    "account_age_days",
    "ip_risk_score",
    "geo_mismatch_flag",
    "unusual_device_flag",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "user_location",
    "device_type",
]

MODEL_SPECS = {
    "Logistic Regression": {
        "max_iter": 2000,
        "solver": "lbfgs",
        "C": 1.5,
    },
    "Random Forest": {
        "n_estimators": 350,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "n_jobs": -1,
    },
    "XGBoost": {
        "n_estimators": 320,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
    },
}

DEFAULT_THRESHOLD = 0.50
RISK_LEVELS = {
    "Low": 0.33,
    "Medium": 0.67,
}

APP_TITLE = "FraudShield AI"
APP_TAGLINE = "Fraud detection, risk scoring, and SHAP-backed analyst support."
DISCLAIMER = "Synthetic data only. Do not use this project for production financial decisions."

MODEL_EXAMPLES = {
    "fraud": {
        "transaction_amount": 32500.0,
        "transaction_hour": 2,
        "is_weekend": 1,
        "merchant_category": "gift_card",
        "user_location": "Delhi",
        "device_type": "web",
        "previous_transaction_gap": 1.5,
        "transaction_velocity": 7,
        "failed_login_attempts": 4,
        "account_age_days": 12,
        "ip_risk_score": 91.0,
        "geo_mismatch_flag": 1,
        "unusual_device_flag": 1,
    },
    "legit": {
        "transaction_amount": 480.0,
        "transaction_hour": 14,
        "is_weekend": 0,
        "merchant_category": "grocery",
        "user_location": "Mumbai",
        "device_type": "mobile",
        "previous_transaction_gap": 240.0,
        "transaction_velocity": 1,
        "failed_login_attempts": 0,
        "account_age_days": 760,
        "ip_risk_score": 11.0,
        "geo_mismatch_flag": 0,
        "unusual_device_flag": 0,
    },
}
