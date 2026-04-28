import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# ==========================================
# CONFIG
# ==========================================
DATA_PATH = "data/fraud_transactions.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "is_fraud"
RANDOM_STATE = 42


# ==========================================
# LOAD DATA
# ==========================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Dataset Loaded:", df.shape)
    return df


# ==========================================
# PREPROCESSOR
# ==========================================
def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return preprocessor


# ==========================================
# MODELS
# ==========================================
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


# ==========================================
# EVALUATE MODEL
# ==========================================
def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor):

    pipeline = ImbPipeline(steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1 Score  :", round(f1, 4))
    print("ROC AUC   :", round(roc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline, recall, precision, f1, roc


# ==========================================
# MAIN TRAINING
# ==========================================
def main():

    print("FraudShield AI Training Started...\n")

    df = load_data()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)

    models = get_models()

    best_model = None
    best_score = 0
    best_name = ""

    results = []

    for name, model in models.items():

        trained_model, recall, precision, f1, roc = evaluate_model(
            name, model,
            X_train, X_test,
            y_train, y_test,
            preprocessor
        )

        results.append({
            "Model": name,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "ROC_AUC": roc
        })

        # Choose best model by Recall first, then ROC
        score = (recall * 0.7) + (roc * 0.3)

        if score > best_score:
            best_score = score
            best_model = trained_model
            best_name = name

    # Save Best Model
    model_path = MODEL_DIR / "fraud_model.pkl"
    joblib.dump(best_model, model_path)

    # Save Results
    result_df = pd.DataFrame(results)
    result_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

    print("\n" + "#" * 60)
    print("BEST MODEL SELECTED:", best_name)
    print("Saved To:", model_path)
    print("#" * 60)

    print("\nModel Comparison:")
    print(result_df.sort_values(by="Recall", ascending=False))


if __name__ == "__main__":
    main()
