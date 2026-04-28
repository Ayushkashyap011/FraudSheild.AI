"""Train the FraudShield AI models and persist the best artifact.

The training flow keeps SMOTE inside the training split only, compares the
candidate models on a validation holdout, tunes the operating threshold with
precision-recall analysis, and saves a timestamped model artifact.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


warnings.filterwarnings("ignore")


def load_data() -> pd.DataFrame:
    """Load the fraud dataset and validate the required columns."""

    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {config.DATA_PATH}. Run src/generate_data.py first."
        )

    df = pd.read_csv(config.DATA_PATH)
    missing = [column for column in config.FEATURE_COLUMNS + [config.TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].copy()


def build_preprocessor() -> ColumnTransformer:
    """Create a reusable preprocessing block for numeric and categorical data."""

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Any]:
    """Return the candidate estimators to compare."""

    return {
        "Logistic Regression": LogisticRegression(**config.MODEL_SPECS["Logistic Regression"]),
        "Random Forest": RandomForestClassifier(**config.MODEL_SPECS["Random Forest"]),
        "XGBoost": XGBClassifier(**config.MODEL_SPECS["XGBoost"]),
    }


def build_pipeline(model: Any) -> ImbPipeline:
    """Create the full training pipeline with SMOTE only on the training split."""

    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute the main fraud metrics for a probability threshold."""

    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select the operating threshold that maximizes F1 from the PR curve."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = np.where(
        (precision + recall) > 0,
        (2 * precision * recall) / (precision + recall),
        0.0,
    )

    threshold_frame = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
        }
    )

    best_index = int(threshold_frame["f1"].idxmax())
    best_threshold = float(threshold_frame.loc[best_index, "threshold"])
    return best_threshold, threshold_frame


def build_feature_importance(pipeline: ImbPipeline) -> pd.DataFrame:
    """Extract a simple global importance table from the trained model."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        values = np.zeros(len(feature_names), dtype=float)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)

    return importance.reset_index(drop=True)


def serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy values into JSON-friendly types."""

    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


def main() -> None:
    print("FraudShield AI training started.\n")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.VALIDATION_SIZE,
        stratify=y_train_full,
        random_state=config.RANDOM_STATE,
    )

    comparison_rows = []
    fitted_candidates: dict[str, ImbPipeline] = {}

    for model_name, estimator in build_models().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        validation_metrics = evaluate_predictions(y_validation, validation_probabilities, config.DEFAULT_THRESHOLD)
        selection_score = validation_metrics["f1"] + validation_metrics["recall"]

        comparison_rows.append(
            {
                "model": model_name,
                "precision": round(validation_metrics["precision"], 4),
                "recall": round(validation_metrics["recall"], 4),
                "f1": round(validation_metrics["f1"], 4),
                "roc_auc": round(validation_metrics["roc_auc"], 4),
                "selection_score": round(selection_score, 4),
            }
        )
        fitted_candidates[model_name] = pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["selection_score", "f1", "recall"],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = str(comparison_df.loc[0, "model"])
    validation_pipeline = fitted_candidates[best_model_name]

    validation_probabilities = validation_pipeline.predict_proba(X_validation)[:, 1]
    best_threshold, threshold_curve = tune_threshold(y_validation, validation_probabilities)

    final_pipeline = build_pipeline(build_models()[best_model_name])
    final_pipeline.fit(X_train_full, y_train_full)

    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = evaluate_predictions(y_test, test_probabilities, best_threshold)

    timestamp = datetime.now().strftime("%Y%m%d")
    model_path = config.MODEL_DIR / f"{config.MODEL_PREFIX}_{timestamp}.pkl"
    metadata_path = model_path.with_suffix(".json")

    model_payload = {
        "pipeline": final_pipeline,
        "model_name": best_model_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": best_threshold,
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "comparison_table": comparison_df,
        "threshold_curve": threshold_curve,
        "feature_importance": build_feature_importance(final_pipeline),
        "background_data": X_train_full.sample(
            n=min(config.SHAP_BACKGROUND_SIZE, len(X_train_full)),
            random_state=config.RANDOM_STATE,
        ).reset_index(drop=True),
        "metrics": final_metrics,
    }
    joblib.dump(model_payload, model_path)

    metadata = {
        "model_name": best_model_name,
        "trained_at": model_payload["trained_at"],
        "threshold": best_threshold,
        "feature_columns": config.FEATURE_COLUMNS,
        "metrics": serialize_metrics(final_metrics),
        "comparison_table": comparison_df.to_dict(orient="records"),
        "feature_importance": model_payload["feature_importance"].head(15).to_dict(orient="records"),
        "threshold_curve_sample": threshold_curve.head(20).round(6).to_dict(orient="records"),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)

    print("Model comparison table (validation split):")
    print(comparison_df.to_string(index=False))
    print("\nBest model:", best_model_name)
    print("Saved model:", model_path)
    print("Saved metadata:", metadata_path)
    print("\nFinal test metrics:")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall   : {final_metrics['recall']:.4f}")
    print(f"F1       : {final_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {final_metrics['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(np.array(final_metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()