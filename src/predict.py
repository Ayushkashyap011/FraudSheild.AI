"""Prediction and SHAP explanation helpers for FraudShield AI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


def get_risk_level(probability: float) -> str:
    """Convert a fraud probability into a business-friendly band."""

    if probability < config.RISK_LEVELS["Low"]:
        return "Low"
    if probability < config.RISK_LEVELS["Medium"]:
        return "Medium"
    return "High"


def _candidate_model_paths() -> list[Path]:
    timestamped = sorted(config.MODEL_DIR.glob(f"{config.MODEL_PREFIX}_*.pkl"))
    legacy = config.MODEL_DIR / "fraud_model.pkl"
    candidates = list(reversed(timestamped))
    if legacy.exists():
        candidates.append(legacy)
    return candidates


def _normalize_artifact(raw_artifact: Any) -> dict[str, Any]:
    """Support both the upgraded artifact dictionary and legacy pipelines."""

    if isinstance(raw_artifact, dict) and "pipeline" in raw_artifact:
        artifact = dict(raw_artifact)
    else:
        artifact = {"pipeline": raw_artifact}

    artifact.setdefault("threshold", config.DEFAULT_THRESHOLD)
    artifact.setdefault("model_name", "Legacy artifact")
    artifact.setdefault("trained_at", "unknown")
    artifact.setdefault("feature_columns", config.FEATURE_COLUMNS)
    artifact.setdefault("background_data", pd.DataFrame(columns=config.FEATURE_COLUMNS))
    artifact.setdefault("feature_importance", pd.DataFrame(columns=["feature", "importance"]))
    return artifact


@lru_cache(maxsize=2)
def load_latest_model_artifact() -> tuple[dict[str, Any], Path]:
    """Load the newest saved model artifact from disk."""

    for model_path in _candidate_model_paths():
        if model_path.exists():
            return _normalize_artifact(joblib.load(model_path)), model_path
    raise FileNotFoundError(
        f"No trained model found in {config.MODEL_DIR}. Run src/train.py first."
    )


def _validate_input(input_data: dict[str, Any]) -> pd.DataFrame:
    """Ensure the user payload matches the model's expected columns."""

    missing = [column for column in config.FEATURE_COLUMNS if column not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return pd.DataFrame(
        [[input_data[column] for column in config.FEATURE_COLUMNS]],
        columns=config.FEATURE_COLUMNS,
    )


def _find_transformer_step(pipeline: Any) -> Any | None:
    """Find the preprocessing step in both current and legacy pipelines."""

    if not hasattr(pipeline, "named_steps"):
        return None

    preferred_names = (
        "preprocessor",
        "prep",
        "preprocess",
        "column_transformer",
        "transformer",
    )
    for step_name in preferred_names:
        step = pipeline.named_steps.get(step_name)
        if step is not None and hasattr(step, "transform"):
            return step

    for step_name, step in pipeline.named_steps.items():
        if step_name in {"smote", "model"}:
            continue
        if hasattr(step, "transform"):
            return step

    return None


def _fallback_explanations(artifact: dict[str, Any], input_df: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic fallback when SHAP cannot be computed."""

    feature_importance = artifact.get("feature_importance")
    if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
        candidate_features = feature_importance.head(config.TOP_SHAP_FEATURES)["feature"].tolist()
    else:
        candidate_features = list(input_df.columns[: config.TOP_SHAP_FEATURES])

    rows: list[dict[str, Any]] = []
    for feature_name in candidate_features:
        if feature_name in input_df.columns:
            value = input_df.iloc[0][feature_name]
        elif "__" in feature_name:
            value = input_df.iloc[0].get(feature_name.split("__", 1)[-1], "n/a")
        else:
            value = "n/a"

        rows.append(
            {
                "feature": feature_name,
                "shap_value": 0.0,
                "direction": f"Fallback explanation for {value}",
            }
        )

    return pd.DataFrame(rows)


def _build_explainer(artifact: dict[str, Any]):
    """Create a SHAP explainer for the transformed feature space."""

    pipeline = artifact["pipeline"]
    transformer = _find_transformer_step(pipeline)
    model = pipeline.named_steps["model"] if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps else pipeline

    background = artifact.get("background_data")
    if not isinstance(background, pd.DataFrame) or background.empty:
        background = pd.DataFrame(columns=config.FEATURE_COLUMNS)

    if transformer is not None:
        transformed_background = transformer.transform(background)

        def predict_positive(transformed_rows: Any) -> np.ndarray:
            return model.predict_proba(transformed_rows)[:, 1]

        explainer = shap.Explainer(predict_positive, transformed_background)
        return explainer, transformer, True

    def predict_positive(raw_rows: Any) -> np.ndarray:
        return pipeline.predict_proba(raw_rows)[:, 1]

    explainer = shap.Explainer(predict_positive, background)
    return explainer, None, False


@lru_cache(maxsize=2)
def get_explainer_cached(model_path_str: str):
    artifact, _ = load_latest_model_artifact()
    return _build_explainer(artifact)


def explain_transaction(input_df: pd.DataFrame) -> pd.DataFrame:
    """Return the top SHAP features and their direction for one transaction."""

    artifact, model_path = load_latest_model_artifact()

    try:
        explainer, transformer, uses_transformed_space = get_explainer_cached(str(model_path))
        if uses_transformed_space and transformer is not None:
            explanation = explainer(transformer.transform(input_df))
            feature_names = list(transformer.get_feature_names_out())
        else:
            explanation = explainer(input_df)
            feature_names = list(input_df.columns)

        if getattr(explanation, "values", None) is None:
            return _fallback_explanations(artifact, input_df)

        shap_values = np.asarray(explanation.values).reshape(-1)
        feature_frame = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": shap_values,
            }
        )
        feature_frame["direction"] = np.where(
            feature_frame["shap_value"] >= 0,
            "Increases fraud risk",
            "Decreases fraud risk",
        )
        feature_frame["abs_shap"] = feature_frame["shap_value"].abs()
        return feature_frame.sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"]).head(config.TOP_SHAP_FEATURES).reset_index(drop=True)
    except Exception:
        return _fallback_explanations(artifact, input_df)


def predict_transaction(input_data: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
    """Score one transaction and attach SHAP explanations."""

    artifact, model_path = load_latest_model_artifact()
    transaction_df = _validate_input(input_data)
    probability = float(artifact["pipeline"].predict_proba(transaction_df)[:, 1][0])
    applied_threshold = float(threshold if threshold is not None else artifact.get("threshold", config.DEFAULT_THRESHOLD))
    prediction = int(probability >= applied_threshold)
    explanation_frame = explain_transaction(transaction_df)

    return {
        "model_path": str(model_path),
        "model_name": artifact.get("model_name", "Unknown"),
        "trained_at": artifact.get("trained_at", "unknown"),
        "threshold": applied_threshold,
        "prediction": "Fraud" if prediction == 1 else "Legit",
        "fraud_probability": round(probability * 100, 2),
        "risk_level": get_risk_level(probability),
        "model_confidence": round(max(probability, 1 - probability) * 100, 2),
        "shap_explanations": explanation_frame.to_dict(orient="records"),
        "shap_table": explanation_frame,
    }


def predict_batch(input_frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Score a batch of transactions from a CSV upload or dataframe."""

    artifact, _ = load_latest_model_artifact()
    missing = [column for column in config.FEATURE_COLUMNS if column not in input_frame.columns]
    if missing:
        raise ValueError(f"Batch file is missing required columns: {missing}")

    scored = input_frame.copy()[config.FEATURE_COLUMNS]
    probabilities = artifact["pipeline"].predict_proba(scored)[:, 1]
    applied_threshold = float(threshold if threshold is not None else artifact.get("threshold", config.DEFAULT_THRESHOLD))
    scored["fraud_probability"] = probabilities
    scored["fraud_probability_pct"] = (scored["fraud_probability"] * 100).round(2)
    scored["prediction"] = np.where(scored["fraud_probability"] >= applied_threshold, "Fraud", "Legit")
    scored["risk_level"] = scored["fraud_probability"].apply(get_risk_level)
    scored["threshold"] = applied_threshold
    return scored


def get_model_summary() -> dict[str, Any]:
    """Expose the active model metadata for the Streamlit sidebar."""

    artifact, model_path = load_latest_model_artifact()
    return {
        "model_name": artifact.get("model_name", "Unknown"),
        "trained_at": artifact.get("trained_at", "unknown"),
        "threshold": float(artifact.get("threshold", config.DEFAULT_THRESHOLD)),
        "model_path": str(model_path),
        "feature_importance": artifact.get("feature_importance"),
        "comparison_table": artifact.get("comparison_table"),
        "metrics": artifact.get("metrics", {}),
    }


if __name__ == "__main__":
    example = config.MODEL_EXAMPLES["fraud"]
    result = predict_transaction(example)
    print(result)