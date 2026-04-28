# FraudShield AI

FraudShield AI is a production-style fraud detection portfolio project. It trains multiple models, keeps SMOTE inside the training split only, tunes the operating threshold with precision-recall analysis, and serves predictions through a Streamlit dashboard with SHAP-backed explanations.

## What it does

The project compares Logistic Regression, Random Forest, and XGBoost on a synthetic transaction dataset. The best model is saved as a timestamped artifact, the optimal threshold is stored with the model, and the dashboard supports single-transaction scoring, batch CSV scoring, analytics, and analyst feedback logging.

## Model comparison

`src/train.py` prints and saves a model comparison table during training. After training, the results are available at `models/model_comparison.csv`.

| Model | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.6954 | 0.0987 | 0.7935 |
| Random Forest | 0.0761 | 0.0852 | 0.7530 |
| XGBoost | 0.0102 | 0.0190 | 0.7592 |

Best model: Logistic Regression

Final hold-out test metrics after threshold tuning:

- Precision: 0.1406
- Recall: 0.3147
- F1: 0.1944
- ROC-AUC: 0.8117

## Limitations

The dataset is synthetic and was created to mimic risky transaction behavior, not to represent a real bank or fintech fraud distribution. Thresholds, feature relationships, and feedback logging are suitable for a portfolio demonstration, but they are not a substitute for production governance, model monitoring, privacy review, or compliance controls.

## Screenshot

Add a dashboard screenshot here after launching the app.

## How to run

1. Install dependencies.

```powershell
python -m pip install -r requirements.txt
```

2. Generate the synthetic dataset if `data/fraud_transactions.csv` is missing.

```powershell
python src/generate_data.py
```

3. Train and save the best model artifact.

```powershell
python src/train.py
```

4. Launch the dashboard in one command.

```powershell
python -m streamlit run app/app.py
```

## Project structure

```text
fraudshield_ai/
├── data/fraud_transactions.csv
├── src/generate_data.py
├── src/train.py
├── src/predict.py
├── models/fraud_model_YYYYMMDD.pkl
├── app/app.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

The dashboard includes a synthetic-data disclaimer, live threshold adjustment, SHAP explanations for the current prediction, batch scoring, and feedback logging to `feedback.csv`.
