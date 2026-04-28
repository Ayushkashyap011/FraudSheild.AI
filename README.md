# 🛡️ FraudShield AI

An end-to-end Machine Learning project for **fraud detection and risk scoring** in financial transactions. FraudShield AI predicts suspicious transactions in real time, assigns probability-based risk levels, and provides an analyst-friendly dashboard for decision-making.

---

## 🚀 Live Value Proposition

Banks and fintech companies lose millions due to fraudulent activity. Traditional rule-based systems struggle to adapt to evolving attack patterns.

**FraudShield AI** uses Machine Learning to:

* Detect fraudulent transactions in real time
* Score transaction risk (Low / Medium / High)
* Explain suspicious signals
* Help analysts investigate faster
* Reduce financial losses and false alerts

---

## 📌 Features

### 🤖 Machine Learning Engine

* Synthetic fintech transaction dataset with realistic fraud behavior
* Handles imbalanced classes using SMOTE
* Trained and compared multiple models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Best model selected using fraud-focused metrics

### 📊 Risk Intelligence Dashboard

Built with Streamlit:

* Real-time fraud prediction
* Fraud probability %
* Risk meter
* Confidence score
* Suspicious factor alerts
* Fraud analytics charts
* Clean premium UI

### 🧠 Explainability

* Human-readable suspicious indicators
* Feature impact style explanation panel

---

## 🧰 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Plotly
* Streamlit
* Joblib

---

## 📁 Project Structure

```text
fraudshield_ai/
│── data/
│   └── fraud_transactions.csv
│── src/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
│── models/
│   └── fraud_model.pkl
│── app/
│   └── app.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Run

### 1. Clone Repo

```bash
git clone <your-repo-url>
cd FraudShield-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Dataset

```bash
python src/generate_data.py
```

### 4. Train Model

```bash
python src/train.py
```

### 5. Launch Dashboard

```bash
streamlit run app/app.py
```

---

## 📈 Model Performance

Best model selected: **Logistic Regression**

Example metrics:

* Recall: **71%**
* ROC-AUC: **81%**
* Fraud-focused optimization to catch maximum suspicious transactions

> In fraud systems, Recall is critical because missing fraud means direct financial loss.

---

## 🧪 Sample Prediction Output

```text
Prediction: Fraud
Fraud Probability: 99.99%
Risk Level: High
Confidence: 99.99%
```

Suspicious Factors:

* High transaction amount
* Risky IP address
* Multiple failed logins
* Geo mismatch
* Unusual device detected

---

## 💼 Resume Impact

**Built FraudShield AI, an end-to-end fraud detection platform using Machine Learning and Streamlit that predicts suspicious financial transactions, assigns risk scores, and provides explainable fraud insights through an interactive dashboard.**

---

## 🔮 Future Enhancements

* SHAP explainability integration
* Threshold tuning controls
* Real-time API with FastAPI
* Analyst feedback retraining loop
* Live database transaction monitoring
* Cloud deployment with CI/CD

---

## 👨‍💻 Author

**Kashyap**
Built as a portfolio-grade Data Science / ML Engineering project.

---

## ⭐ If You Like This Project

Give it a star on GitHub and share feedback.
