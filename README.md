# 🛡️ Real-Time Fraud Detection Platform

LLM-powered fraud detection platform built with XGBoost, Plotly, and Streamlit

![LIVE APP](https://img.shields.io/badge/LIVE%20APP-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![PYTHON](https://img.shields.io/badge/PYTHON-3.x-3776AB?style=flat-square&logo=python)
![XGBOOST](https://img.shields.io/badge/XGBOOST-3.2.0-FF6600?style=flat-square)
![SCIKIT-LEARN](https://img.shields.io/badge/SCIKIT--LEARN-1.8.0-F7931E?style=flat-square)
![PLOTLY](https://img.shields.io/badge/PLOTLY-6.4.0-3F4F75?style=flat-square)

---

## 🚀 What It Does

Enter transaction details → get instant ML-powered fraud risk decision with full explainability:

- 🔍 **Transaction Analyser** — real-time APPROVE / REVIEW / BLOCK with fraud probability score
- 📊 **Risk Dashboard** — KPIs: total transactions, fraud detected, fraud rate, safe transactions
- 🤖 **Model Performance** — XGBoost metrics, confusion matrix, feature importance
- ⚙️ **Risk Factor Analysis** — 7 risk flags per transaction (High Amount, Odd Hours, Far from Home, High Velocity, Failed Attempts, Location Risk, Low Device Trust)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Model | XGBoost + SMOTE |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly |
| Frontend | Streamlit |

---

## ⚙️ How It Works

1. Transaction features engineered from raw input (amount, hour, location, velocity, device score)
2. XGBoost trained on 100K+ synthetic financial transactions with SMOTE class balancing
3. Model outputs fraud probability → mapped to APPROVE / REVIEW / BLOCK decision
4. Rule-based risk factors evaluated alongside ML prediction for full explainability
5. Dashboard visualises transaction distribution, fraud rate by hour, and feature importance

---

## 🏃 Run Locally
```bash
# Clone the repo
git clone https://github.com/raam214/fraud-detection-ml-platform

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 93% |
| F1-Score | 0.91 |
| Fraud Rate (Dataset) | 7.9% |
| Training Transactions | 100,000+ |
| Engineered Features | 12+ |
