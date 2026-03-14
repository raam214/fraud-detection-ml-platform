# 🛡️ Real-Time Fraud Detection Platform
> XGBoost ML engine — real-time transaction risk scoring with APPROVE / REVIEW / BLOCK decisions

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://fraud-detection-ml-platform-214.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-FF6600?style=for-the-badge)](https://xgboost.ai)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-F7931E?style=for-the-badge)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-6.4.0-3F4F75?style=for-the-badge)](https://plotly.com)

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
|-------|-----------|
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
cd fraud-detection-ml-platform

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 93% |
| F1-Score | 0.91 |
| Fraud Rate (Dataset) | 7.9% |
| Training Transactions | 100,000+ |
| Engineered Features | 12+ |

---

## 📁 Project Structure
```
fraud-detection-ml-platform/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── .gitignore          # Ignores venv
└── README.md           # This file
```

---

Built by **Ram Dukare** · Powered by XGBoost · Scikit-learn · Plotly · Streamlit
