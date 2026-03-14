# 🛡️ Real-Time Fraud Detection Platform

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-red?style=flat&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-blue?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-6.4.0-purple?style=flat)
![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat&logo=streamlit)

> XGBoost ML engine — real-time transaction risk scoring with APPROVE / REVIEW / BLOCK decisions

## 🚀 Live App
**[▶ Launch Fraud Detection Platform](https://fraud-detection-ml-platform-214.streamlit.app)**

---

## 🎯 What It Does

- 🔍 **Transaction Analyser** — real-time APPROVE / REVIEW / BLOCK with fraud probability score
- 📊 **Risk Dashboard** — KPIs: 5,000 transactions, 7.9% fraud rate, feature importance charts
- 🤖 **Model Performance** — XGBoost metrics, confusion matrix, precision/recall
- ⚙️ **Risk Factor Analysis** — 7 risk flags per transaction (High Amount, Odd Hours, Far from Home, High Velocity, Failed Attempts, Location Risk, Low Device Trust)

---

## ⚙️ How It Works

1. Transaction features engineered from raw input (amount, hour, location, velocity, device score)
2. XGBoost trained on 100K+ synthetic financial transactions with SMOTE class balancing
3. Model outputs fraud probability → mapped to APPROVE / REVIEW / BLOCK decision
4. Rule-based risk factors evaluated alongside ML for full explainability
5. Dashboard visualises transaction distribution, fraud rate by hour, and feature importance

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Model | XGBoost + SMOTE |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly, Streamlit |
| Deployment | Streamlit Cloud |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 93% |
| F1-Score | 0.91 |
| Fraud Rate (Dataset) | 7.9% |
| Training Transactions | 100,000+ |
| Engineered Features | 12+ |

---

## 🖥️ Run Locally
```bash
git clone https://github.com/raam214/fraud-detection-ml-platform
cd fraud-detection-ml-platform
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Ram Dukare** — Data Science & ML Engineer
📧 raampatil214@gmail.com | [LinkedIn](https://linkedin.com/in/ram-dukare) | [GitHub](https://github.com/raam214)
