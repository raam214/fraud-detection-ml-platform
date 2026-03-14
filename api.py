from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

app = FastAPI(title="Fraud Detection API", version="1.0")


def train():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'amount': np.random.exponential(200, n),
        'hour': np.random.randint(0, 24, n),
        'day_of_week': np.random.randint(0, 7, n),
        'merchant_category': np.random.randint(0, 10, n),
        'distance_from_home': np.random.exponential(50, n),
        'velocity_24h': np.random.randint(1, 20, n),
        'device_score': np.random.uniform(0, 1, n),
        'location_risk': np.random.uniform(0, 1, n),
        'card_age_days': np.random.randint(1, 2000, n),
        'failed_attempts': np.random.randint(0, 5, n),
    })
    fraud_prob = (
        (df['amount'] > 500).astype(int) * 0.25 +
        (df['hour'] < 6).astype(int) * 0.2 +
        (df['distance_from_home'] > 100).astype(int) * 0.2 +
        (df['velocity_24h'] > 10).astype(int) * 0.15 +
        (df['failed_attempts'] > 2).astype(int) * 0.15 +
        (df['location_risk'] > 0.7).astype(int) * 0.1 +
        np.random.uniform(0, 0.15, n)
    )
    df['is_fraud'] = (fraud_prob > 0.6).astype(int)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=4,
                          learning_rate=0.1, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model


model = train()


class Transaction(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    merchant_category: int
    distance_from_home: float
    velocity_24h: int
    device_score: float
    location_risk: float
    card_age_days: int
    failed_attempts: int


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(txn: Transaction):
    data = np.array([[txn.amount, txn.hour, txn.day_of_week,
                      txn.merchant_category, txn.distance_from_home,
                      txn.velocity_24h, txn.device_score, txn.location_risk,
                      txn.card_age_days, txn.failed_attempts]])
    prob = model.predict_proba(data)[0][1]
    if prob < 0.3:
        decision = "APPROVE"
    elif prob < 0.6:
        decision = "REVIEW"
    else:
        decision = "BLOCK"
    return {
        "fraud_probability": round(float(prob), 4),
        "decision": decision,
        "risk_score": round(float(prob) * 100, 2)
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost"}
