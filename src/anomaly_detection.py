import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os


def detect_anomalies(df):

    model = IsolationForest(contamination=0.05, random_state=42)

    df["anomaly"] = model.fit_predict(df[["total_spent", "risk_score"]])

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/anomaly_model.pkl")

    return df
