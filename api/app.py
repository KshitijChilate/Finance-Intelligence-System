from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Financial Intelligence API")

# -------- Load Models -------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

risk_scaler = joblib.load(os.path.join(BASE_DIR, "models", "risk_scaler.pkl"))
anomaly_model = joblib.load(os.path.join(BASE_DIR, "models", "anomaly_model.pkl"))
kmeans_model = joblib.load(os.path.join(BASE_DIR, "models", "kmeans_model.pkl"))
segment_scaler = joblib.load(os.path.join(BASE_DIR, "models", "segment_scaler.pkl"))


# -------- Request Schema -------- #

class UserInput(BaseModel):
    credit_score: float
    employment_years: float
    income_expense_ratio: float
    spend_std: float
    annual_income: float
    total_spent: float


# -------- Root Endpoint -------- #

@app.get("/")
def home():
    return {"message": "Financial Intelligence API is running"}


# -------- Prediction Endpoint -------- #

@app.post("/predict")
def predict(customer: UserInput):

    df = pd.DataFrame([customer.dict()])

    # -------- Risk Score -------- #
    features = [
        "credit_score",
        "income_expense_ratio",
        "spend_std",
        "employment_years"
    ]

    scaled = risk_scaler.transform(df[features])

    # compute risk score
    score = float(scaled.mean() * 100)

    # clamp score between 0–100
    score = max(0.0, min(100.0, score))

    df["risk_score"] = score

    # -------- Risk Tier -------- #

    score = df["risk_score"].iloc[0]

    if score > 80:
        risk_tier = "Critical"
        action = "Escalate to Manual Review"
    elif score > 65:
        risk_tier = "High"
        action = "Flag for Monitoring"
    elif score > 40:
        risk_tier = "Medium"
        action = "Monitor Activity"
    else:
        risk_tier = "Low"
        action = "No Action Needed"

    # -------- Anomaly Detection -------- #

    anomaly = anomaly_model.predict(
        df[["total_spent", "risk_score"]]
    )[0]

    anomaly_flag = 1 if anomaly == -1 else 0

    # -------- Customer Segmentation -------- #

    scaled_features = segment_scaler.transform(
        df[["annual_income", "total_spent", "risk_score"]]
    )

    cluster = int(kmeans_model.predict(scaled_features)[0])

    # -------- Response -------- #

    return {
        "risk_score": float(score),
        "risk_tier": risk_tier,
        "recommended_action": action,
        "anomaly": anomaly_flag,
        "cluster": cluster
    }