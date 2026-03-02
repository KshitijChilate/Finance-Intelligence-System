from fastapi import FastAPI
from pydantic import BaseModel
from src.risk_scoring import calculate_risk_score
import joblib
import pandas as pd
import os

app = FastAPI(title="Financial Intelligence API")


# -------- Load Models -------- #

risk_scaler = joblib.load("models/risk_scaler.pkl")
anomaly_model = joblib.load("models/anomaly_model.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")
segment_scaler = joblib.load("models/segment_scaler.pkl")


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

    # Risk Score
    df["risk_score"] = risk_scaler.transform(
        df[["credit_score", "employment_years",
            "income_expense_ratio", "spend_std"]]
    ).mean(axis=1) * 100

    # Anomaly Detection
    df["anomaly"] = anomaly_model.predict(
        df[["total_spent", "risk_score"]]
    )

    # Segmentation
    scaled_features = segment_scaler.transform(
        df[["annual_income", "total_spent", "risk_score"]]
    )

    df["cluster"] = kmeans_model.predict(scaled_features)

    return {
        "risk_score": float(df["risk_score"].iloc[0]),
        "anomaly": int(df["anomaly"].iloc[0]),
        "cluster": int(df["cluster"].iloc[0]),
    }
