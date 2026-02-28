from .data_loader import load_data
from .feature_engineering import create_user_metrics
from .risk_scoring import compute_risk_score
from .anomaly_detection import detect_anomalies
from .segmentation import perform_segmentation

import joblib
import os


def run_pipeline(users_path, transactions_path):

    users, transactions = load_data(users_path, transactions_path)

    user_metrics = create_user_metrics(users, transactions)

    user_metrics, risk_scaler = compute_risk_score(user_metrics)

    user_metrics = detect_anomalies(user_metrics)

    user_metrics = perform_segmentation(user_metrics)

    os.makedirs("models", exist_ok=True)

    return user_metrics
