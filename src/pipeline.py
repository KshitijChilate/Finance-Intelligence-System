print("THIS IS THE NEW PIPELINE FILE")

from .data_loader import load_data
from .feature_engineering import create_user_metrics
from .risk_scoring import compute_risk_score, create_risk_label
from .risk_model import train_risk_model
from .anomaly_detection import detect_anomalies
from .segmentation import perform_segmentation

import os


def run_pipeline(users_path, transactions_path):

    print("Loading data...")
    users, transactions = load_data(users_path, transactions_path)

    print("Creating user metrics...")
    user_metrics = create_user_metrics(users, transactions)

    print("Computing behavioral risk score...")
    user_metrics, risk_scaler = compute_risk_score(user_metrics)

    print("Creating synthetic risk labels...")
    user_metrics = create_risk_label(user_metrics)

    print("Training supervised risk model...")
    risk_model = train_risk_model(user_metrics)

    print("Detecting anomalies...")
    user_metrics = detect_anomalies(user_metrics)

    print("Performing customer segmentation...")
    user_metrics = perform_segmentation(user_metrics)

    os.makedirs("models", exist_ok=True)

    print("Pipeline execution completed successfully.")

    return user_metrics