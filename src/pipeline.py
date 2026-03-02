print("THIS IS THE NEW PIPELINE FILE")

from .data_loader import load_data
from .feature_engineering import create_user_metrics
from .risk_scoring import compute_risk_score, create_risk_label
from .risk_model import train_risk_model
from .anomaly_detection import detect_anomalies
from .segmentation import perform_segmentation
from .risk_engine import RiskEngine
from .alert_manager import AlertManager

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

    print("Initializing Risk Engine...")
    risk_engine = RiskEngine(risk_model)
    alert_manager = AlertManager()

    print("Detecting anomalies...")
    user_metrics = detect_anomalies(user_metrics)

    print("Performing customer segmentation...")
    user_metrics = perform_segmentation(user_metrics)

    os.makedirs("models", exist_ok=True)

    print("Evaluating risk tiers and actions...")

    risk_results = []

    model_features = [
    "credit_score",
    "income_expense_ratio",
    "spend_std",
    "employment_years"
    ]

    for _, row in user_metrics.iterrows():
        transaction_df = row[model_features].to_frame().T in ["risk_label"]


        result = risk_engine.evaluate(transaction_df)
        action = alert_manager.generate_action(result["risk_tier"])

        risk_results.append({
            "risk_score": result["risk_score"],
            "risk_tier": result["risk_tier"],
            "recommended_action": action
        })

    risk_df = user_metrics.reset_index(drop=True)
    risk_details_df = user_metrics[["user_id"]].reset_index(drop=True)

    for key in ["risk_score", "risk_tier", "recommended_action"]:
        risk_details_df[key] = [r[key] for r in risk_results]

    user_metrics = user_metrics.merge(risk_details_df, on="user_id", how="left")

    print("Pipeline execution completed successfully.")

    return user_metrics

