import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import joblib


def compute_risk_score(user_metrics):

    features_to_scale = [
        "credit_score",
        "income_expense_ratio",
        "spend_std",
        "employment_years"
    ]

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(user_metrics[features_to_scale])

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[f"{col}_scaled" for col in features_to_scale]
    )

    user_metrics = pd.concat([user_metrics, scaled_df], axis=1)

    user_metrics["risk_score"] = (
        0.35 * (1 - user_metrics["credit_score_scaled"]) +
        0.30 * (1 - user_metrics["income_expense_ratio_scaled"]) +
        0.20 * user_metrics["spend_std_scaled"] +
        0.15 * (1 - user_metrics["employment_years_scaled"])
    )

    user_metrics["risk_score"] *= 100

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/risk_scaler.pkl")

    return user_metrics, scaler

def create_risk_label(user_metrics):

    high_threshold = user_metrics["risk_score"].quantile(0.75)
    low_threshold = user_metrics["risk_score"].quantile(0.40)

    # Binary label
    user_metrics["risk_label"] = (
        user_metrics["risk_score"] >= high_threshold
    ).astype(int)

    conditions = [
        user_metrics["risk_score"] < low_threshold,
        (user_metrics["risk_score"] >= low_threshold) &
        (user_metrics["risk_score"] < high_threshold),
        user_metrics["risk_score"] >= high_threshold
    ]

    choices = ["Low", "Medium", "High"]

    user_metrics["risk_tier"] = np.select(
        conditions,
        choices,
        default="Medium"
    )

    print("\nRisk Tier Distribution:")
    print(user_metrics["risk_tier"].value_counts())

    return user_metrics

