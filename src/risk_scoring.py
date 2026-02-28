from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def compute_risk_score(user_metrics):

    scaler = MinMaxScaler()

    features_to_scale = [
        "credit_score",
        "income_expense_ratio",
        "spend_std",
        "employment_years"
    ]

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

    return user_metrics, scaler
