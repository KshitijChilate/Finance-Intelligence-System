from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os


def train_risk_model(user_metrics):

    features = [
        "credit_score",
        "income_expense_ratio",
        "spend_std",
        "employment_years"
    ]

    X = user_metrics[features]
    y = user_metrics["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Risk Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/risk_model.pkl")

    return model