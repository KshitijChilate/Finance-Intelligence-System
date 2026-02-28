import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os


def perform_segmentation(df):

    features = df[["annual_income", "total_spent", "risk_score"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(scaled_features)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/segment_scaler.pkl")

    return df
