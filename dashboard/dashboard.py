import shap
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

model = joblib.load("models/risk_model.pkl")
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Financial Intelligence Dashboard", layout="wide")

st.title("💳 Financial Intelligence System Dashboard")

# Load predictions
data = pd.read_csv("results/predictions.csv")

st.sidebar.header("Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Tier",
    options=data["risk_tier_y"].unique(),
    default=data["risk_tier_y"].unique()
)

filtered_data = data[data["risk_tier_y"].isin(risk_filter)]

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_data))
col2.metric(
    "High Risk Customers",
    len(filtered_data[filtered_data["risk_tier_y"] == "Critical"])
)
col3.metric("Anomalies Detected", filtered_data["anomaly"].sum())

st.divider()

# Risk Distribution
st.subheader("Risk Tier Distribution")

risk_counts = filtered_data["risk_tier_y"].value_counts().reset_index()
risk_counts.columns = ["Risk Tier", "Count"]

fig = px.bar(
    risk_counts,
    x="Risk Tier",
    y="Count",
    color="Risk Tier",
)

st.plotly_chart(fig, use_container_width=True)

# Customer Segmentation
st.subheader("Customer Segments")

segment_counts = filtered_data["cluster"].value_counts().reset_index()
segment_counts.columns = ["Segment", "Count"]

fig2 = px.pie(
    segment_counts,
    names="Segment",
    values="Count"
)

st.plotly_chart(fig2, use_container_width=True)

# High Risk Customers
st.subheader("High Risk Customers")

high_risk = filtered_data[filtered_data["risk_tier_y"] == "Critical"]

st.dataframe(
    high_risk[
        ["user_id", "risk_score_y", "recommended_action", "cluster"]
    ].sort_values(by="risk_score_y", ascending=False)
)

# Anomaly Transactions
st.subheader("Anomaly Detection")

anomalies = filtered_data[filtered_data["anomaly"] == 1]

st.dataframe(anomalies.head(20))

# =========================
# SHAP Explainability
# =========================

st.subheader("Explain Customer Risk (SHAP)")

customer_id = st.selectbox(
    "Select Customer",
    data["user_id"].unique()
)

customer_data = data[data["user_id"] == customer_id]

col1, col2 = st.columns(2)

col1.metric(
    "Risk Score",
    round(customer_data["risk_score_y"].values[0], 3)
)

col2.metric(
    "Risk Tier",
    customer_data["risk_tier_y"].values[0]
)

features = [
    "age",
    "annual_income",
    "credit_score",
    "employment_years",
    "total_spent",
    "income_expense_ratio"
]

X = customer_data[features]

shap_values = explainer(X)

fig, ax = plt.subplots()

shap.plots.waterfall(
    shap_values[0, :, 1],
    max_display=len(features),
    show=False
)

st.pyplot(fig)