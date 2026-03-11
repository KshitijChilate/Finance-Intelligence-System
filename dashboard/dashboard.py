import requests
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# Load Models
# ===============================

model = joblib.load("models/risk_model.pkl")
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Financial Intelligence Dashboard", layout="wide")

st.title("💳 Financial Intelligence System Dashboard")

# ===============================
# Load Prediction Results
# ===============================

try:
    data = pd.read_csv("results/predictions.csv")
except:
    st.warning("Prediction results not found. Run pipeline first.")
    st.stop()

# ===============================
# Sidebar Filters
# ===============================

st.sidebar.header("Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Tier",
    options=data["risk_tier_y"].unique(),
    default=data["risk_tier_y"].unique()
)

filtered_data = data[data["risk_tier_y"].isin(risk_filter)]

# ===============================
# KPI Metrics
# ===============================

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_data))

col2.metric(
    "High Risk Customers",
    len(filtered_data[filtered_data["risk_tier_y"] == "Critical"])
)

col3.metric(
    "Anomalies Detected",
    filtered_data["anomaly"].sum()
)

st.divider()

# ===============================
# Risk Distribution
# ===============================

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

# ===============================
# Customer Segmentation
# ===============================

st.subheader("Customer Segments")

segment_counts = filtered_data["cluster"].value_counts().reset_index()
segment_counts.columns = ["Segment", "Count"]

fig2 = px.pie(
    segment_counts,
    names="Segment",
    values="Count"
)

st.plotly_chart(fig2, use_container_width=True)

# ===============================
# High Risk Customers Table
# ===============================

st.subheader("High Risk Customers")

high_risk = filtered_data[filtered_data["risk_tier_y"] == "Critical"]

st.dataframe(
    high_risk[
        ["user_id", "risk_score_y", "recommended_action", "cluster"]
    ].sort_values(by="risk_score_y", ascending=False)
)

# ===============================
# Anomaly Detection
# ===============================

st.subheader("Anomaly Detection")

anomalies = filtered_data[filtered_data["anomaly"] == 1]

st.dataframe(anomalies.head(20))

# ===============================
# SHAP Explainability
# ===============================

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

# ===============================
# Real-Time API Prediction
# ===============================

st.divider()
st.subheader("Real-Time Risk Prediction (API)")

age = st.number_input("Age", 18, 100, 35)
credit_score = st.number_input("Credit Score", 300, 900, 650)
employment_years = st.number_input("Employment Years", 0, 40, 5)
income_expense_ratio = st.number_input("Income Expense Ratio", 0.0, 2.0, 0.5)
spend_std = st.number_input("Spending Std Dev", 0.0, 5000.0, 1000.0)
annual_income = st.number_input("Annual Income", 0, 10000000, 1500000)
total_spent = st.number_input("Total Spent", 0, 10000000, 800000)

if st.button("Predict Risk"):

    payload = {
        "credit_score": credit_score,
        "employment_years": employment_years,
        "income_expense_ratio": income_expense_ratio,
        "spend_std": spend_std,
        "annual_income": annual_income,
        "total_spent": total_spent
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    st.success("Prediction Complete")

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", round(result["risk_score"], 2))
    col2.metric("Risk Tier", result["risk_tier"])
    col3.metric("Cluster", result["cluster"])

    st.write("Recommended Action:", result["recommended_action"])

# ===============================
# CSV Upload Insight Generator
# ===============================

st.divider()
st.header("📂 Upload Customer Dataset for Automated Insights")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing customer financial data",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(uploaded_data.head())

    if st.button("Generate Insights"):

        data_upload = uploaded_data.copy()

        # Simple risk estimation
        data_upload["risk_score"] = (
            (1 - data_upload["credit_score"] / 850) * 50 +
            data_upload["income_expense_ratio"] * 30 +
            (data_upload["spend_std"] / data_upload["spend_std"].max()) * 20
        )

        # Risk tiers
        data_upload["risk_tier"] = pd.cut(
            data_upload["risk_score"],
            bins=[-1, 40, 70, 100],
            labels=["Low", "Medium", "High"]
        )

        st.success("Insights Generated Successfully")

        st.subheader("Processed Dataset")
        st.dataframe(data_upload)

        # Risk Distribution
        st.subheader("Risk Score Distribution")

        fig3 = px.histogram(
            data_upload,
            x="risk_score",
            nbins=20,
            title="Risk Score Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Insight summary
        st.subheader("Key Insights")

        high_risk_count = len(data_upload[data_upload["risk_tier"] == "High"])

        st.write(f"High Risk Customers: {high_risk_count}")
        st.write(f"Total Customers: {len(data_upload)}")