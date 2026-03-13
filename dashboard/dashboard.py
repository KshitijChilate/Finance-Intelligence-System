import requests
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# Page Config
# ===============================

st.set_page_config(page_title="Financial Intelligence Dashboard", layout="wide")

st.title("💳 Financial Intelligence System Dashboard")

# ===============================
# Custom Styling
# ===============================

st.markdown("""
<style>

.metric-container {
    background-color: #0E1117;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #262730;
}

.metric-title {
    font-size: 16px;
    color: #9FA6B2;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #00C897;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# Load Models
# ===============================

model = joblib.load("models/risk_model.pkl")
explainer = shap.TreeExplainer(model)

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
# Tabs Layout
# ===============================

tabs = st.tabs([
    "Overview",
    "Risk Analysis",
    "Customer Segmentation",
    "Anomaly Detection",
    "Explainability",
    "Real-Time Prediction",
    "Data Insights"
])

# ===============================
# Overview
# ===============================

with tabs[0]:

    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
            <div class="metric-title">Total Customers</div>
            <div class="metric-value">{len(filtered_data)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        high_risk = len(filtered_data[filtered_data["risk_tier_y"] == "Critical"])

        st.markdown(
            f"""
            <div class="metric-container">
            <div class="metric-title">High Risk Customers</div>
            <div class="metric-value">{high_risk}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        anomalies = filtered_data["anomaly"].sum()

        st.markdown(
            f"""
            <div class="metric-container">
            <div class="metric-title">Anomalies Detected</div>
            <div class="metric-value">{anomalies}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Risk Tier Distribution")

    risk_counts = filtered_data["risk_tier_y"].value_counts().reset_index()
    risk_counts.columns = ["Risk Tier", "Count"]

    fig = px.bar(
        risk_counts,
        x="Risk Tier",
        y="Count",
        color="Risk Tier",
        text="Count",
        color_discrete_map={
            "Low": "#00C897",
            "Medium": "#FFC107",
            "High": "#FF4B4B",
            "Critical": "#D90429"
        }
    )

    fig.update_layout(template="plotly_dark", height=450)

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Risk Analysis
# ===============================

with tabs[1]:

    st.header("High Risk Customers")

    high_risk = filtered_data[filtered_data["risk_tier_y"] == "Critical"]

    st.dataframe(
        high_risk[
            ["user_id", "risk_score_y", "recommended_action", "cluster"]
        ].sort_values(by="risk_score_y", ascending=False),
        use_container_width=True
    )

# ===============================
# Customer Segmentation
# ===============================

with tabs[2]:

    st.header("Customer Segmentation")

    segment_counts = filtered_data["cluster"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    fig2 = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        hole=0.45
    )

    fig2.update_layout(template="plotly_dark", height=450)

    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# Anomaly Detection
# ===============================

with tabs[3]:

    st.header("Anomaly Detection")

    anomalies = filtered_data[filtered_data["anomaly"] == 1]

    st.dataframe(anomalies.head(20), use_container_width=True)

# ===============================
# SHAP Explainability
# ===============================

with tabs[4]:

    st.header("Explain Customer Risk (SHAP)")

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
# Real-Time Prediction
# ===============================

with tabs[5]:

    st.header("Real-Time Risk Prediction")

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

        try:

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

        except:

            st.error("Prediction API is not running. Start FastAPI server.")

# ===============================
# CSV Upload Insights
# ===============================

with tabs[6]:

    st.header("Upload Dataset for Automated Insights")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(uploaded_data.head(), use_container_width=True)

        if st.button("Generate Insights"):

            data_upload = uploaded_data.copy()

            data_upload["risk_score"] = (
                (1 - data_upload["credit_score"] / 850) * 50 +
                data_upload["income_expense_ratio"] * 30 +
                (data_upload["spend_std"] / data_upload["spend_std"].max()) * 20
            )

            data_upload["risk_tier"] = pd.cut(
                data_upload["risk_score"],
                bins=[-1, 40, 70, 100],
                labels=["Low", "Medium", "High"]
            )

            st.success("Insights Generated")

            st.dataframe(data_upload, use_container_width=True)

            # Histogram
            fig3 = px.histogram(
                data_upload,
                x="risk_score",
                color="risk_tier",
                template="plotly_dark"
            )

            st.plotly_chart(fig3, use_container_width=True)

            # Insight summary
            st.subheader("Key Insights")

            high_risk_count = len(data_upload[data_upload["risk_tier"] == "High"])

            st.write(f"Total Customers: {len(data_upload)}")
            st.write(f"High Risk Customers: {high_risk_count}")

            csv = data_upload.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Insights",
                csv,
                "financial_risk_insights.csv",
                "text/csv"
            )