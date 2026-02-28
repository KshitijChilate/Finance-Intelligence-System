# 💰 Financial Intelligence System

An end-to-end modular **Financial Risk Intelligence System** that performs:

- Risk Scoring  
- Anomaly Detection  
- Customer Segmentation  
- Production-ready ML pipeline orchestration  

Built with **Python, Scikit-Learn**, and clean modular architecture.

---

## 📌 Project Overview

This system analyzes customer financial data to:

- Engineer behavioral spending features  
- Calculate normalized risk scores  
- Detect anomalous (high-risk) customers  
- Segment customers into strategic clusters  
- Persist trained ML models for deployment  

The project follows **production-level architecture principles** with modular components and model persistence.

---

## 🏗 Architecture

```text
Finance-Intelligence-System
├── data/                     # Raw CSV data
├── models/                   # Saved ML models (gitignored)
├── notebooks/                # Data generation & exploration
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── risk_scoring.py
│   ├── anomaly_detection.py
│   ├── segmentation.py
│   └── pipeline.py
├── main.py                   # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- Python 3.11  
- Pandas  
- NumPy  
- Scikit-Learn  
- Joblib  
- Git & GitHub  
- VS Code  

---

## 🔄 Pipeline Flow

1. Load user & transaction data  
2. Perform feature engineering  
3. Calculate risk score using scaling  
4. Detect anomalies using Isolation Forest  
5. Segment customers using KMeans clustering  
6. Save trained models for deployment  

---

## 📊 Machine Learning Components

### 🔹 Risk Scoring
- MinMaxScaler  
- Financial behavior normalization  

### 🔹 Anomaly Detection
- Isolation Forest  
- Detects unusual high-risk behavior  

### 🔹 Customer Segmentation
- KMeans Clustering  
- 3 customer groups based on:
  - Annual Income  
  - Total Spend  
  - Risk Score  

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd Finance-Intelligence-System
```
2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run Pipeline
```bash
python -m main
📈 Sample Output
Pipeline executed successfully.

   user_id  total_spent  risk_score  anomaly  cluster
0        1    852118.09    52.94        1        0
```
...
## 🎯 Project Phases Completed
✅ Phase 1 — Environment Setup

✅ Phase 2 — Feature Engineering

✅ Phase 3 — Risk Scoring

✅ Phase 4 — Anomaly Detection

✅ Phase 5 — Customer Segmentation

✅ Production Refactor (Modular Architecture)

## 🚀 Upcoming Enhancements
FastAPI deployment layer

Streamlit dashboard

Docker containerization

Model monitoring

CI/CD integration

## 👨‍💻 Author
Kshitij Chilate
Data Science Student
Finance & Risk Analytics Enthusiast
