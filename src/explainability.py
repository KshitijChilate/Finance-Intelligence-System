import shap
import pandas as pd
import joblib

class RiskExplainer:
    def __init__(self, model_path="models/risk_model.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X):
        shap_values = self.explainer.shap_values(X)
        return shap_values