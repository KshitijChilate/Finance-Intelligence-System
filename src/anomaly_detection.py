import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination='auto')

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        predictions = self.model.predict(X)
        return ['High-Risk' if pred == -1 else 'Low-Risk' for pred in predictions]