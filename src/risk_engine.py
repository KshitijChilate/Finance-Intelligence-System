# src/risk_engine.py

class RiskEngine:
    def __init__(self, model):
        self.model = model

    def score_transaction(self, transaction_df):
        """
        Returns probability score between 0 and 1
        """
        probability = self.model.predict_proba(transaction_df)[0][1]
        return probability

    def assign_risk_tier(self, score):
        """
        Assign tier based on probability score
        """
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Medium"
        elif score < 0.85:
            return "High"
        else:
            return "Critical"

    def evaluate(self, transaction_df):
        score = self.score_transaction(transaction_df)
        tier = self.assign_risk_tier(score)

        return {
            "risk_score": round(score, 4),
            "risk_tier": tier
        }