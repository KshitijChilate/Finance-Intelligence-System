class FeatureEngineer:
    def __init__(self, transactions):
        self.transactions = transactions

    def calculate_average_spending(self):
        """Calculates average spending per category."""
        average_spending = self.transactions.groupby('category')['amount'].mean()
        return average_spending

    def identify_spending_trends(self):
        """Identifies trends in spending over time."""
        trends = self.transactions.groupby('date')['amount'].sum().pct_change()
        return trends

    def create_features(self):
        """Creates behavioral spending features."""
        features = {
            'average_spending': self.calculate_average_spending(),
            'trends': self.identify_spending_trends(),
        }
        return features


# 🔹 Add this function for pipeline compatibility
import pandas as pd

def create_user_metrics(users, transactions):
    metrics = transactions.groupby("user_id").agg(
        total_spent=("amount", "sum"),
        avg_transaction=("amount", "mean"),
        transaction_count=("amount", "count"),
        spend_std=("amount", "std")
    ).reset_index()

    metrics["spend_std"] = metrics["spend_std"].fillna(0)

    user_metrics = users.merge(metrics, on="user_id", how="left")
    
    user_metrics["income_expense_ratio"] = (
        user_metrics["total_spent"] / user_metrics["annual_income"].replace(0, 1)
    )

    return user_metrics