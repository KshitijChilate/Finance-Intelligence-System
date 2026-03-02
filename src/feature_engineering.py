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
