import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load financial transaction data from a CSV file."""
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
            return data
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return None

    def filter_data(self, column_name, value):
        """Filter the data based on a specific column and value."""
        data = self.load_data()
        if data is not None:
            filtered_data = data[data[column_name] == value]
            return filtered_data
        return None
