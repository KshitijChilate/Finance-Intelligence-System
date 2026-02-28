import pandas as pd


def load_data(users_path, transactions_path):
    users = pd.read_csv(users_path)
    transactions = pd.read_csv(transactions_path)

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    return users, transactions
