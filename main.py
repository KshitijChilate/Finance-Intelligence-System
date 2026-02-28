print("Starting pipeline...")

from src.pipeline import run_pipeline


if __name__ == "__main__":

    print("Inside main block")

    final_data = run_pipeline(
        users_path="data/users.csv",
        transactions_path="data/transactions.csv"
    )

    print("Pipeline executed successfully.")
    print(final_data.head())
