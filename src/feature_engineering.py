import pandas as pd


def create_user_metrics(users, transactions):

    user_metrics = (
        transactions
        .groupby("user_id")
        .agg(
            total_spent=("amount", "sum"),
            avg_transaction=("amount", "mean"),
            transaction_count=("amount", "count"),
            spend_std=("amount", "std")
        )
        .reset_index()
    )

    user_metrics = user_metrics.merge(users, on="user_id")

    user_metrics["income_expense_ratio"] = (
        user_metrics["annual_income"] / user_metrics["total_spent"]
    )

    user_metrics["spend_per_transaction"] = (
        user_metrics["total_spent"] / user_metrics["transaction_count"]
    )

    return user_metrics
