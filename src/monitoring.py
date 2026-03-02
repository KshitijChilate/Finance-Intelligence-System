# monitoring.py

def monitor_transaction(transaction, model):
    risk_score = model.predict_proba(transaction)[0][1]

    if risk_score > 0.85:
        alert = True
    else:
        alert = False

    return risk_score, alert