# src/alert_manager.py

class AlertManager:

    def generate_action(self, risk_tier):
        if risk_tier == "Low":
            return "No Action Needed"

        elif risk_tier == "Medium":
            return "Flag for Monitoring"

        elif risk_tier == "High":
            return "Trigger Risk Alert"

        elif risk_tier == "Critical":
            return "Escalate to Manual Review"

        return "Unknown Tier"