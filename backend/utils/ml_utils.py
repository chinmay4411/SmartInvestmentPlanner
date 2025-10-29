import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Path to ML model
MODEL_PATH = Path(__file__).resolve().parents[1] / "ml_model.pkl"

_model = None

def load_model():
    """
    Lazy load the ML model. Ensures it loads only once.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            print(f"[WARN] ML model not found at {MODEL_PATH}, using fallback logic")
            return None
        _model = joblib.load(str(MODEL_PATH))
    return _model

def predict_allocation(income: float, expenses: float, age: int, risk: int) -> dict:
    """
    Predict allocation into SIP, FD, and Stocks.
    - Falls back to rule-based allocation if ML model fails.
    """
    savings = income - expenses
    if savings <= 0:
        return {"SIP": 0.0, "FD": 0.0, "Stocks": 0.0, "Total": 0.0}

    try:
        model = load_model()
        if model is None:
            raise Exception("ML model not loaded")

        X = pd.DataFrame(
            [[income, expenses, age, risk]],
            columns=["Income", "Expenses", "Age", "RiskTolerance"],
        )
        pred = model.predict(X)

        if pred.ndim > 1:
            pred = pred[0]

        sip, fd, stocks = map(float, pred)
        # Ensure no negative values
        sip, fd, stocks = max(0, sip), max(0, fd), max(0, stocks)
        total = sip + fd + stocks

        return {"SIP": sip, "FD": fd, "Stocks": stocks, "Total": total}

    except Exception as e:
        print(f"[WARN] ML prediction failed: {e}. Using fallback allocation.")
        # Fallback allocation based on risk
        if risk <= 2:  # Low risk
            sip, fd, stocks = savings * 0.2, savings * 0.7, savings * 0.1
        elif risk == 3:  # Medium risk
            sip, fd, stocks = savings * 0.3, savings * 0.4, savings * 0.3
        else:  # High risk
            sip, fd, stocks = savings * 0.2, savings * 0.2, savings * 0.6

        total = sip + fd + stocks
        return {"SIP": sip, "FD": fd, "Stocks": stocks, "Total": total}

# ✅ Optional: Test block
if __name__ == "__main__":
    result = predict_allocation(40000, 15000, 25, 3)
    print("Test Prediction:", result)
