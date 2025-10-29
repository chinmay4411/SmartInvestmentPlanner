import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "financial_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "ml_model.pkl")

def create_dataset(n=200000, seed=42):
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        income = np.random.randint(10000, 1000000)
        expenses = np.random.randint(5000, int(income*0.9))
        age = np.random.randint(18, 75)
        risk = np.random.randint(1, 6)
        saving = max(income - expenses, 0)
        sip = saving * (0.15 + 0.05*risk) * np.random.uniform(0.9, 1.1)  # add 10% noise
        fd = saving * (0.5 - 0.06*risk) * np.random.uniform(0.9, 1.1)
        stocks = max(saving - sip - fd, 0) * np.random.uniform(0.95, 1.05)
        rows.append([income, expenses, age, risk, sip, fd, stocks])
    df = pd.DataFrame(rows, columns=["Income","Expenses","Age","RiskTolerance","SIP","FD","Stocks"])
    df.to_csv(CSV_PATH, index=False)
    print(f"Dataset created with {n} rows at {CSV_PATH}")
    return df

def train_and_save(n_rows=200000):
    df = create_dataset(n=n_rows)
    X = df[["Income","Expenses","Age","RiskTolerance"]]
    y = df[["SIP","FD","Stocks"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = Pipeline([
        ("scaler", StandardScaler()), 
        ("rf", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained and saved ML model at {MODEL_PATH}")
    print(f"Training R² score: {model.score(X_train, y_train):.4f}")
    print(f"Test R² score: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_and_save(n_rows=200000)
