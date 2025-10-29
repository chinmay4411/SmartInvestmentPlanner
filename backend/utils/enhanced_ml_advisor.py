import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from typing import Dict, List

class EnhancedMLAdvisor:
    """Offline ML advisor that combines user profile with simulated market data"""
    
    def __init__(self):
        self.allocation_model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'enhanced_ml_model.pkl')
        
    def create_enhanced_dataset(self, n_samples: int = 100000) -> pd.DataFrame:
        """Create enhanced dataset with simulated market conditions"""
        np.random.seed(42)
        rows = []
        
        market_conditions = ['bull', 'bear', 'sideways', 'volatile']
        market_weights = [0.4, 0.2, 0.3, 0.1]
        
        for _ in range(n_samples):
            income = np.random.randint(20000, 2000000)
            expenses = np.random.randint(10000, int(income * 0.8))
            age = np.random.randint(18, 75)
            risk_tolerance = np.random.randint(1, 6)
            
            market_condition = np.random.choice(market_conditions, p=market_weights)
            
            # Simulated market indicators
            if market_condition == 'bull':
                market_sentiment = np.random.uniform(0.6, 1.0)
                volatility = np.random.uniform(0.1, 0.3)
                interest_rate = np.random.uniform(2, 5)
            elif market_condition == 'bear':
                market_sentiment = np.random.uniform(0.0, 0.4)
                volatility = np.random.uniform(0.3, 0.6)
                interest_rate = np.random.uniform(3, 7)
            elif market_condition == 'sideways':
                market_sentiment = np.random.uniform(0.4, 0.6)
                volatility = np.random.uniform(0.2, 0.4)
                interest_rate = np.random.uniform(3, 6)
            else:
                market_sentiment = np.random.uniform(0.3, 0.7)
                volatility = np.random.uniform(0.4, 0.8)
                interest_rate = np.random.uniform(2, 8)
            
            savings = max(income - expenses, 0)
            
            base_sip, base_fd, base_stocks = savings * 0.3, savings * 0.3, savings * 0.4
            
            if market_condition == 'bull':
                stocks_multiplier, sip_multiplier, fd_multiplier = 1.2, 1.1, 0.7
            elif market_condition == 'bear':
                stocks_multiplier, sip_multiplier, fd_multiplier = 0.6, 0.8, 1.4
            elif market_condition == 'volatile':
                stocks_multiplier, sip_multiplier, fd_multiplier = 0.8, 1.3, 0.9
            else:
                stocks_multiplier = sip_multiplier = fd_multiplier = 1.0
            
            risk_multiplier = 0.5 + (risk_tolerance - 1) * 0.25
            stocks_multiplier *= risk_multiplier
            sip_multiplier *= (1.5 - risk_multiplier * 0.5)
            
            sip = base_sip * sip_multiplier
            fd = base_fd * fd_multiplier
            stocks = base_stocks * stocks_multiplier
            
            total = sip + fd + stocks
            if total > 0:
                sip = (sip / total) * savings
                fd = (fd / total) * savings
                stocks = (stocks / total) * savings
            
            sip *= np.random.uniform(0.95, 1.05)
            fd *= np.random.uniform(0.95, 1.05)
            stocks *= np.random.uniform(0.95, 1.05)
            
            rows.append([income, expenses, age, risk_tolerance,
                         market_sentiment, volatility, interest_rate,
                         sip, fd, stocks])
        
        columns = [
            'Income', 'Expenses', 'Age', 'RiskTolerance',
            'MarketSentiment', 'Volatility', 'InterestRate',
            'SIP', 'FD', 'Stocks'
        ]
        return pd.DataFrame(rows, columns=columns)
    
    def train_enhanced_model(self, n_samples: int = 100000):
        """Train the offline enhanced ML model"""
        print("Creating simulated dataset...")
        df = self.create_enhanced_dataset(n_samples)
        
        feature_columns = ['Income', 'Expenses', 'Age', 'RiskTolerance', 
                          'MarketSentiment', 'Volatility', 'InterestRate']
        target_columns = ['SIP', 'FD', 'Stocks']
        
        X = df[feature_columns]
        y = df[target_columns]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.allocation_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42
        )
        self.allocation_model.fit(X_train_scaled, y_train)
        
        print(f"Training R²: {self.allocation_model.score(X_train_scaled, y_train):.4f}")
        print(f"Test R²: {self.allocation_model.score(X_test_scaled, y_test):.4f}")
        
        joblib.dump({
            'allocation_model': self.allocation_model,
            'scaler': self.scaler,
            'feature_columns': feature_columns,
            'target_columns': target_columns,
            'trained_at': datetime.now().isoformat()
        }, self.model_path)
        
        print(f"Offline model saved at: {self.model_path}")
        return True
    
    def load_enhanced_model(self) -> bool:
        """Load saved offline model"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.allocation_model = model_data['allocation_model']
            self.scaler = model_data['scaler']
            return True
        return False
    
    def _get_default_market_conditions(self) -> Dict:
        """Simulated market snapshot"""
        return {
            'market_sentiment': 0.6,
            'volatility': 0.3,
            'interest_rate': 4.5
        }
    
    def predict_enhanced_allocation(self, income: float, expenses: float, age: int, risk_tolerance: int) -> Dict:
        """Predict allocations using simulated market conditions"""
        if not self.load_enhanced_model():
            print("Model not found — training a new one...")
            self.train_enhanced_model()
            self.load_enhanced_model()
        
        market = self._get_default_market_conditions()
        
        features = np.array([[income, expenses, age, risk_tolerance,
                              market['market_sentiment'], market['volatility'], market['interest_rate']]])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.allocation_model.predict(features_scaled)[0]
        
        savings = max(income - expenses, 0)
        sip, fd, stocks = map(lambda x: max(0, x), prediction)
        total = sip + fd + stocks
        
        if total > 0:
            sip = (sip / total) * savings
            fd = (fd / total) * savings
            stocks = (stocks / total) * savings
        
        return {
            'SIP': sip,
            'FD': fd,
            'Stocks': stocks,
            'Total': savings,
            'market_conditions': market,
            'model_type': 'offline_ml'
        }

# Global instance
enhanced_ml_advisor = EnhancedMLAdvisor()
