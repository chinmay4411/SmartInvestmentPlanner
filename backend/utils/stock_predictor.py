import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import os

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'stock_model.pkl')
        
    def get_stock_data(self, symbol, period="2y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def create_features(self, data):
        """Create technical indicators and features for prediction"""
        df = data.copy()
        
        # Price-based features
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Price change features
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_20d'] = df['Close'].pct_change(20)
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_training_data(self, df, target_days=5):
        """Prepare data for training"""
        features = [
            'SMA_5', 'SMA_20', 'SMA_50', 'Volatility', 'RSI', 
            'MACD', 'MACD_Signal', 'Price_Change_1d', 'Price_Change_5d', 
            'Price_Change_20d', 'Volume_Ratio'
        ]
        
        X = df[features].values
        y = df['Close'].shift(-target_days).values  # Predict future price
        
        # Remove last target_days rows where y is NaN
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def train_model(self, symbol, period="2y"):
        """Train the stock prediction model"""
        print(f"Training model for {symbol}...")
        
        # Get data
        data = self.get_stock_data(symbol, period)
        if data is None or len(data) < 100:
            print(f"Insufficient data for {symbol}")
            return False
        
        # Create features
        df = self.create_features(data)
        if len(df) < 50:
            print(f"Insufficient features for {symbol}")
            return False
        
        # Prepare training data
        X, y = self.prepare_training_data(df)
        
        if len(X) < 30:
            print(f"Insufficient training data for {symbol}")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training R²: {train_score:.4f}")
        print(f"Test R²: {test_score:.4f}")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': [
                'SMA_5', 'SMA_20', 'SMA_50', 'Volatility', 'RSI', 
                'MACD', 'MACD_Signal', 'Price_Change_1d', 'Price_Change_5d', 
                'Price_Change_20d', 'Volume_Ratio'
            ],
            'symbol': symbol,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return True
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
    
    def predict_price(self, symbol, days_ahead=5):
        """Predict future stock price"""
        if not self.load_model():
            print("No trained model available")
            return None
        
        # Get recent data
        data = self.get_stock_data(symbol, period="3mo")
        if data is None:
            return None
        
        # Create features
        df = self.create_features(data)
        if len(df) == 0:
            return None
        
        # Get latest features
        latest_features = df.iloc[-1][
            ['SMA_5', 'SMA_20', 'SMA_50', 'Volatility', 'RSI', 
             'MACD', 'MACD_Signal', 'Price_Change_1d', 'Price_Change_5d', 
             'Price_Change_20d', 'Volume_Ratio']
        ].values.reshape(1, -1)
        
        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Predict
        prediction = self.model.predict(latest_features_scaled)[0]
        
        return {
            'current_price': float(df.iloc[-1]['Close']),
            'predicted_price': float(prediction),
            'price_change': float(prediction - df.iloc[-1]['Close']),
            'price_change_pct': float((prediction - df.iloc[-1]['Close']) / df.iloc[-1]['Close'] * 100),
            'days_ahead': days_ahead,
            'prediction_date': (datetime.now() + timedelta(days=days_ahead)).isoformat()
        }
    
    def get_stock_analysis(self, symbol):
        """Get comprehensive stock analysis"""
        data = self.get_stock_data(symbol, period="1y")
        if data is None:
            return None
        
        df = self.create_features(data)
        if len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        
        # Technical analysis
        analysis = {
            'symbol': symbol,
            'current_price': float(latest['Close']),
            'volume': int(latest['Volume']),
            'sma_20': float(latest['SMA_20']),
            'sma_50': float(latest['SMA_50']),
            'rsi': float(latest['RSI']),
            'volatility': float(latest['Volatility']),
            'macd': float(latest['MACD']),
            'price_change_1d': float(latest['Price_Change_1d'] * 100),
            'price_change_5d': float(latest['Price_Change_5d'] * 100),
            'price_change_20d': float(latest['Price_Change_20d'] * 100),
        }
        
        # Generate signals
        signals = []
        if latest['Close'] > latest['SMA_20']:
            signals.append("Price above 20-day SMA (Bullish)")
        else:
            signals.append("Price below 20-day SMA (Bearish)")
        
        if latest['RSI'] > 70:
            signals.append("RSI indicates overbought conditions")
        elif latest['RSI'] < 30:
            signals.append("RSI indicates oversold conditions")
        else:
            signals.append("RSI in neutral range")
        
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append("MACD bullish crossover")
        else:
            signals.append("MACD bearish crossover")
        
        analysis['signals'] = signals
        
        return analysis

# Global instance
stock_predictor = StockPredictor()
