#!/usr/bin/env python3
"""
Script to train the enhanced ML model with live market data integration
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from utils.enhanced_ml_advisor import enhanced_ml_advisor
from utils.live_data_service import live_data_service

def main():
    print("🚀 Starting Enhanced ML Model Training...")
    print("=" * 50)
    
    try:
        # Train the enhanced model
        print("📊 Training enhanced ML model with market conditions...")
        success = enhanced_ml_advisor.train_enhanced_model(n_samples=100000)
        
        if success:
            print("✅ Enhanced ML model trained successfully!")
            print("📈 Model now includes:")
            print("   - User profile analysis")
            print("   - Live market conditions")
            print("   - Market sentiment analysis")
            print("   - Volatility assessment")
            print("   - Interest rate considerations")
        else:
            print("❌ Failed to train enhanced model")
            return False
            
        # Test the model
        print("\n🧪 Testing the model...")
        test_allocation = enhanced_ml_advisor.predict_enhanced_allocation(
            income=100000,
            expenses=60000,
            age=30,
            risk_tolerance=3
        )
        
        print("📋 Test Results:")
        print(f"   Income: ₹100,000")
        print(f"   Expenses: ₹60,000")
        print(f"   Age: 30")
        print(f"   Risk: 3/5")
        print(f"   → SIP: ₹{test_allocation['SIP']:,.0f}")
        print(f"   → FD: ₹{test_allocation['FD']:,.0f}")
        print(f"   → Stocks: ₹{test_allocation['Stocks']:,.0f}")
        
        if 'market_conditions' in test_allocation:
            market = test_allocation['market_conditions']
            print(f"   → Market Sentiment: {market['market_sentiment']:.2f}")
            print(f"   → Volatility: {market['volatility']:.2f}")
            print(f"   → Recent Return: {market['recent_return']:.2f}%")
        
        # Test stock recommendations
        print("\n📈 Testing stock recommendations...")
        user_profile = {
            'income': 100000,
            'expenses': 60000,
            'age': 30,
            'risk_tolerance': 3
        }
        
        recommendations = enhanced_ml_advisor.get_stock_recommendations(user_profile, limit=3)
        
        if recommendations:
            print("🎯 Top Stock Recommendations:")
            for i, stock in enumerate(recommendations, 1):
                print(f"   {i}. {stock['symbol']} - {stock['name']}")
                print(f"      Price: ${stock['current_price']:.2f}")
                print(f"      Change: {stock['price_change']:+.2f}%")
                print(f"      Recommendation: {stock['recommendation']}")
                print(f"      Score: {stock['score']}/10")
        
        print("\n🎉 Enhanced ML model is ready for production!")
        print("💡 The model now provides:")
        print("   ✅ Live market data integration")
        print("   ✅ Personalized stock recommendations")
        print("   ✅ Market condition analysis")
        print("   ✅ Risk-adjusted investment advice")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
