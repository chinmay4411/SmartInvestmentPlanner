from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
from utils.ml_utils import predict_allocation  # ML allocation
from utils.rag_utils import retrieve, get_investment_advice_context  # KB retrieval
from utils.report import create_report
from utils.stock_predictor import stock_predictor
from utils.live_data_service import live_data_service
from utils.enhanced_ml_advisor import enhanced_ml_advisor
from tinydb import TinyDB
from datetime import datetime
import requests

# -------------------------
# LOAD ENV 
# -------------------------
load_dotenv()

# -------------------------
# INIT APP
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# -------------------------
# PATHS & DB CONFIG
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
KB_DIR = os.path.join(DATA_DIR, "kb")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "db.json")
db = TinyDB(DB_FILE)
records_table = db.table("records")

# -------------------------
# HELPER: Get advice from local Ollama model or fallback
# -------------------------
OLLAMA_URL = "http://127.0.0.1:11434"  # Ollama default local server
OLLAMA_MODEL = "llama3.1:8b"  # exact model name from `ollama list`

def get_ollama_advice(prompt):
    try:
        print(f"\n[Ollama] Sending prompt to {OLLAMA_MODEL}...")
        print(f"[Ollama] Prompt length: {len(prompt)} characters")
        
        # Use the new Ollama API format
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            advice = data.get("response", "").strip()
            if advice:
                print(f"[Ollama] Generated advice: {len(advice)} characters")
                return advice
            else:
                print("[ERROR] Empty response from Ollama")
                return None
        else:
            print(f"[ERROR] Ollama API returned status: {response.status_code}")
            print("[Ollama] Raw response:", response.text[:500])
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Ollama request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama. Is it running?")
        return None
    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        return None

def get_fallback_advice(income, expenses, age, risk, allocation):
    """Generate fallback advice when Ollama is not available"""
    savings = income - expenses
    savings_rate = (savings / income) * 100 if income > 0 else 0
    
    advice_parts = []
    
    # Savings rate advice
    if savings_rate < 10:
        advice_parts.append(f"Your savings rate is {savings_rate:.1f}%, which is quite low. Consider reducing expenses or increasing income.")
    elif savings_rate < 20:
        advice_parts.append(f"Your savings rate is {savings_rate:.1f}%, which is moderate. Try to increase it to 20% or more for better financial security.")
    else:
        advice_parts.append(f"Great! Your savings rate is {savings_rate:.1f}%, which is excellent for building wealth.")
    
    # Age-based advice
    if age < 30:
        advice_parts.append("Since you're young, you can afford to take more risks for higher long-term returns.")
    elif age < 50:
        advice_parts.append("You're in your prime earning years - focus on growth while maintaining some stability.")
    else:
        advice_parts.append("As you approach retirement, prioritize capital preservation and stable returns.")
    
    # Risk-based advice
    if risk <= 2:
        advice_parts.append("Your conservative approach is good for capital preservation. Consider gradually increasing equity exposure for better long-term growth.")
    elif risk == 3:
        advice_parts.append("Your balanced approach provides good growth potential while managing risk effectively.")
    else:
        advice_parts.append("Your aggressive approach can generate higher returns, but ensure you have an emergency fund and can handle market volatility.")
    
    # Allocation-specific advice
    sip_pct = (allocation['SIP'] / savings) * 100 if savings > 0 else 0
    fd_pct = (allocation['FD'] / savings) * 100 if savings > 0 else 0
    stocks_pct = (allocation['Stocks'] / savings) * 100 if savings > 0 else 0
    
    advice_parts.append(f"Your recommended allocation: {sip_pct:.0f}% SIP, {fd_pct:.0f}% FD, {stocks_pct:.0f}% Stocks. This balances growth, stability, and liquidity based on your risk profile.")
    
    return " ".join(advice_parts)

def get_enhanced_advice(income, expenses, age, risk, allocation, kb_context=""):
    """Generate enhanced advice combining multiple sources"""
    advice_parts = []
    
    # Get contextual advice
    context_advice = get_investment_advice_context(income, expenses, age, risk, allocation)
    advice_parts.append(context_advice)
    
    # Add knowledge base context if available
    if kb_context:
        advice_parts.append(f"Based on financial knowledge: {kb_context}")
    
    # Add specific allocation advice
    savings = income - expenses
    if savings > 0:
        sip_pct = (allocation['SIP'] / savings) * 100
        fd_pct = (allocation['FD'] / savings) * 100
        stocks_pct = (allocation['Stocks'] / savings) * 100
        
        advice_parts.append(f"Your recommended allocation: {sip_pct:.0f}% SIP for systematic investing, {fd_pct:.0f}% FD for safety, and {stocks_pct:.0f}% stocks for growth. This balanced approach helps manage risk while building wealth over time.")
    
    # Add market timing advice
    advice_parts.append("Remember: Time in the market beats timing the market. Start investing early and stay consistent with your SIPs regardless of market conditions.")
    
    return " ".join(advice_parts)

# -------------------------
# ROUTES
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"msg": "No JSON payload provided"}), 400

    # Parse user input
    try:
        income = float(payload.get("income", 0))
        expenses = float(payload.get("expenses", 0))
        age = int(payload.get("age", 0))
        risk = int(payload.get("risk", 3))
    except Exception as e:
        print("[ERROR] Invalid input:", e)
        return jsonify({"msg": "Invalid input types"}), 400

    if any(v < 0 for v in [income, expenses, age]) or not (1 <= risk <= 5):
        return jsonify({"msg": "Invalid values"}), 400

    # -------------------------
    # Enhanced ML Allocation with Live Data
    # -------------------------
    try:
        # Use enhanced ML advisor with live market data
        allocation = enhanced_ml_advisor.predict_enhanced_allocation(income, expenses, age, risk)
        print(f"[INFO] Using enhanced ML allocation with market conditions")
    except Exception as e:
        print("[WARN] Enhanced ML model not available, trying basic model:", e)
        try:
            allocation = predict_allocation(income, expenses, age, risk)
        except Exception as e2:
            print("[WARN] Basic ML model not loaded, using fallback allocation:", e2)
            allocation = {"SIP": income * 0.3, "FD": income * 0.3, "Stocks": income * 0.4, "Total": income}

    # Save record
    record_id = len(records_table) + 1
    records_table.insert({
        "id": record_id,
        "income": income,
        "expenses": expenses,
        "age": age,
        "risk": risk,
        "sip": allocation.get("SIP", 0),
        "fd": allocation.get("FD", 0),
        "stocks": allocation.get("Stocks", 0),
        "created_at": datetime.utcnow().isoformat()
    })

    # -------------------------
    # KB Context
    # -------------------------
    try:
        kb_ctx = retrieve(f"Recommended strategy for user: income {income}, expenses {expenses}, age {age}, risk {risk}", top_k=3)
        kb_text = "\n\n".join([f"{k['source']}: {k['text']}" for k in kb_ctx])
    except Exception as e:
        print("[ERROR] KB retrieval failed:", e)
        kb_text = ""

    # -------------------------
    # Build enhanced prompt for Ollama
    # -------------------------
    savings = income - expenses
    savings_rate = (savings / income) * 100 if income > 0 else 0
    
    prompt = f"""You are an expert financial advisor with 20+ years of experience. Provide personalized investment advice based on the following user profile:

USER PROFILE:
- Monthly Income: ₹{income:,.0f}
- Monthly Expenses: ₹{expenses:,.0f}
- Monthly Savings: ₹{savings:,.0f} ({savings_rate:.1f}% savings rate)
- Age: {age} years
- Risk Tolerance: {risk}/5 ({'Conservative' if risk <= 2 else 'Moderate' if risk == 3 else 'Aggressive'})

ML-PREDICTED ALLOCATION:
- SIP (Systematic Investment Plan): ₹{allocation['SIP']:,.0f} ({allocation['SIP']/savings*100:.0f}% of savings)
- Fixed Deposits: ₹{allocation['FD']:,.0f} ({allocation['FD']/savings*100:.0f}% of savings)
- Stocks/Equity: ₹{allocation['Stocks']:,.0f} ({allocation['Stocks']/savings*100:.0f}% of savings)

FINANCIAL KNOWLEDGE CONTEXT:
{kb_text}

INSTRUCTIONS:
1. Analyze the user's financial situation and risk profile
2. Explain why this allocation makes sense for their age and risk tolerance
3. Provide specific actionable advice for each investment category
4. Mention any risks or considerations
5. Keep the advice practical and easy to understand
6. Use a professional but friendly tone
7. Limit response to 4-6 sentences

Generate personalized investment advice:"""

    # -------------------------
    # Get enhanced advice with stock recommendations
    # -------------------------
    try:
        # Get stock recommendations
        user_profile = {
            'income': income,
            'expenses': expenses,
            'age': age,
            'risk_tolerance': risk
        }
        stock_recommendations = enhanced_ml_advisor.get_stock_recommendations(user_profile, limit=5)
        
        # Generate comprehensive advice
        advice = enhanced_ml_advisor.generate_investment_advice(
            user_profile, allocation, stock_recommendations
        )
        
        # Add stock recommendations to response
        allocation['stock_recommendations'] = stock_recommendations
        
    except Exception as e:
        print(f"[WARN] Enhanced advice generation failed: {e}")
        # Fallback to Ollama or basic advice
        advice = get_ollama_advice(prompt)
        if not advice:
            print("[INFO] Using enhanced fallback advice system")
            advice = get_enhanced_advice(income, expenses, age, risk, allocation, kb_text)

    return jsonify({"allocation": allocation, "advice": advice})

# -------------------------
# History & Report
# -------------------------
@app.route("/history", methods=["GET"])
def history():
    user_records = records_table.all()
    user_records.sort(key=lambda r: r["created_at"], reverse=True)
    return jsonify(user_records)

@app.route("/report/<int:rec_id>", methods=["GET"])
def get_report(rec_id):
    record = next((r for r in records_table.all() if r["id"] == rec_id), None)
    if not record:
        return jsonify({"msg": "Not found"}), 404

    allocation = {
        "SIP": record["sip"],
        "FD": record["fd"],
        "Stocks": record["stocks"],
        "Total": record["sip"] + record["fd"] + record["stocks"]
    }
    user_info = {
        "income": record["income"],
        "expenses": record["expenses"],
        "age": record["age"],
        "risk": record["risk"]
    }

    report_path = create_report(
        allocation, user_info, advice="See dashboard",
        out_path=os.path.join(UPLOADS_DIR, f"report_{rec_id}.pdf")
    )
    return send_file(report_path, as_attachment=True)

# -------------------------
# Stock Prediction Routes
# -------------------------
@app.route("/stock/predict/<symbol>", methods=["GET"])
def predict_stock_price(symbol):
    """Predict stock price for a given symbol"""
    try:
        days_ahead = int(request.args.get('days', 5))
        prediction = stock_predictor.predict_price(symbol.upper(), days_ahead)
        
        if prediction:
            return jsonify({
                "success": True,
                "symbol": symbol.upper(),
                "prediction": prediction
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Could not predict price for {symbol}. Please ensure the symbol is valid and try again."
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Stock prediction failed: {e}")
        return jsonify({
            "success": False,
            "message": "Stock prediction failed. Please try again later."
        }), 500

@app.route("/stock/analyze/<symbol>", methods=["GET"])
def analyze_stock(symbol):
    """Get comprehensive stock analysis"""
    try:
        analysis = stock_predictor.get_stock_analysis(symbol.upper())
        
        if analysis:
            return jsonify({
                "success": True,
                "symbol": symbol.upper(),
                "analysis": analysis
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Could not analyze {symbol}. Please ensure the symbol is valid."
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Stock analysis failed: {e}")
        return jsonify({
            "success": False,
            "message": "Stock analysis failed. Please try again later."
        }), 500

@app.route("/stock/train/<symbol>", methods=["POST"])
def train_stock_model(symbol):
    """Train stock prediction model for a symbol"""
    try:
        period = request.json.get('period', '2y') if request.json else '2y'
        success = stock_predictor.train_model(symbol.upper(), period)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Model trained successfully for {symbol.upper()}"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to train model for {symbol}. Please check if the symbol is valid."
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        return jsonify({
            "success": False,
            "message": "Model training failed. Please try again later."
        }), 500

@app.route("/stocks/popular", methods=["GET"])
def get_popular_stocks():
    """Get list of popular stocks for analysis"""
    popular_stocks = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "NFLX", "name": "Netflix Inc."},
        {"symbol": "AMD", "name": "Advanced Micro Devices"},
        {"symbol": "INTC", "name": "Intel Corporation"}
    ]
    
    return jsonify({
        "success": True,
        "stocks": popular_stocks
    })

# -------------------------
# Live Data Routes
# -------------------------
@app.route("/live/stock/<symbol>", methods=["GET"])
def get_live_stock_data(symbol):
    """Get live stock data"""
    try:
        data = live_data_service.get_live_stock_data(symbol.upper())
        
        if data:
            return jsonify({
                "success": True,
                "data": data
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Could not fetch live data for {symbol}"
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Live data fetch failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to fetch live data"
        }), 500

@app.route("/live/market-overview", methods=["GET"])
def get_market_overview():
    """Get market overview"""
    try:
        market_data = live_data_service.get_market_overview()
        
        return jsonify({
            "success": True,
            "market_data": market_data
        })
        
    except Exception as e:
        print(f"[ERROR] Market overview failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to fetch market overview"
        }), 500

@app.route("/live/sector-performance", methods=["GET"])
def get_sector_performance():
    """Get sector performance"""
    try:
        sector_data = live_data_service.get_sector_performance()
        
        return jsonify({
            "success": True,
            "sector_data": sector_data
        })
        
    except Exception as e:
        print(f"[ERROR] Sector performance failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to fetch sector performance"
        }), 500

@app.route("/live/trending", methods=["GET"])
def get_trending_stocks():
    """Get trending stocks"""
    try:
        limit = int(request.args.get('limit', 10))
        trending = live_data_service.get_trending_stocks(limit)
        
        return jsonify({
            "success": True,
            "trending_stocks": trending
        })
        
    except Exception as e:
        print(f"[ERROR] Trending stocks failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to fetch trending stocks"
        }), 500

@app.route("/live/stock/<symbol>/news", methods=["GET"])
def get_stock_news(symbol):
    """Get stock news"""
    try:
        limit = int(request.args.get('limit', 5))
        news = live_data_service.get_stock_news(symbol.upper(), limit)
        
        return jsonify({
            "success": True,
            "news": news
        })
        
    except Exception as e:
        print(f"[ERROR] Stock news failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to fetch stock news"
        }), 500

@app.route("/live/stock/<symbol>/recommendation", methods=["GET"])
def get_stock_recommendation(symbol):
    """Get investment recommendation for a stock"""
    try:
        risk_profile = int(request.args.get('risk', 3))
        recommendation = live_data_service.get_investment_recommendation(symbol.upper(), risk_profile)
        
        if 'error' not in recommendation:
            return jsonify({
                "success": True,
                "recommendation": recommendation
            })
        else:
            return jsonify({
                "success": False,
                "message": recommendation['error']
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Stock recommendation failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to generate recommendation"
        }), 500

@app.route("/live/stock/<symbol>/technical", methods=["GET"])
def get_technical_indicators(symbol):
    """Get technical indicators for a stock"""
    try:
        period = request.args.get('period', '3mo')
        indicators = live_data_service.calculate_technical_indicators(symbol.upper(), period)
        
        return jsonify({
            "success": True,
            "indicators": indicators
        })
        
    except Exception as e:
        print(f"[ERROR] Technical indicators failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to calculate technical indicators"
        }), 500

# -------------------------
# Enhanced ML Routes
# -------------------------
@app.route("/ml/train-enhanced", methods=["POST"])
def train_enhanced_model():
    """Train the enhanced ML model"""
    try:
        n_samples = request.json.get('n_samples', 100000) if request.json else 100000
        success = enhanced_ml_advisor.train_enhanced_model(n_samples)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Enhanced ML model trained successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to train enhanced model"
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Enhanced model training failed: {e}")
        return jsonify({
            "success": False,
            "message": "Model training failed"
        }), 500

@app.route("/ml/stock-recommendations", methods=["POST"])
def get_ml_stock_recommendations():
    """Get ML-powered stock recommendations"""
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"msg": "No JSON payload provided"}), 400
        
        user_profile = {
            'income': float(payload.get('income', 0)),
            'expenses': float(payload.get('expenses', 0)),
            'age': int(payload.get('age', 0)),
            'risk_tolerance': int(payload.get('risk', 3))
        }
        
        limit = int(payload.get('limit', 5))
        recommendations = enhanced_ml_advisor.get_stock_recommendations(user_profile, limit)
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
        
    except Exception as e:
        print(f"[ERROR] Stock recommendations failed: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to generate recommendations"
        }), 500

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print(f"TinyDB path: {DB_FILE}")
    app.run(port=5500, debug=True)
