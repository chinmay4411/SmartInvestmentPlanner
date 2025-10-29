import yfinance as yf
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional
from yahoo_fin import stock_info as si


class LiveDataService:
    """Optimized service for fetching live financial data with caching and rate-limit handling"""

    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.popular_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SPOT'
        ]
        self.major_indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        self.last_request_time = {}
        self.request_gap = 2  # seconds per symbol to avoid rate limit

    # ---- Cache Helpers ----
    def _cache_get(self, key: str) -> Optional[dict]:
        current_time = time.time()
        if key in self.cache:
            data, timestamp = self.cache[key]
            if current_time - timestamp < self.cache_duration:
                return data
        return None

    def _cache_set(self, key: str, data: dict):
        self.cache[key] = (data, time.time())

    # ---- Throttling ----
    def _throttle(self, symbol: str):
        now = time.time()
        if symbol in self.last_request_time:
            elapsed = now - self.last_request_time[symbol]
            if elapsed < self.request_gap:
                time.sleep(self.request_gap - elapsed)
        self.last_request_time[symbol] = time.time()

    # ---- Safe Ticker ----
    def _safe_yf_ticker(self, symbol: str) -> Optional[yf.Ticker]:
        for attempt in range(5):
            try:
                self._throttle(symbol)
                return yf.Ticker(symbol)
            except Exception as e:
                if "429" in str(e):
                    wait = 2 ** attempt
                    print(f"[429] Rate limited for {symbol}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[ERROR] Ticker init failed for {symbol}: {e}")
                    return None
        return None

    # ---- Live Stock Data ----
    def get_live_stock_data(self, symbol: str, period: str = "1d") -> Optional[Dict]:
        cache_key = f"{symbol}_{period}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        ticker = self._safe_yf_ticker(symbol)
        if not ticker:
            return None

        try:
            info = ticker.info
            hist = ticker.history(period=period)
            if hist.empty:
                print(f"[WARN] No data from yfinance for {symbol}, trying yahoo_fin fallback...")
                try:
                    price = float(si.get_live_price(symbol))
                    data = {
                        'symbol': symbol.upper(),
                        'name': info.get('longName', symbol),
                        'current_price': price,
                        'previous_close': info.get('previousClose', price),
                        'change': None,
                        'change_percent': None,
                        'volume': None,
                        'high': None,
                        'low': None,
                        'open': None,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'dividend_yield': info.get('dividendYield', 0),
                        'market_state': info.get('marketState', 'UNKNOWN'),
                        'is_market_open': False,
                        'last_updated': datetime.now().isoformat(),
                        'currency': info.get('currency', 'USD')
                    }
                    self._cache_set(cache_key, data)
                    return data
                except Exception as e:
                    print(f"[FALLBACK ERROR] Yahoo_fin also failed for {symbol}: {e}")
                    return None

            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            prev_close = float(info.get('previousClose', current_price))
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0
            market_state = info.get('marketState', 'UNKNOWN')
            is_market_open = market_state == 'REGULAR'

            data = {
                'symbol': symbol.upper(),
                'name': info.get('longName', symbol),
                'current_price': current_price,
                'previous_close': prev_close,
                'change': change,
                'change_percent': change_pct,
                'volume': int(latest['Volume']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(latest['Open']),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'market_state': market_state,
                'is_market_open': is_market_open,
                'last_updated': datetime.now().isoformat(),
                'currency': info.get('currency', 'USD')
            }

            self._cache_set(cache_key, data)
            return data

        except Exception as e:
            print(f"[ERROR] Fetching {symbol} failed: {e}")
            return None

    # ---- Multiple Stocks ----
    def get_multiple_stocks_data(self, symbols: List[str]) -> Dict[str, Dict]:
        results = {}
        symbols_to_fetch = []

        # Use cache first
        for symbol in symbols:
            cached = self._cache_get(f"{symbol}_1d")
            if cached:
                results[symbol] = cached
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            return results

        tickers = yf.Tickers(" ".join(symbols_to_fetch))
        for sym, ticker in tickers.tickers.items():
            data = self.get_live_stock_data(sym)
            if data:
                results[sym] = data
            time.sleep(1)  # spacing requests

        return results

    # ---- Market Overview ----
    def get_market_overview(self) -> Dict:
        return self.get_multiple_stocks_data(self.major_indices)

    # ---- Trending Stocks ----
    def get_trending_stocks(self, limit: int = 10) -> List[Dict]:
        data_dict = self.get_multiple_stocks_data(self.popular_stocks)
        trending = [d for d in data_dict.values() if d.get('volume', 0) > 1_000_000]
        trending.sort(key=lambda x: abs(x['change_percent']) * x['volume'], reverse=True)
        return trending[:limit]

    # ---- Technical Indicators ----
    def calculate_technical_indicators(self, symbol: str, period: str = "3mo") -> Dict:
        ticker = self._safe_yf_ticker(symbol)
        if not ticker:
            return {}
        hist = ticker.history(period=period)
        if hist.empty:
            return {}

        df = hist.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        latest = df.iloc[-1]
        return {
            'sma_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
            'sma_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None
        }

    # ---- Investment Recommendation ----
    def get_investment_recommendation(self, symbol: str, user_risk_profile: int = 3) -> Dict:
        data = self.get_live_stock_data(symbol)
        tech = self.calculate_technical_indicators(symbol)
        if not data or not tech:
            return {'error': f'Unable to fetch data for {symbol}'}

        score = 0
        signals = []

        change_pct = data['change_percent']
        if change_pct > 2:
            signals.append("Strong positive momentum")
            score += 2
        elif change_pct > 0:
            signals.append("Positive momentum")
            score += 1
        elif change_pct < -2:
            signals.append("Negative momentum")
            score -= 2
        else:
            signals.append("Neutral momentum")

        if tech.get('sma_20') and data['current_price'] > tech['sma_20']:
            signals.append("Above 20-day SMA")
            score += 1
        elif tech.get('sma_20') and data['current_price'] < tech['sma_20']:
            signals.append("Below 20-day SMA")
            score -= 1

        if user_risk_profile <= 2:
            recommendation = "BUY" if score >= 3 else "HOLD" if score >= 1 else "AVOID"
        elif user_risk_profile == 3:
            recommendation = "BUY" if score >= 2 else "HOLD" if score >= 0 else "SELL"
        else:
            recommendation = "BUY" if score >= 1 else "HOLD" if score >= -1 else "SELL"

        return {
            'symbol': symbol.upper(),
            'recommendation': recommendation,
            'score': score,
            'signals': signals,
            'current_price': data['current_price'],
            'change_percent': change_pct,
            'analysis_date': datetime.now().isoformat()
        }


# ---- Global Instance ----
live_data_service = LiveDataService()
