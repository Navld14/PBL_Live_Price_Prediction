import requests
import pandas as pd
import ta
from statsmodels.tsa.seasonal import seasonal_decompose
from requests.exceptions import RequestException

def fetch_binance_data(limit=240, interval="1m"):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", 
                        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"])
        
        # Convert and clean data
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float).ffill()
        
        # Advanced preprocessing
        if len(df) >= 100:
            try:
                # Decomposition
                decomposition = seasonal_decompose(df['close'], model='additive', period=24)
                df['detrended'] = df['close'] - decomposition.trend
                df['seasonal_adjusted'] = df['close'] - decomposition.seasonal
                
                # Smoothing
                df['smoothed'] = df['close'].ewm(span=14, adjust=False).mean()
            except Exception as e:
                print(f"Decomposition error: {e}")
        
        # Enhanced technical indicators
        if len(df) >= 20:
            try:
                df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
                df["EMA"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
                df["MACD"] = ta.trend.MACD(df["close"]).macd()
                bb = ta.volatility.BollingerBands(df["close"])
                df["Bollinger_High"] = bb.bollinger_hband()
                df["Bollinger_Low"] = bb.bollinger_lband()
            except Exception as e:
                print(f"Indicator error: {e}")
        
        return df[["timestamp", "open", "high", "low", "close", "volume", 
                 "RSI", "EMA", "MACD", "Bollinger_High", "Bollinger_Low"]]
    
    except RequestException as e:
        print(f"API error: {e}")
        return pd.DataFrame()
