import os
import requests
import json
import time
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# Alpaca API imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "True").lower() == "true"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if PAPER_TRADING_MODE else "https://api.alpaca.markets"
COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"

SYMBOLS_TO_SCAN = {
    "AAPL": "stock",
    "MSFT": "stock",
    "GOOGL": "stock",
    "BTCUSD": "crypto",
    "ETHUSD": "crypto"
}

TRADE_AMOUNT_PER_ASSET = 100
MAX_POSITIONS = 5
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.03

LOOKBACK_PERIOD_MINUTES = 60
RSI_PERIOD = 14
SMA_SHORT_PERIOD = 20
SMA_LONG_PERIOD = 200
DIP_THRESHOLD_PERCENT = 0.015
RSI_OVERSOLD = 30

POSITIONS_FILE = 'positions.json'


# --- Initialization ---
def initialize_alpaca_client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Alpaca API keys not found.")
        return None
    try:
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING_MODE)
        account = client.get_account()
        print(f"Alpaca client initialized. Account status: {account.status}, Buying Power: {account.buying_power}")
        return client
    except Exception as e:
        print(f"Error initializing Alpaca client: {e}")
        return None

alpaca_client = initialize_alpaca_client()

# --- Data Fetching ---
def get_stock_data(symbol, interval='5m', lookback_minutes=LOOKBACK_PERIOD_MINUTES):
    try:
        data = yf.download(symbol, interval=interval, period="5d", progress=False)
        if data.empty:
            print(f"No stock data fetched for {symbol}.")
            return None
        if len(data) < SMA_LONG_PERIOD:
            print(f"Warning: Not enough historical data for {symbol} for {SMA_LONG_PERIOD}-period SMA.")
        
        bars_needed = lookback_minutes // int(interval.replace('m', ''))
        return data.tail(bars_needed)
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def get_crypto_data(symbol):
    coingecko_id = ""
    if symbol == "BTCUSD":
        coingecko_id = "bitcoin"
    elif symbol == "ETHUSD":
        coingecko_id = "ethereum"
    else:
        print(f"Unsupported crypto symbol: {symbol}")
        return None
    try:
        hourly_url = f"{COINGECKO_API_BASE_URL}/coins/{coingecko_id}/market_chart?vs_currency=usd&days=14"
        hourly_response = requests.get(hourly_url).json()
        prices = [p[1] for p in hourly_response.get('prices', [])]
        timestamps = [p[0] for p in hourly_response.get('prices', [])]

        if not prices:
            print(f"No historical crypto data for {symbol}.")
            return None

        df = pd.DataFrame(prices, columns=['Close'])
        df.index = pd.to_datetime(timestamps, unit='ms')
        df.index.name = 'Datetime'
        
        df['Open'] = df['High'] = df['Low'] = df['Close']
        df['Volume'] = 0

        if len(df) < SMA_LONG_PERIOD:
            print(f"Warning: Not enough historical crypto data for {SMA_LONG_PERIOD}-period SMA.")

        return df
    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {e}")
        return None

# --- Indicator Calculation ---
def calculate_technical_indicators(df):
    if df is None or df.empty:
        return None, None, None
    
    close_prices = df['Close'].squeeze()
    
    rsi = RSIIndicator(close=close_prices, window=RSI_PERIOD).rsi().iloc[-1]
    sma_long = SMAIndicator(close=close_prices, window=SMA_LONG_PERIOD).sma_indicator().iloc[-1]
    recent_high = df['High'].rolling(window=SMA_SHORT_PERIOD, min_periods=1).max().iloc[-1]
    
    return rsi, sma_long, recent_high

# --- State Management ---
def load_bot_state():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_bot_state(state):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(state, f, indent=4)

# --- Main Logic ---
def run_trading_strategy():
    print(f"\n--- Starting scan at {pd.Timestamp.now(tz='UTC')} ---")
    bot_managed_positions = load_bot_state()

    for symbol, asset_type in SYMBOLS_TO_SCAN.items():
        print(f"\nProcessing {symbol} ({asset_type})...")

        df_data = None
        if asset_type == "stock":
            df_data = get_stock_data(symbol)
        elif asset_type == "crypto":
            df_data = get_crypto_data(symbol)

        if df_data is None or df_data.empty:
            print(f"Skipping {symbol}: No data.")
            continue

        latest_price = df_data['Close'].iloc[-1]
        rsi, sma_long, recent_high = calculate_technical_indicators(df_data)

        if pd.isna(rsi) or pd.isna(sma_long) or pd.isna(recent_high):
            print(f"  Skipping {symbol}: Not enough data for indicators.")
            continue

        print(f"  Price: {latest_price:.2f}, RSI: {rsi:.2f}, SMA_200: {sma_long:.2f}")

        if symbol not in bot_managed_positions:
            if len(bot_managed_positions) >= MAX_POSITIONS:
                print("  Skipping BUY: Max positions reached.")
                continue

            is_uptrend = latest_price > sma_long
            is_oversold = rsi <= RSI_OVERSOLD
            is_dip = (recent_high - latest_price) / recent_high >= DIP_THRESHOLD_PERCENT
            
            print(f"  Buy conditions: Uptrend={is_uptrend}, Oversold={is_oversold}, Dip={is_dip}")

            if is_uptrend and is_oversold and is_dip:
                print(f"  SIGNAL: BUY {symbol}!")
                # Add order placement logic here
    
    save_bot_state(bot_managed_positions)
    print("\n--- Scan complete ---")

if __name__ == "__main__":
    run_trading_strategy()
