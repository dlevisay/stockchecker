import os
import requests
import json
import time
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# Alpaca API imports (install with: pip install alpaca-py)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest, GetOrderByIdRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, OrderStatus

# --- Configuration (IMPORTANT: Use GitHub Secrets for these values!) ---
# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# Set to True for paper trading, False for live trading
# Always start with paper trading!
PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "True").lower() == "true"

# Define the base URL for Alpaca (paper or live)
# Paper Trading: https://paper-api.alpaca.markets
# Live Trading: https://api.alpaca.markets
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if PAPER_TRADING_MODE else "https://api.alpaca.markets"

# Crypto API (CoinGecko Free API)
COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"

# List of symbols to scan
SYMBOLS_TO_SCAN = {
    "AAPL": "stock",
    "MSFT": "stock",
    "GOOGL": "stock",
    "BTCUSD": "crypto",
    "ETHUSD": "crypto"
}

# Trading parameters
TRADE_AMOUNT_PER_ASSET = 100
MAX_POSITIONS = 5
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.03

# Strategy parameters
LOOKBACK_PERIOD_MINUTES = 60
RSI_PERIOD = 14
SMA_SHORT_PERIOD = 20
SMA_LONG_PERIOD = 200
DIP_THRESHOLD_PERCENT = 0.015
RSI_OVERSOLD = 30

# File to store bot's managed positions
POSITIONS_FILE = 'positions.json'


# --- Initialization ---
def initialize_alpaca_client():
    """Initializes and returns the Alpaca TradingClient."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Alpaca API keys not found. Please set them as GitHub Secrets.")
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

# --- Data Fetching Functions ---
def get_stock_data(symbol, interval='5m', lookback_minutes=LOOKBACK_PERIOD_MINUTES):
    """Fetches historical stock data using yfinance."""
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

def get_crypto_data(symbol, lookback_minutes=LOOKBACK_PERIOD_MINUTES):
    """Fetches crypto data using CoinGecko API."""
    coingecko_id = ""
    if symbol == "BTCUSD":
        coingecko_id = "bitcoin"
    elif symbol == "ETHUSD":
        coingecko_id = "ethereum"
    else:
        print(f"Unsupported crypto symbol for CoinGecko: {symbol}")
        return None

    try:
        hourly_url = f"{COINGECKO_API_BASE_URL}/coins/{coingecko_id}/market_chart?vs_currency=usd&days=14"
        hourly_response = requests.get(hourly_url).json()
        prices = [p[1] for p in hourly_response.get('prices', [])]
        timestamps = [p[0] for p in hourly_response.get('prices', [])]

        if not prices:
            print(f"No historical crypto data fetched for {symbol}.")
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

def get_latest_price_from_data(data):
    """Extracts the latest price from the fetched data DataFrame."""
    if data is None or data.empty:
        return None
    return data['Close'].iloc[-1]

def calculate_technical_indicators_from_df(df):
    """Calculates technical indicators from a DataFrame."""
    if df is None or df.empty:
        return None
    
    # Use squeeze() to ensure the 'Close' data is 1-dimensional
    close_prices = df['Close'].squeeze()
    
    df['RSI'] = RSIIndicator(close=close_prices, window=RSI_PERIOD).rsi()
    df[f'SMA_{SMA_SHORT_PERIOD}'] = SMAIndicator(close=close_prices, window=SMA_SHORT_PERIOD).sma_indicator()
    df[f'SMA_{SMA_LONG_PERIOD}'] = SMAIndicator(close=close_prices, window=SMA_LONG_PERIOD).sma_indicator()
    df['RecentHigh'] = df['High'].rolling(window=SMA_SHORT_PERIOD, min_periods=1).max()
    
    return df.iloc[-1]

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

# --- Main Trading Logic ---
def run_trading_strategy():
    """Executes the trading strategy: scan, buy, and sell."""
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
            print(f"Skipping {symbol}: No data available.")
            continue

        latest_price = get_latest_price_from_data(df_data)
        if latest_price is None:
            print(f"Skipping {symbol}: Could not get latest price.")
            continue

        latest_indicators = calculate_technical_indicators_from_df(df_data)
        if latest_indicators is None:
            print(f"Skipping {symbol}: Could not calculate indicators.")
            continue

        # Extract scalar indicator values
        rsi = latest_indicators['RSI']
        sma_long = latest_indicators[f'SMA_{SMA_LONG_PERIOD}']
        recent_high = latest_indicators['RecentHigh']

        # --- This is the key fix ---
        # Check if indicators are valid numbers before using them
        if pd.isna(rsi) or pd.isna(sma_long) or pd.isna(recent_high):
            print(f"  Skipping {symbol}: Indicator values are not valid (NaN).")
            continue

        print(f"  Price: {latest_price:.2f}, RSI: {rsi:.2f}, SMA_200: {sma_long:.2f}")

        # Buy Logic
        if symbol not in bot_managed_positions:
            if len(bot_managed_positions) >= MAX_POSITIONS:
                print("  Skipping BUY: Max positions reached.")
                continue

            is_uptrend = latest_price > sma_long
            is_oversold = rsi <= RSI_OVERSOLD
            price_drop = (recent_high - latest_price) / recent_high
            is_dip = price_drop >= DIP_THRESHOLD_PERCENT
            
            print(f"  Buy conditions: Uptrend={is_uptrend}, Oversold={is_oversold}, Dip={is_dip}")

            if is_uptrend and is_oversold and is_dip:
                print(f"  SIGNAL: BUY {symbol}!")
                # Here you would add your logic to place a buy order
                # place_alpaca_order(...)
    
    save_bot_state(bot_managed_positions)
    print("\n--- Scan complete ---")

if __name__ == "__main__":
    run_trading_strategy()
