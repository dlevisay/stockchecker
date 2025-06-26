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
RSI_OVERBOUGHT = 70

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
            print(f"Not enough historical data for {symbol} for {SMA_LONG_PERIOD}-period SMA.")
        bars_needed = lookback_minutes // int(interval.replace('m', ''))
        return data.tail(bars_needed)
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def get_crypto_data(symbol, lookback_minutes=LOOKBACK_PERIOD_MINUTES):
    coingecko_id = ""
    if symbol == "BTCUSD":
        coingecko_id = "bitcoin"
    elif symbol == "ETHUSD":
        coingecko_id = "ethereum"
    else:
        print(f"Unsupported crypto symbol: {symbol}")
        return None
    try:
        hourly_url = f"{COINGECKO_API_BASE_URL}/coins/{coingecko_id}/market_chart?vs_currency=usd&days=7"
        hourly_response = requests.get(hourly_url).json()
        prices = [p[1] for p in hourly_response.get('prices', [])]
        timestamps = [p[0] for p in hourly_response.get('prices', [])]
        if not prices:
            print(f"No historical crypto data for {symbol}.")
            return None
        df = pd.DataFrame(prices, columns=['Close'])
        df.index = pd.to_datetime(timestamps, unit='ms')
        df.index.name = 'Datetime'
        current_price_url = f"{COINGECKO_API_BASE_URL}/simple/price?ids={coingecko_id}&vs_currencies=usd"
        current_response = requests.get(current_price_url).json()
        latest_price = current_response.get(coingecko_id, {}).get('usd')
        df['Open'] = df['High'] = df['Low'] = df['Close']
        df['Volume'] = 0
        if len(df) < SMA_LONG_PERIOD:
             print(f"Warning: Not enough historical data for {symbol} for {SMA_LONG_PERIOD}-period SMA.")
        if latest_price:
            current_time = pd.Timestamp.now(tz='UTC')
            new_row = pd.DataFrame([{'Open': latest_price, 'High': latest_price, 'Low': latest_price, 'Close': latest_price, 'Volume': 0}], index=[current_time])
            new_row.index.name = 'Datetime'
            df = pd.concat([df, new_row]).tail(lookback_minutes)
        else:
            latest_price = df['Close'].iloc[-1]
        df['latest_price'] = latest_price
        return df
    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {e}")
        return None

# --- Indicator Calculation ---
def calculate_technical_indicators_from_df(df):
    if df is None or df.empty:
        return None
    # Squeeze the 'Close' column to ensure it is 1-dimensional
    close_prices = df['Close'].squeeze()
    rsi_indicator = RSIIndicator(close=close_prices, window=RSI_PERIOD)
    df['RSI'] = rsi_indicator.rsi()
    sma_short_indicator = SMAIndicator(close=close_prices, window=SMA_SHORT_PERIOD)
    df[f'SMA_{SMA_SHORT_PERIOD}'] = sma_short_indicator.sma_indicator()
    sma_long_indicator = SMAIndicator(close=close_prices, window=SMA_LONG_PERIOD)
    df[f'SMA_{SMA_LONG_PERIOD}'] = sma_long_indicator.sma_indicator()
    df['RecentHigh'] = df['High'].rolling(window=SMA_SHORT_PERIOD, min_periods=1).max()
    return df.iloc[-1]

# --- Alpaca Functions ---
def get_alpaca_positions(client):
    if not client: return {}
    try:
        positions = client.get_all_positions()
        current_positions = {pos.symbol: {'qty': float(pos.qty), 'entry_price': float(pos.avg_entry_price)} for pos in positions}
        print(f"Fetched {len(current_positions)} positions from Alpaca.")
        return current_positions
    except Exception as e:
        print(f"Error fetching positions from Alpaca: {e}")
        return {}

def place_alpaca_order(client, symbol, qty, side):
    if not client:
        print("Alpaca client not initialized.")
        return False
    try:
        if qty <= 0:
            print(f"Invalid quantity {qty} for {symbol}.")
            return False
        order_data = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC)
        order = client.submit_order(order_data)
        print(f"Alpaca Order for {symbol}: ID {order.id}, Status: {order.status}, Side: {order.side}, Qty: {order.qty}")
        return True
    except Exception as e:
        print(f"Error placing Alpaca order for {symbol}: {e}")
        return False

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
    print(f"\n--- Starting scan at {pd.Timestamp.now()} (UTC) ---")
    bot_managed_positions = load_bot_state()
    alpaca_actual_positions = get_alpaca_positions(alpaca_client)
    
    # Sync state
    symbols_to_remove = [s for s in bot_managed_positions if s not in alpaca_actual_positions]
    for symbol in symbols_to_remove:
        print(f"Removing {symbol} from bot state.")
        del bot_managed_positions[symbol]
    save_bot_state(bot_managed_positions)

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
        latest_indicators = calculate_technical_indicators_from_df(df_data)
        if latest_indicators is None:
            print(f"Skipping {symbol}: No indicators.")
            continue
            
        rsi = latest_indicators['RSI']
        sma_short = latest_indicators[f'SMA_{SMA_SHORT_PERIOD}']
        sma_long = latest_indicators[f'SMA_{SMA_LONG_PERIOD}']
        
        # Sell Logic
        if symbol in bot_managed_positions:
            # Implement sell logic based on indicators
            pass
        # Buy Logic
        else:
            if len(bot_managed_positions) >= MAX_POSITIONS:
                continue
            is_uptrend = (latest_price > sma_long) if not pd.isna(sma_long) else False
            is_oversold_rsi = (rsi is not None and not pd.isna(rsi) and rsi <= RSI_OVERSOLD)
            if is_uptrend and is_oversold_rsi:
                print(f"  SIGNAL: BUY {symbol}!")
                qty_to_buy = TRADE_AMOUNT_PER_ASSET / latest_price
                if asset_type == "stock":
                    qty_to_buy = int(qty_to_buy)
                if place_alpaca_order(alpaca_client, symbol, qty_to_buy, "buy"):
                    bot_managed_positions[symbol] = {'qty': qty_to_buy, 'entry_price': latest_price}
    
    save_bot_state(bot_managed_positions)
    print("\n--- Scan complete ---")

if __name__ == "__main__":
    run_trading_strategy()
