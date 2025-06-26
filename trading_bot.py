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

# Crypto API (CoinGecko Free API - rate limit 100 calls/minute, 1000/day)
# No API key needed for basic usage, but be mindful of rate limits.
COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"

# List of symbols to scan (mix of stocks and crypto)
# Note: Ensure these symbols are available on Alpaca and CoinGecko
SYMBOLS_TO_SCAN = {
    "AAPL": "stock",
    "MSFT": "stock",
    "GOOGL": "stock",
    "BTCUSD": "crypto",  # Crypto symbols for CoinGecko are usually lowercase "bitcoin"
    "ETHUSD": "crypto"   # Alpaca takes BTCUSD, ETHUSD. CoinGecko takes "ethereum" etc.
}

# Trading parameters
TRADE_AMOUNT_PER_ASSET = 100  # USD value to buy/sell per trade
MAX_POSITIONS = 5  # Max number of open positions the bot will manage
STOP_LOSS_PERCENT = 0.02  # 2% stop loss from entry price
TAKE_PROFIT_PERCENT = 0.03  # 3% take profit from entry price

# Strategy parameters
LOOKBACK_PERIOD_MINUTES = 60  # Number of 5-minute bars to fetch for indicators (e.g., 60 bars for 5 hours)
RSI_PERIOD = 14  # Standard RSI period
SMA_SHORT_PERIOD = 20 # Short-term SMA for dip identification
SMA_LONG_PERIOD = 200 # Long-term SMA for trend identification (needs longer historical data)
DIP_THRESHOLD_PERCENT = 0.015  # Example: 1.5% drop from recent high/SMA to be considered a dip
RSI_OVERSOLD = 30 # RSI level to consider oversold
RSI_OVERBOUGHT = 70 # RSI level to consider overbought

# File to store bot's managed positions (local state persistence for GitHub Actions)
POSITIONS_FILE = 'positions.json'
# Path to store historical data cache (optional, for debugging/speeding up indicator calc)
HISTORICAL_DATA_CACHE_DIR = 'historical_data_cache'


# --- Initialization ---
def initialize_alpaca_client():
    """Initializes and returns the Alpaca TradingClient."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Alpaca API keys not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY as GitHub Secrets.")
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
        # yfinance interval '5m' for 5-minute bars.
        # The 'period' parameter for yfinance must be a string like '1d', '5d', '1mo', etc.
        # We'll fetch a few days of data to ensure we have enough for our indicators.
        # A 5-day period should be sufficient for 5-minute interval data.
        data = yf.download(symbol, interval=interval, period="5d", progress=False)
        if data.empty:
            print(f"No stock data fetched for {symbol}.")
            return None
        # Ensure we have at least enough data for the longest SMA
        if len(data) < SMA_LONG_PERIOD:
            print(f"Not enough historical data for {symbol} to calculate {SMA_LONG_PERIOD}-period SMA. Need at least {SMA_LONG_PERIOD} bars, got {len(data)}.")
            # Depending on the strategy, you might want to return None or handle this differently.
            # For now, we will proceed, and the SMA calculation will result in NaN, which is handled later.

        # Calculate the number of bars needed based on the lookback period in minutes
        bars_needed = lookback_minutes // int(interval.replace('m', ''))
        return data.tail(bars_needed) # Return the most recent 'n' bars
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def get_crypto_data(symbol, interval='5m', lookback_minutes=LOOKBACK_PERIOD_MINUTES):
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
        # Fetch historical hourly data (closest to 5-min for indicator calculation)
        hourly_url = f"{COINGECKO_API_BASE_URL}/coins/{coingecko_id}/market_chart?vs_currency=usd&days=7" # Fetch 7 days of hourly data
        hourly_response = requests.get(hourly_url).json()
        prices = [p[1] for p in hourly_response.get('prices', [])]
        timestamps = [p[0] for p in hourly_response.get('prices', [])]

        if not prices:
            print(f"No historical crypto data fetched for {symbol}.")
            return None

        # Create a DataFrame for technical analysis
        df = pd.DataFrame(prices, columns=['Close'])
        df.index = pd.to_datetime(timestamps, unit='ms') # Set timestamps as index
        df.index.name = 'Datetime' # Name the index

        # Get latest current price separately
        current_price_url = f"{COINGECKO_API_BASE_URL}/simple/price?ids={coingecko_id}&vs_currencies=usd"
        current_response = requests.get(current_price_url).json()
        latest_price = current_response.get(coingecko_id, {}).get('usd')

        df['Open'] = df['High'] = df['Low'] = df['Close'] # Simplistic for indicator calc
        df['Volume'] = 0 # Dummy volume

        if len(df) < SMA_LONG_PERIOD:
             print(f"Warning: Not enough historical data for {symbol} to calculate {SMA_LONG_PERIOD}-period SMA for crypto. Need at least {SMA_LONG_PERIOD} bars, got {len(df)}. Strategy may be impaired.")

        if latest_price:
            current_time = pd.Timestamp.now(tz='UTC')
            new_row = pd.DataFrame([{'Open': latest_price, 'High': latest_price, 'Low': latest_price, 'Close': latest_price, 'Volume': 0}], index=[current_time])
            new_row.index.name = 'Datetime'
            df = pd.concat([df, new_row]).tail(lookback_minutes)
        else:
            latest_price = df['Close'].iloc[-1] # Fallback to last historical close

        df['latest_price'] = latest_price
        return df

    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {e}")
        return None


def get_latest_price_from_data(data, asset_type):
    """Extracts the latest price from the fetched data DataFrame."""
    if data is None or data.empty:
        return None
    if asset_type == "stock":
        return data['Close'].iloc[-1]
    elif asset_type == "crypto":
        return data['latest_price'].iloc[-1]
    return None

def calculate_technical_indicators_from_df(df):
    """Calculates RSI, Short SMA, Long SMA, and Recent High from a DataFrame."""
    if df is None or df.empty:
        return None

    # Calculate RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=RSI_PERIOD)
    df['RSI'] = rsi_indicator.rsi()

    # Calculate Short SMA
    sma_short_indicator = SMAIndicator(close=df['Close'], window=SMA_SHORT_PERIOD)
    df[f'SMA_{SMA_SHORT_PERIOD}'] = sma_short_indicator.sma_indicator()

    # Calculate Long SMA
    sma_long_indicator = SMAIndicator(close=df['Close'], window=SMA_LONG_PERIOD)
    df[f'SMA_{SMA_LONG_PERIOD}'] = sma_long_indicator.sma_indicator()

    # Calculate a simple "recent high" within the last few bars
    df['RecentHigh'] = df['High'].rolling(window=SMA_SHORT_PERIOD, min_periods=1).max()

    return df.iloc[-1] # Return the latest calculated indicators


# --- Alpaca Brokerage Functions ---

def get_alpaca_positions(client):
    """Fetches current open positions from Alpaca."""
    if not client: return {}
    try:
        positions = client.get_all_positions()
        current_positions = {}
        for pos in positions:
            current_positions[pos.symbol] = {
                'qty': float(pos.qty),
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_plpc': float(pos.unrealized_plpc)
            }
        print(f"Fetched {len(current_positions)} positions from Alpaca.")
        return current_positions
    except Exception as e:
        print(f"Error fetching positions from Alpaca: {e}")
        return {}

def place_alpaca_order(client, symbol, qty, side):
    """Places a market order via Alpaca API."""
    if not client:
        print("Alpaca client not initialized, cannot place order.")
        return False
    try:
        if qty <= 0:
            print(f"Invalid quantity {qty} for order on {symbol}. Skipping.")
            return False

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        order = client.submit_order(order_data)
        print(f"Alpaca Order submitted for {symbol}: ID {order.id}, Status: {order.status}, Side: {order.side}, Qty: {order.qty}")
        return True
    except Exception as e:
        print(f"Error placing Alpaca order for {symbol} ({side} {qty}): {e}")
        return False

# --- Bot State Management (persisting positions) ---

def load_bot_state():
    """Loads the bot's managed positions from a JSON file."""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                state = json.load(f)
                print(f"Loaded bot state from {POSITIONS_FILE}: {len(state)} positions.")
                return state
        except json.JSONDecodeError as e:
            print(f"Error decoding {POSITIONS_FILE}: {e}. Starting with empty state.")
            return {}
    return {}

def save_bot_state(state):
    """Saves the bot's managed positions to a JSON file."""
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"Bot state saved to {POSITIONS_FILE}.")
    except Exception as e:
        print(f"Error saving bot state to {POSITIONS_FILE}: {e}")

# --- Main Trading Logic ---

def run_trading_strategy():
    """Executes the trading strategy: scan, buy, and sell."""
    print(f"\n--- Starting 5-minute scan at {pd.Timestamp.now()} (UTC) ---")

    bot_managed_positions = load_bot_state()
    alpaca_actual_positions = get_alpaca_positions(alpaca_client)

    symbols_to_remove = []
    for symbol in bot_managed_positions.keys():
        if symbol not in alpaca_actual_positions:
            print(f"Warning: Bot was tracking {symbol}, but it's not in Alpaca positions. Removing from bot's state.")
            symbols_to_remove.append(symbol)
    for symbol in symbols_to_remove:
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
            print(f"Skipping {symbol}: No sufficient data available.")
            continue

        latest_price = get_latest_price_from_data(df_data, asset_type)
        if latest_price is None:
            print(f"Skipping {symbol}: Could not get latest price from data.")
            continue

        latest_indicators = calculate_technical_indicators_from_df(df_data)
        if latest_indicators is None:
            print(f"Skipping {symbol}: Could not calculate indicators.")
            continue

        rsi = latest_indicators['RSI']
        sma_short = latest_indicators[f'SMA_{SMA_SHORT_PERIOD}']
        sma_long = latest_indicators[f'SMA_{SMA_LONG_PERIOD}']
        recent_high = latest_indicators['RecentHigh']

        print(f"  Current Price: {latest_price:.4f}, RSI: {rsi:.2f}, SMA_{SMA_SHORT_PERIOD}: {sma_short:.4f}, SMA_{SMA_LONG_PERIOD}: {sma_long:.4f}, Recent High ({SMA_SHORT_PERIOD} bars): {recent_high:.4f}")

        if symbol in bot_managed_positions:
            pos_data = bot_managed_positions[symbol]
            entry_price = pos_data['entry_price']
            target_sell_price = pos_data['target_sell_price']
            stop_loss_price = pos_data['stop_loss_price']
            qty = pos_data['qty']

            print(f"  Position for {symbol}: Qty={qty}, Entry={entry_price:.4f}, Target={target_sell_price:.4f}, StopLoss={stop_loss_price:.4f}")

            if latest_price >= target_sell_price:
                print(f"  SIGNAL: SELL {symbol} - Take Profit reached!")
                if place_alpaca_order(alpaca_client, symbol, qty, "sell"):
                    del bot_managed_positions[symbol]
                else:
                    print(f"  WARNING: Failed to place SELL order for {symbol}. Will retry on next run.")
                continue

            if latest_price <= stop_loss_price:
                print(f"  SIGNAL: SELL {symbol} - Stop Loss triggered!")
                if place_alpaca_order(alpaca_client, symbol, qty, "sell"):
                    del bot_managed_positions[symbol]
                else:
                    print(f"  WARNING: Failed to place SELL order for {symbol}. Will retry on next run.")
                continue

            print(f"  Holding {symbol}. No sell signal yet.")

        else:
            if len(bot_managed_positions) >= MAX_POSITIONS:
                print(f"  Skipping BUY for {symbol}: Max positions ({MAX_POSITIONS}) already open.")
                continue

            is_uptrend = (latest_price > sma_long) if not pd.isna(sma_long) else False
            is_significant_dip = False
            if not pd.isna(recent_high) and recent_high > 0:
                price_drop_from_high = (recent_high - latest_price) / recent_high
                is_significant_dip = price_drop_from_high >= DIP_THRESHOLD_PERCENT

            is_oversold_rsi = (rsi is not None and not pd.isna(rsi) and rsi <= RSI_OVERSOLD)

            print(f"  Buy Conditions for {symbol}: Uptrend={is_uptrend}, SignificantDip={is_significant_dip}, OversoldRSI={is_oversold_rsi}")

            if is_uptrend and is_significant_dip and is_oversold_rsi:
                print(f"  SIGNAL: BUY {symbol} - Conditions met!")

                qty_to_buy_float = TRADE_AMOUNT_PER_ASSET / latest_price

                if asset_type == "stock":
                    qty_to_buy = max(1, int(qty_to_buy_float))
                elif asset_type == "crypto":
                    qty_to_buy = round(qty_to_buy_float, 6)
                    if qty_to_buy < 0.00001 and symbol == "BTCUSD":
                        print("  WARN: BTCUSD quantity too small, adjusting to minimum viable.")
                        qty_to_buy = 0.00001
                    if qty_to_buy < 0.001 and symbol == "ETHUSD":
                        print("  WARN: ETHUSD quantity too small, adjusting to minimum viable.")
                        qty_to_buy = 0.001

                if qty_to_buy > 0:
                    if place_alpaca_order(alpaca_client, symbol, qty_to_buy, "buy"):
                        bot_managed_positions[symbol] = {
                            'qty': qty_to_buy,
                            'entry_price': latest_price,
                            'target_sell_price': latest_price * (1 + TAKE_PROFIT_PERCENT),
                            'stop_loss_price': latest_price * (1 - STOP_LOSS_PERCENT),
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                    else:
                        print(f"  WARNING: Failed to place BUY order for {symbol}.")
                else:
                    print(f"  Skipping BUY for {symbol}: Calculated quantity is 0 or too small.")
            else:
                print(f"  No BUY signal for {symbol}.")

    save_bot_state(bot_managed_positions)
    print("\n--- Scan and trading logic complete ---")

if __name__ == "__main__":
    run_trading_strategy()
