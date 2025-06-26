# --- TRADING BOT DISCLAIMER ---
# This script is for educational and experimental purposes only.
# Automated trading involves significant risk, including the risk of losing your entire investment.
# Past performance is not indicative of future results.
# The logic herein is not guaranteed to be profitable.
# ALWAYS run this in a paper trading environment first to understand its behavior and performance.
# You are solely responsible for any financial decisions and outcomes.

import os
import time
import pandas as pd
import logging
from datetime import datetime, timedelta

# Technical Analysis library
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# Alpaca API imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest # Corrected: Removed BracketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass # Corrected: Added OrderClass
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- Configuration ---
# Fetch API keys from environment variables for security
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Set to False to switch to a live trading account
PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "True").lower() == "true"

# Define the assets to scan. The script now handles both asset types uniformly.
SYMBOLS_TO_SCAN = {
    "AAPL": "stock",
    "MSFT": "stock",
    "GOOGL": "stock",
    "BTC/USD": "crypto", # Use Alpaca's format for crypto pairs
    "ETH/USD": "crypto"
}

# --- Strategy & Risk Management Parameters ---
TRADE_AMOUNT_PER_ASSET = 100  # Notional value for each trade in USD
MAX_POSITIONS = 5             # Maximum number of concurrent open positions
STOP_LOSS_PERCENT = 0.02      # 2% stop loss from entry price
TAKE_PROFIT_PERCENT = 0.03    # 3% take profit from entry price
SCAN_INTERVAL_SECONDS = 300   # How often to scan the market (300s = 5 minutes)

# --- Technical Indicator Parameters ---
TIME_INTERVAL = TimeFrame.Minute_5 # Timeframe for data bars
RSI_PERIOD = 14
SMA_LONG_PERIOD = 200
DIP_THRESHOLD_PERCENT = 0.015
RSI_OVERSOLD = 30

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Initialization ---
def initialize_clients():
    """Initializes and validates all necessary API clients."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logging.error("CRITICAL: Alpaca API keys not found in environment variables.")
        return None, None, None

    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING_MODE)
        account = trading_client.get_account()
        logging.info(f"Trading client initialized. Account Status: {account.status}")
        logging.info(f"Buying Power: ${float(account.buying_power):,.2f}")

        stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        crypto_data_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logging.info("Historical data clients initialized.")

        return trading_client, stock_data_client, crypto_data_client
    except Exception as e:
        logging.error(f"Error initializing Alpaca clients: {e}")
        return None, None, None

# --- Data Fetching ---
def get_historical_data(symbol, asset_type, stock_client, crypto_client):
    """Fetches historical bar data for a given symbol from Alpaca."""
    try:
        # We need enough data for the 200-period SMA. Fetching data from the last 3 days for 5-min intervals is safe.
        start_time = datetime.now() - timedelta(days=3)
        
        if asset_type == "stock":
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TIME_INTERVAL,
                start=start_time
            )
            bars = stock_client.get_stock_bars(request_params).df
        elif asset_type == "crypto":
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TIME_INTERVAL,
                start=start_time
            )
            bars = crypto_client.get_crypto_bars(request_params).df
        else:
            logging.warning(f"Unsupported asset type for {symbol}")
            return None
        
        # Alpaca returns data with a multi-index, we just need the data for our one symbol
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.loc[symbol]

        if bars.empty or len(bars) < SMA_LONG_PERIOD:
            logging.warning(f"Not enough historical data for {symbol} (needed {SMA_LONG_PERIOD}, got {len(bars)}).")
            return None
            
        return bars
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return None

# --- Indicator Calculation ---
def calculate_technical_indicators(df):
    """Calculates and returns technical indicators from a DataFrame."""
    if df is None or df.empty:
        return None, None, None
    
    close_prices = df['close'].squeeze()
    
    rsi = RSIIndicator(close=close_prices, window=RSI_PERIOD).rsi().iloc[-1]
    sma_long = SMAIndicator(close=close_prices, window=SMA_LONG_PERIOD).sma_indicator().iloc[-1]
    # Use a rolling window of 20 periods for the recent high
    recent_high = df['high'].rolling(window=20, min_periods=1).max().iloc[-1]
    
    return rsi, sma_long, recent_high

# --- Trading Logic ---
def execute_bracket_order(symbol, trading_client):
    """Submits a bracket order for the given symbol."""
    try:
        # For crypto, the API for latest trade is slightly different.
        if "/" in symbol:
             latest_price = trading_client.get_latest_crypto_trade(symbol).price
        else:
             latest_price = trading_client.get_latest_trade(symbol).price
        
        # *** CORRECTED LOGIC FOR BRACKET ORDER ***
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=TRADE_AMOUNT_PER_ASSET,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,  # Good 'til Canceled
            order_class=OrderClass.BRACKET, # Specify the order class as BRACKET
            take_profit={'limit_price': round(latest_price * (1 + TAKE_PROFIT_PERCENT), 2)},
            stop_loss={'stop_price': round(latest_price * (1 - STOP_LOSS_PERCENT), 2)}
        )
        
        order = trading_client.submit_order(order_data=market_order_data)
        logging.info(f"SUCCESS: Submitted bracket order for {symbol}. Order ID: {order.id}")
        logging.info(f"  - Entry Price (approx): ${latest_price:,.2f}")
        logging.info(f"  - Take Profit Price: ${market_order_data.take_profit['limit_price']:,.2f}")
        logging.info(f"  - Stop Loss Price: ${market_order_data.stop_loss['stop_price']:,.2f}")
        
    except Exception as e:
        logging.error(f"Failed to submit bracket order for {symbol}: {e}")

# --- Main Bot Loop ---
def run_trading_bot():
    """Main function to run the trading bot continuously."""
    trading_client, stock_data_client, crypto_data_client = initialize_clients()
    if not all([trading_client, stock_data_client, crypto_data_client]):
        return # Exit if clients failed to initialize

    while True:
        try:
            logging.info("--- Starting new scan cycle ---")
            
            # 1. Get current account status and positions from the broker (source of truth)
            account = trading_client.get_account()
            positions = trading_client.get_all_positions()
            held_symbols = {p.symbol for p in positions}
            logging.info(f"Currently holding {len(held_symbols)} positions: {list(held_symbols)}")

            # 2. Check if we have room for new positions
            if len(held_symbols) >= MAX_POSITIONS:
                logging.warning(f"Max positions ({MAX_POSITIONS}) reached. Will not scan for new trades.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # 3. Check for sufficient buying power
            if float(account.buying_power) < TRADE_AMOUNT_PER_ASSET:
                logging.warning("Insufficient buying power to place a new trade. Pausing.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # 4. Iterate through symbols and check for trade signals
            for symbol, asset_type in SYMBOLS_TO_SCAN.items():
                if symbol in held_symbols:
                    logging.info(f"Already hold a position in {symbol}, skipping scan.")
                    continue
                
                logging.info(f"Processing {symbol} ({asset_type})...")
                
                df_data = get_historical_data(symbol, asset_type, stock_data_client, crypto_data_client)
                
                if df_data is None:
                    continue # Error or not enough data, skip to next symbol

                latest_price = df_data['close'].iloc[-1]
                rsi, sma_long, recent_high = calculate_technical_indicators(df_data)

                if pd.isna(rsi) or pd.isna(sma_long) or pd.isna(recent_high):
                    logging.warning(f"Could not calculate all indicators for {symbol}. Skipping.")
                    continue
                
                logging.info(f"  -> Price: ${latest_price:,.2f}, RSI: {rsi:.2f}, SMA_200: ${sma_long:,.2f}, Recent High: ${recent_high:,.2f}")

                # --- The Buy Strategy Logic ---
                is_uptrend = latest_price > sma_long
                is_oversold = rsi <= RSI_OVERSOLD
                is_dip = (recent_high - latest_price) / recent_high >= DIP_THRESHOLD_PERCENT
                
                logging.info(f"  -> Buy conditions: Uptrend={is_uptrend}, Oversold={is_oversold}, Dip={is_dip}")

                if is_uptrend and is_oversold and is_dip:
                    logging.info(f"**** BUY SIGNAL DETECTED FOR {symbol} ****")
                    execute_bracket_order(symbol, trading_client)
                    # Small sleep after an order to allow broker state to update
                    time.sleep(5)
                    # Break the loop to re-fetch positions in the next cycle
                    break
            
            logging.info(f"--- Scan complete. Waiting for {SCAN_INTERVAL_SECONDS} seconds. ---")
            time.sleep(SCAN_INTERVAL_SECONDS)

        except Exception as e:
            logging.critical(f"A critical error occurred in the main loop: {e}")
            logging.info("Restarting scan in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    run_trading_bot()
