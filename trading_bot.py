# --- TRADING BOT DISCLAIMER ---
# This script is for educational and experimental purposes only.
# It includes a dynamic screener and manual indicator calculations.
# It is designed to be run once per day after market close.

import os
import time
import pandas as pd
import logging
from datetime import datetime, timedelta

# Alpaca API imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "True").lower() == "true"

# --- Screener & Strategy Parameters ---
MAX_SYMBOLS_TO_ANALYZE = 100 # Analyze the top 100 most liquid stocks that pass the screener
MIN_SHARE_PRICE = 20.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000 # 20 Million

MAX_POSITIONS = 5
ACCOUNT_RISK_PERCENT = 0.01
STOP_LOSS_PERCENT = 0.04
TAKE_PROFIT_PERCENT = 0.08

TIME_INTERVAL = TimeFrame(1, TimeFrameUnit.Day)
RSI_PERIOD = 14
SMA_LONG_PERIOD = 200
DIP_THRESHOLD_PERCENT = 0.02
RSI_OVERSOLD = 35
DIP_ROLLING_PERIOD = 20

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def initialize_clients():
    """Initializes and returns the trading and data clients."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logging.error("CRITICAL: Alpaca API keys not found.")
        return None, None
    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING_MODE)
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logging.info("Trading and Data clients initialized.")
        return trading_client, data_client
    except Exception as e:
        logging.error(f"Error initializing Alpaca clients: {e}")
        return None, None

def get_screened_symbols(trading_client, data_client):
    """Scans the market for high-quality, liquid stocks to trade."""
    logging.info("--- Starting Market Scan/Screening Process ---")
    
    # 1. Get all US Equity assets from Alpaca
    search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = trading_client.get_assets(search_params)
    tradable_assets = [asset for asset in assets if asset.tradable and getattr(asset, 'easy_to_borrow', False)]
    logging.info(f"Found {len(tradable_assets)} tradable US equity assets.")

    # 2. Filter assets based on liquidity and price
    qualified_symbols = []
    chunk_size = 200 # Process symbols in chunks to avoid overwhelming the API
    for i in range(0, len(tradable_assets), chunk_size):
        asset_chunk = tradable_assets[i:i + chunk_size]
        symbols_chunk = [asset.symbol for asset in asset_chunk]
        
        try:
            # Fetch snapshots for volume and price data
            snapshots = data_client.get_stock_snapshots(symbols_chunk)
            
            for symbol, snapshot in snapshots.items():
                if snapshot and snapshot.daily_bar and snapshot.latest_trade:
                    avg_dollar_volume = snapshot.daily_bar.volume * snapshot.daily_bar.close
                    if (avg_dollar_volume > MIN_AVG_DOLLAR_VOLUME and
                        snapshot.latest_trade.price > MIN_SHARE_PRICE):
                        qualified_symbols.append({
                            "symbol": symbol,
                            "dollar_volume": avg_dollar_volume
                        })
        except Exception as e:
            logging.warning(f"Could not process chunk {i // chunk_size + 1}: {e}")
        time.sleep(1) # Pause between chunks to respect rate limits

    # 3. Sort by dollar volume and return the top symbols
    qualified_symbols.sort(key=lambda x: x['dollar_volume'], reverse=True)
    
    final_symbols = [d['symbol'] for d in qualified_symbols[:MAX_SYMBOLS_TO_ANALYZE]]
    
    logging.info(f"Screening complete. Found {len(qualified_symbols)} qualified stocks.")
    logging.info(f"Analyzing top {len(final_symbols)} most liquid symbols: {final_symbols[:10]}...")
    
    return final_symbols

def get_historical_data(symbol, data_client):
    """Fetches historical daily bar data for a given symbol."""
    try:
        start_time = datetime.now() - timedelta(days=365) # Fetch enough data for 200-day SMA
        request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TIME_INTERVAL, start=start_time)
        bars = data_client.get_stock_bars(request_params).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.loc[symbol]
        if bars.empty or len(bars) < SMA_LONG_PERIOD:
            return None
        return bars
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def calculate_technical_indicators(df):
    """
    Calculates technical indicators manually using pandas to avoid library conflicts.
    """
    if df is None or df.empty or len(df) < SMA_LONG_PERIOD:
        return None, None, None

    close_prices = df['close'].squeeze()

    # --- Calculate SMA ---
    sma_long = close_prices.rolling(window=SMA_LONG_PERIOD).mean().iloc[-1]

    # --- Calculate RSI ---
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use Exponential Moving Average for RSI calculation
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    # --- Calculate Recent High ---
    recent_high = df['high'].rolling(window=DIP_ROLLING_PERIOD, min_periods=1).max().iloc[-1]

    return rsi, sma_long, recent_high

def execute_bracket_order(symbol, trade_amount, trading_client):
    """Submits a bracket order with a notional trade amount."""
    try:
        latest_price = trading_client.get_latest_trade(symbol).price
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=trade_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit={'limit_price': round(latest_price * (1 + TAKE_PROFIT_PERCENT), 2)},
            stop_loss={'stop_price': round(latest_price * (1 - STOP_LOSS_PERCENT), 2)}
        )
        order = trading_client.submit_order(order_data=market_order_data)
        logging.info(f"SUCCESS: Submitted bracket order for {symbol} for ${trade_amount:,.2f}. Order ID: {order.id}")
    except Exception as e:
        logging.error(f"Failed to submit bracket order for {symbol}: {e}")

# --- Main Bot Function ---
def run_trading_scan():
    """Main function to screen the market and run a single daily trading scan."""
    trading_client, data_client = initialize_clients()
    if not all([trading_client, data_client]):
        return # Exit if clients failed to initialize

    # --- DYNAMICALLY GET SYMBOLS TO SCAN ---
    symbols_to_scan = get_screened_symbols(trading_client, data_client)
    if not symbols_to_scan:
        logging.warning("Screener returned no symbols. Exiting scan.")
        return

    logging.info("--- Starting Technical Analysis on Screened Stocks ---")
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in positions}
    account_equity = float(account.equity)

    logging.info(f"Account Equity: ${account_equity:,.2f}.")
    logging.info(f"Currently holding {len(held_symbols)} positions: {list(held_symbols)}")

    if len(held_symbols) >= MAX_POSITIONS:
        logging.warning(f"Max positions ({MAX_POSITIONS}) reached. Exiting scan.")
        return

    # --- DYNAMIC RISK MANAGEMENT ---
    max_risk_per_trade = account_equity * ACCOUNT_RISK_PERCENT
    trade_amount_per_asset = max_risk_per_trade / STOP_LOSS_PERCENT
    logging.info(f"Risk config: Max risk/trade: ${max_risk_per_trade:,.2f}. Position Size: ${trade_amount_per_asset:,.2f}")

    if float(account.buying_power) < trade_amount_per_asset:
        logging.warning(f"Insufficient buying power for new trade. Exiting.")
        return

    for symbol in symbols_to_scan:
        if symbol in held_symbols:
            continue
        
        logging.info(f"Processing {symbol}...")
        df_data = get_historical_data(symbol, data_client)
        if df_data is None: continue

        latest_price = df_data['close'].iloc[-1]
        rsi, sma_long, recent_high = calculate_technical_indicators(df_data)
        if any(pd.isna(x) for x in [rsi, sma_long, recent_high]):
            logging.warning(f"Could not calculate all indicators for {symbol}. Skipping.")
            continue
        
        logging.info(f"  -> Price: ${latest_price:,.2f}, RSI: {rsi:.2f}, SMA_200: ${sma_long:,.2f}")
        
        is_uptrend = latest_price > sma_long
        is_oversold = rsi <= RSI_OVERSOLD
        is_dip = (recent_high - latest_price) / recent_high >= DIP_THRESHOLD_PERCENT
        
        logging.info(f"  -> Buy conditions: Uptrend={is_uptrend}, Oversold={is_oversold}, Dip={is_dip}")

        if is_uptrend and is_oversold and is_dip:
            logging.info(f"**** BUY SIGNAL DETECTED FOR {symbol} ****")
            execute_bracket_order(symbol, trade_amount_per_asset, trading_client)
            logging.info("Trade placed. Ending this scan cycle.")
            return # Exit after placing one trade to prevent rapid-fire orders
            
    logging.info("--- Scan complete ---")

if __name__ == "__main__":
    run_trading_scan()
