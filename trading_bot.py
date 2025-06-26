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
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
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
MAX_POSITION_SIZE_PERCENT = 0.10

ATR_PERIOD = 14
ATR_STOP_MULTIPLE = 2.0
ATR_TAKE_PROFIT_MULTIPLE = 4.0

TIME_INTERVAL = TimeFrame(1, TimeFrameUnit.Day)
RSI_PERIOD = 14
SMA_LONG_PERIOD = 200
DIP_THRESHOLD_PERCENT = 0.02
RSI_OVERSOLD = 35
DIP_ROLLING_PERIOD = 20

MARKET_INDEX_SYMBOL = 'SPY'
MARKET_REGIME_SMA_PERIOD = 50

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
        # CORRECTED: The 'feed' parameter is removed from the client initialization.
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logging.info("Trading and Data clients initialized.")
        return trading_client, data_client
    except Exception as e:
        logging.error(f"Error initializing Alpaca clients: {e}")
        return None, None

def get_screened_symbols(trading_client, data_client):
    """Scans the market for high-quality, liquid stocks to trade."""
    logging.info("--- Starting Market Scan/Screening Process ---")
    
    try:
        request_params = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = trading_client.get_all_assets(request_params)
        tradable_assets = [asset for asset in assets if asset.tradable and getattr(asset, 'easy_to_borrow', False)]
        logging.info(f"Found {len(tradable_assets)} tradable US equity assets.")

    except Exception as e:
        logging.error(f"Failed to get assets from Alpaca: {e}")
        return []

    qualified_symbols = []
    chunk_size = 200
    for i in range(0, len(tradable_assets), chunk_size):
        asset_chunk = tradable_assets[i:i + chunk_size]
        symbols_chunk = [asset.symbol for asset in asset_chunk]
        
        try:
            # CORRECTED: Added feed='iex' to the data request.
            request_params = StockLatestBarRequest(symbol_or_symbols=symbols_chunk, feed='iex')
            latest_bars = data_client.get_stock_latest_bar(request_params)
            
            for symbol, bar in latest_bars.items():
                if bar:
                    dollar_volume = bar.volume * bar.close
                    if (dollar_volume > MIN_AVG_DOLLAR_VOLUME and
                        bar.close > MIN_SHARE_PRICE):
                        qualified_symbols.append({
                            "symbol": symbol,
                            "dollar_volume": dollar_volume
                        })
        except Exception as e:
            logging.warning(f"Could not process chunk {i // chunk_size + 1}: {e}")
        time.sleep(1) 

    qualified_symbols.sort(key=lambda x: x['dollar_volume'], reverse=True)
    final_symbols = [d['symbol'] for d in qualified_symbols[:MAX_SYMBOLS_TO_ANALYZE]]
    
    logging.info(f"Screening complete. Found {len(qualified_symbols)} qualified stocks.")
    logging.info(f"Analyzing top {len(final_symbols)} most liquid symbols: {final_symbols[:10]}...")
    
    return final_symbols

def get_historical_data(symbol, data_client, days=365):
    """Fetches historical daily bar data for a given symbol."""
    try:
        start_time = datetime.now() - timedelta(days=days)
        # CORRECTED: Added feed='iex' to the data request.
        request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TIME_INTERVAL, start=start_time, feed='iex')
        bars = data_client.get_stock_bars(request_params).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.loc[symbol]
        min_bars_needed = max(SMA_LONG_PERIOD, ATR_PERIOD, RSI_PERIOD)
        if bars.empty or len(bars) < min_bars_needed:
            return None
        return bars
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def calculate_technical_indicators(df):
    """
    Calculates technical indicators manually using pandas.
    """
    if df is None: return None, None, None, None
    
    max_period = max(SMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD)
    if len(df) < max_period: return None, None, None, None

    close_prices = df['close'].squeeze()
    
    sma_long = close_prices.rolling(window=SMA_LONG_PERIOD).mean().iloc[-1]
    
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    recent_high = df['high'].rolling(window=DIP_ROLLING_PERIOD, min_periods=1).max().iloc[-1]

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean().iloc[-1]

    return rsi, sma_long, recent_high, atr

def execute_bracket_order(symbol, trade_amount, take_profit_price, stop_loss_price, trading_client):
    """Submits a bracket order with a notional trade amount and precise exit prices."""
    try:
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=trade_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit={'limit_price': round(take_profit_price, 2)},
            stop_loss={'stop_price': round(stop_loss_price, 2)}
        )
        order = trading_client.submit_order(order_data=market_order_data)
        logging.info(f"SUCCESS: Submitted bracket order for {symbol} for ${trade_amount:,.2f}. Order ID: {order.id}")
        logging.info(f"  -> TP: ${take_profit_price:,.2f}, SL: ${stop_loss_price:,.2f}")
    except Exception as e:
        logging.error(f"Failed to submit bracket order for {symbol}: {e}")

def run_trading_scan():
    """Main function to screen, rank, and trade based on the enhanced strategy."""
    trading_client, data_client = initialize_clients()
    if not all([trading_client, data_client]): return

    # --- 1. Market Regime Filter ---
    logging.info(f"--- Checking Market-Wide Regime via {MARKET_INDEX_SYMBOL} ---")
    spy_data = get_historical_data(MARKET_INDEX_SYMBOL, data_client, days=100)
    if spy_data is None:
        return
    
    spy_sma = spy_data['close'].rolling(window=MARKET_REGIME_SMA_PERIOD).mean().iloc[-1]
    spy_latest_price = spy_data['close'].iloc[-1]
    
    if spy_latest_price < spy_sma:
        logging.warning(f"MARKET REGIME IS 'RISK-OFF': {MARKET_INDEX_SYMBOL} price (${spy_latest_price:,.2f}) is below its {MARKET_REGIME_SMA_PERIOD}-day SMA (${spy_sma:,.2f}). No new long trades will be placed.")
        return
    else:
        logging.info(f"MARKET REGIME IS 'RISK-ON': {MARKET_INDEX_SYMBOL} is trading above its {MARKET_REGIME_SMA_PERIOD}-day SMA.")

    # --- 2. Get Account Status and Screen for Symbols ---
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in positions}
    account_equity = float(account.equity)
    
    logging.info(f"Account Equity: ${account_equity:,.2f}.")
    logging.info(f"Currently holding {len(held_symbols)} positions: {list(held_symbols)}")
    
    if len(held_symbols) >= MAX_POSITIONS:
        logging.warning(f"Max positions ({MAX_POSITIONS}) reached. Exiting scan.")
        return

    symbols_to_scan = get_screened_symbols(trading_client, data_client)
    if not symbols_to_scan:
        logging.warning("Screener returned no symbols. Exiting scan.")
        return

    # --- 3. Analyze All Symbols and Collect Valid Signals ---
    logging.info("--- Starting Technical Analysis on Screened Stocks ---")
    buy_signals = []
    for symbol in symbols_to_scan:
        if symbol in held_symbols: continue
        
        logging.info(f"Processing {symbol}...")
        df_data = get_historical_data(symbol, data_client)
        if df_data is None: continue

        rsi, sma_long, recent_high, atr = calculate_technical_indicators(df_data)
        if any(pd.isna(x) for x in [rsi, sma_long, recent_high, atr]):
            logging.warning(f"Could not calculate all indicators for {symbol}. Skipping.")
            continue
        
        latest_price = df_data['close'].iloc[-1]
        logging.info(f"  -> Price: ${latest_price:,.2f}, RSI: {rsi:.2f}, ATR: {atr:.2f}, SMA_200: ${sma_long:,.2f}")
        
        is_uptrend = latest_price > sma_long
        is_oversold = rsi <= RSI_OVERSOLD
        is_dip = (recent_high - latest_price) / recent_high >= DIP_THRESHOLD_PERCENT
        
        if is_uptrend and is_oversold and is_dip:
            buy_signals.append({
                "symbol": symbol,
                "price": latest_price,
                "rsi": rsi,
                "atr": atr
            })
            logging.info(f"  -> VALID BUY SIGNAL FOUND for {symbol}. Added to candidate list.")
            
    # --- 4. Rank Signals and Select the Best One ---
    if not buy_signals:
        logging.info("--- Scan complete. No valid buy signals found today. ---")
        return
        
    buy_signals.sort(key=lambda x: x['rsi'])
    best_signal = buy_signals[0]
    
    logging.info(f"--- Found {len(buy_signals)} signal(s). Best candidate is {best_signal['symbol']} with RSI {best_signal['rsi']:.2f} ---")

    # --- 5. Refined Position Sizing ---
    symbol_to_buy = best_signal['symbol']
    latest_price = best_signal['price']
    atr = best_signal['atr']
    
    stop_loss_per_share = atr * ATR_STOP_MULTIPLE
    
    max_risk_per_trade_dollar = account_equity * ACCOUNT_RISK_PERCENT
    risk_based_notional_size = (max_risk_per_trade_dollar / stop_loss_per_share) * latest_price
    
    max_notional_by_cap = account_equity * MAX_POSITION_SIZE_PERCENT
    
    final_trade_amount = min(risk_based_notional_size, max_notional_by_cap)
    
    logging.info(f"--- Dynamic Position Sizing for {symbol_to_buy} ---")
    logging.info(f"  -> Max Risk per Trade: ${max_risk_per_trade_dollar:,.2f}")
    logging.info(f"  -> Risk-Based Size: ${risk_based_notional_size:,.2f}")
    logging.info(f"  -> Max Allocation Cap Size: ${max_notional_by_cap:,.2f}")
    logging.info(f"  -> FINAL POSITION SIZE: ${final_trade_amount:,.2f}")

    if float(account.buying_power) < final_trade_amount:
        logging.warning(f"Insufficient buying power (${float(account.buying_power):,.2f}) for new trade of ${final_trade_amount:,.2f}. Exiting.")
        return
        
    # --- 6. Execute Trade with ATR-based Exits ---
    stop_loss_price = latest_price - (atr * ATR_STOP_MULTIPLE)
    take_profit_price = latest_price + (atr * ATR_TAKE_PROFIT_MULTIPLE)
    
    logging.info(f"**** PLACING TRADE FOR {symbol_to_buy} ****")
    execute_bracket_order(symbol_to_buy, final_trade_amount, take_profit_price, stop_loss_price, trading_client)
            
    logging.info("--- Scan complete ---")

if __name__ == "__main__":
    run_trading_scan()
