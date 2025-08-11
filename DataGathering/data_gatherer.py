from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime, timedelta, UTC
import requests
import pandas_market_calendars as mcal


from dotenv import load_dotenv
import os
import database
import time
import logging

# Set your Alpaca API credentials here or use environment variables
load_dotenv()  # Load variables from .env
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

#Fin hub
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"

#Initialize the historical data client for stocks and crypto
historical_client_stock = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
historical_client_crypto = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


#configuration settings
SLEEP_BETWEEN_REQUESTS = .2  # seconds

#logging
logging = logging.getLogger(__name__)
'''
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stock_data.log"),
        logging.StreamHandler()
    ]
)
'''

#Used for gathering the data at different intervals
TIMEFRAME_MAP = {
    "1Min": TimeFrame.Minute,
    "5Min": TimeFrame(5, "Minute"),
    "15Min": TimeFrame(15, "Minute"),
    "1Hour": TimeFrame.Hour,
    "1Day": TimeFrame.Day,
}

def get_trading_days(start, end):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start, end_date=end)
    return schedule.index.to_list()

def chunk_dates(start, end, chunk_days, is_crypto=False):
    if is_crypto:
        #Generate all calendar days between start and end
        current = start
        trading_days = []
        while current <= end:
            trading_days.append(current)
            current += timedelta(days=1)
    else:
        #Use trading days (business days) for stocks
        trading_days = get_trading_days(start, end)

    #Group days into chunks
    chunks = []
    for i in range(0, len(trading_days), chunk_days):
        chunk_start = trading_days[i]
        chunk_end = trading_days[min(i + chunk_days - 1, len(trading_days) - 1)]
        chunks.append((chunk_start, chunk_end))

    return chunks


async def get_realtime_data(symbol: str):
    try:
        stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        async def quote_handler(data):
            print(f"Realtime quote for {symbol}: {data}")

        # Subscribe to quote data
        stream.subscribe_quotes(quote_handler, symbol)

        await stream._run_forever()
    except Exception as ex:
        logging.error(f"[{symbol}] Failed to get real time data: {ex}")


def get_historical_data(
        symbol: str,
        start_date: str,
        end_date: str,
        connection = None,
        timeframe: str = "1Day",
        export_to_database: bool = False,
        crypto: bool = False,
        reset_index: bool = True
):
    """
    Get historical stock data using Alpaca API.

    Parameters:
        symbol (str): Ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in ISO format (YYYY-MM-DD)
        end_date (str): End date in ISO format (YYYY-MM-DD)
        timeframe (str): Interval between data points. Options: 1Min, 5Min, 15Min, 1Hour, 1Day
        plot (bool): If True, displays a plot of closing prices
        export_csv (bool): If True, saves the data to 'output.csv'
        reset_index (bool): If True, resets DataFrame index

    Returns:
        pd.DataFrame: Historical data
    """

    try:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe '{timeframe}'. Choose from: {list(TIMEFRAME_MAP.keys())}")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAME_MAP[timeframe],
            start=start_date,
            end=end_date
        )

        if crypto == True:
            bars = historical_client_crypto.get_crypto_bars(request_params)
        else:
            bars = historical_client_stock.get_stock_bars(request_params)


        df = bars.df

        if df.empty:
            print("No data returned.")
            return df


        if reset_index:
            df = df.reset_index()

        if export_to_database:
            if crypto:
                prepare_and_submit(df, connection, symbol[:-4])
            else:
                if symbol == 'BRK.B':
                    prepare_and_submit(df, connection, 'BRKB')
                else:
                    prepare_and_submit(df, connection, symbol)
                return

        return df

    except Exception as ex:
        logging.error(f"[{symbol}] Failed to gather data from alpaca: {ex}")
        return pd.DataFrame()


def get_latest_bar(symbol: str, is_crypto: bool = False):
    try:
        if is_crypto:
            now = datetime.now(UTC)
            start = now - timedelta(minutes=15)

            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=now
            )
            bars = historical_client_crypto.get_crypto_bars(request_params)
        else:
            delay_minutes = 15
            now = datetime.now(UTC)
            end = now - timedelta(minutes=delay_minutes)
            start = end - timedelta(minutes=15)

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )
            bars = historical_client_stock.get_stock_bars(request_params)

        if bars.df.empty:
            return None

        latest_row = bars.df.iloc[-1]
        latest_index = bars.df.index[-1]  # This is a tuple like (symbol, datetime)
        latest_datetime = latest_index[1] if isinstance(latest_index, tuple) else latest_index

        return [[latest_datetime, float(latest_row["volume"]), float(latest_row["trade_count"]), float(latest_row["vwap"])]]

    except Exception as ex:
        logging.error(f"[{symbol}] Failed to gather latest data from Alpaca: {ex}")
        return None


def get_fundamentals(symbol: str):
    """
    Fetch fundamental metrics for a stock symbol from Finnhub.

    Returns a dictionary of metrics or None on failure.
    """
    url = f"{BASE_URL}/stock/metric"
    params = {
        "symbol": symbol,
        "metric": "all",
        "token": FINNHUB_API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("metric")  # contains PE, PEG, ROE, etc.
    else:
        print(f"Error fetching fundamentals: {response.status_code} - {response.text}")
        return None


def get_financial_statements(symbol: str, statement_type: str = "ic", freq: str = "annual"):
    """
    Fetch financial statements for a stock symbol from Finnhub.

    Parameters:
        - statement_type: 'ic' (income statement), 'bs' (balance sheet), 'cf' (cash flow)
        - freq: 'annual' or 'quarterly'

    Returns a list of statement entries or None on failure.
    """
    url = f"{BASE_URL}/stock/financials-reported"
    params = {
        "symbol": symbol,
        "statement": statement_type,
        "freq": freq,
        "token": FINNHUB_API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("data")  # list of financial statement reports
    else:
        print(f"Error fetching financial statements: {response.status_code} - {response.text}")
        return None

def prepare_and_submit(df, dbConnection, ticker):
    try:
        #Getting df correct
        df = df[["timestamp", "volume", "trade_count", "vwap"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        data = df[["timestamp", "volume", "trade_count", "vwap"]].values.tolist()

        #Submit to database
        database.submitting_to_database(dbConnection, ticker, data)
    except Exception as ex:
        logging.error(f"[{ticker}] Failed to submit data to database: {ex}")


if __name__ == '__main__':
    #use to run different chunks of code
    use_crypto = False
    use_stocks = False
    testing = True

    if use_crypto:
        dbConnection = database.connection_to_database('crypto')

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 6, 20)
        symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'USDT/USD', 'SOL/USD', 'DOGE/USD', 'USDC/USD',
                   'AVAX/USD', 'DOT/USD', 'LINK/USD', 'LTC/USD', 'SHIB/USD', 'BCH/USD', 'UNI/USD',
                   'GRT/USD', 'AAVE/USD', 'SUSHI/USD', 'MKR/USD', 'BAT/USD', 'YFI/USD']

        date_chunks = chunk_dates(start_date, end_date, chunk_days=2, is_crypto = True)
        for symbol in symbols:

            for i, (start, end) in enumerate(date_chunks):

                if (i + 1) % 10 == 0:  # Prints every 10 chunks
                    print(f"\n[{i + 1}/{len(date_chunks)}] Fetching {symbol} from {start} to {end}")

                try:
                    df = get_historical_data(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        connection=dbConnection,
                        timeframe="1Min",
                        crypto=True,
                        export_to_database=True
                    )
                except Exception as e:
                    logging.error(f"Failed: {e}")

                time.sleep(SLEEP_BETWEEN_REQUESTS)

    if use_stocks:
        dbConnection = database.connection_to_database('stocks')
        symbols = ["AVGO", "CRM", "COST", "JPM", "META", "BTI", "WM", "NVDA", "NFLX", "TSLA", "AAPL", "GOOGL", "AMZN",
                   "MSFT", "LMT", "XOM", "BRK.B", "PLTR", "JNJ", "GS", "WMT", "ENB", "DUK", "CRWD", "AMD", "QCOM", "ED",
                   "ALB", "HD", "T"]

        start_date = datetime(2025, 6, 20)
        end_date = datetime(2025, 6, 20)

        date_chunks = chunk_dates(start_date, end_date, chunk_days=2)
        for symbol in symbols:
            print(f"Fetching data for {symbol}")
            print(f"Total chunks to process: {len(date_chunks)}")

            for i, (start, end) in enumerate(date_chunks):

                if (i + 1) % 10 == 0:#Prints every 10 chunks
                    print(f"\n[{i + 1}/{len(date_chunks)}] Fetching {symbol} from {start} to {end}")

                try:
                    df = get_historical_data(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        connection=dbConnection,
                        timeframe="1Min",
                        export_to_database=True
                    )
                except Exception as e:
                    logging.error(f"Failed: {e}")

                time.sleep(SLEEP_BETWEEN_REQUESTS)


    if testing:
        symbol = 'ETH/USD'
        start = datetime(2025, 6, 17)
        end = datetime(2025, 6, 19)

        df = get_latest_bar(symbol, is_crypto=True)

        print(df)



