import time
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from dateutil import tz
import logging
import data_gatherer
import database

# Define the delay after market open before data is available
DATA_DELAY_MINUTES = 15

#logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stock_updater.log"),
        logging.StreamHandler()
    ]
)

def is_market_open_now():
    nyse = mcal.get_calendar('NYSE')
    now = datetime.now(tz=tz.gettz('US/Eastern'))
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False

    market_open = schedule.iloc[0]['market_open']
    market_close = schedule.iloc[0]['market_close']
    # Start pulling data only after DATA_DELAY_MINUTES past market open
    data_start_time = market_open + timedelta(minutes=DATA_DELAY_MINUTES)
    # Optionally extend close by 15 minutes if needed
    extended_close = market_close + timedelta(minutes=15)

    return data_start_time <= now <= extended_close

def get_next_market_open_time():
    nyse = mcal.get_calendar('NYSE')
    now = datetime.now(tz=tz.gettz('US/Eastern'))
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date() + timedelta(days=10))
    future_opens = schedule[schedule['market_open'] > now]
    if future_opens.empty:
        raise RuntimeError("No future market opens found in schedule.")
    # Return market_open + DATA_DELAY_MINUTES so we start after the delay
    return future_opens.iloc[0]['market_open'] + timedelta(minutes=DATA_DELAY_MINUTES)

def wait_until_next_minute():
    now = datetime.now(tz=tz.gettz('US/Eastern'))
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    sleep_duration = (next_minute - now).total_seconds()
    time.sleep(sleep_duration)

def wait_until_market_opens():
    now = datetime.now(tz=tz.gettz('US/Eastern'))
    next_open = get_next_market_open_time()
    seconds_until_open = (next_open - now).total_seconds()
    print(f"Market data available at {next_open}, sleeping for {seconds_until_open / 60:.2f} minutes.")
    if seconds_until_open > 0:
        time.sleep(seconds_until_open)

# Main loop
symbols = ["AVGO", "CRM", "COST", "JPM", "META", "BTI", "WM", "NVDA", "NFLX", "TSLA",
           "AAPL", "GOOGL", "AMZN", "MSFT", "LMT", "XOM", "BRK.B", "PLTR", "JNJ", "GS",
           "WMT", "ENB", "DUK", "CRWD", "AMD", "QCOM", "ED", "ALB", "HD", "T"]

dbConnection = database.connection_to_database('stocks')
while True:
    try:
        if is_market_open_now():
            for symbol in symbols:
                data = data_gatherer.get_latest_bar(symbol)

                if data:
                    if symbol == 'BRK.B':
                        database.submitting_to_database(dbConnection, 'BRKB', data)
                    else:
                        database.submitting_to_database(dbConnection, symbol, data)
                else:
                    logging.warning(f"No data received from data_gatherer for {symbol}")
            wait_until_next_minute()
        else:
            print(f"Market is closed or data not available yet at {datetime.now()}, waiting for next data availability...")
            wait_until_market_opens()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

        # Attempt to close and reopen the DB connection
        try:
            if dbConnection:
                dbConnection.close()
                logging.info("Database connection closed due to error.")
        except Exception as close_err:
            logging.warning(f"Failed to close database connection: {close_err}")

        try:
            dbConnection = database.connection_to_database('stocks')
            logging.info("Database connection reopened.")
        except Exception as reconnect_err:
            logging.critical(f"Failed to reconnect to the database: {reconnect_err}")
            time.sleep(60)
