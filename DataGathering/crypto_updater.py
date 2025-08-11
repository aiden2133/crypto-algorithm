import time
import logging
import data_gatherer
import database
from datetime import datetime, timedelta

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crypto_updater.log"),
        logging.StreamHandler()
    ]
)

# Configurable wait interval (in minutes)
WAIT_INTERVAL_MINUTES = 3  # Change to 5 if needed

symbols = [
    'BTC/USD','ETH/USD','XRP/USD','USDT/USD','SOL/USD','DOGE/USD','USDC/USD',
    'AVAX/USD','DOT/USD','LINK/USD','LTC/USD','SHIB/USD','BCH/USD','UNI/USD',
    'GRT/USD','AAVE/USD','SUSHI/USD','MKR/USD','BAT/USD','YFI/USD'
]

def wait_until_next_interval(interval_minutes):
    now = datetime.now()
    # Round up to the next interval
    next_time = (now + timedelta(minutes=interval_minutes)).replace(second=0, microsecond=0)
    next_time = next_time - timedelta(minutes=next_time.minute % interval_minutes)
    if next_time <= now:
        next_time += timedelta(minutes=interval_minutes)
    sleep_duration = (next_time - now).total_seconds()
    #logging.info(f"Sleeping for {sleep_duration:.2f} seconds until next interval...")
    time.sleep(sleep_duration)

dbConnection = database.connection_to_database('crypto')

try:
    while True:
        for symbol in symbols:
            data = data_gatherer.get_latest_bar(symbol, is_crypto=True)

            if data:
                database.submitting_to_database(dbConnection, symbol[:-4], data)
            else:
                logging.warning(f"No data received from data_gatherer for {symbol}")

        wait_until_next_interval(WAIT_INTERVAL_MINUTES)

except Exception as e:
    logging.error(f"Error occurred: {e}")
    dbConnection.close()
