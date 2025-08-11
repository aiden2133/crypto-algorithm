import pymysql
import logging
import config
import pytz

#For error logging
logging = logging.getLogger(__name__)


#Time for data time set to NY
NY_TZ = pytz.timezone('America/New_York')


#Saves data to MySQL
def submitting_to_database(dbConnection, ticker, data):
    """
    Submits data to a SQL database.

    Args: ticker (string): The name of the table you want to sumbit into
    Args: data (array): The data you want to submit into it.

    Returns: void, raises a pymysql.MySQLError if something goes wrong
    """
    insert_query = 'INSERT INTO '+ticker+' (DateTime, Volume, TradeCount, VWAP) VALUES (%s, %s, %s, %s)'
    #Execute the query
    my_cursor = dbConnection.cursor()
    try:
        my_cursor.executemany(insert_query,data)
        dbConnection.commit()
    except pymysql.MySQLError as e:
        if e.args[0] == 1062:
            # Duplicate entry error — ignore silently
            pass
        else:
            logging.error(f"Error saving data for {ticker}: {e}")
            dbConnection.rollback()

def connection_to_database(database_name):
    """
        Creates a connection to the database, based on credentials in config

        Returns: connection to database
        """
    dbConnection = pymysql.connect(
        host=config.SQL_HOST,
        user=config.SQL_USER,
        password=config.SQL_PASSWORD,
        database=database_name,
    )
    return dbConnection
