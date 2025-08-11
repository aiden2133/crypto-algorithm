import pandas as pd
from sqlalchemy import create_engine
from config import *
import os
import joblib
from database import connection_to_database


# Memory for tracking past N predictions per symbol
RECENT_PREDICTIONS = {}

# ========== CONFIG ==========
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(symbol):
    return os.path.join(MODEL_DIR, f'{symbol}.joblib')

def save_model(symbol, model, recent_perf=None):
    path = get_model_path(symbol)
    joblib.dump((model, recent_perf or []), path)

def load_model(symbol):
    path = get_model_path(symbol)
    if os.path.exists(path):
        return joblib.load(path)
    return None, []

# ========== FEATURES ==========
def build_features(df):
    df = df.sort_values('DateTime').copy()
    df['VWAP_prev1'] = df['VWAP'].shift(1)
    df['VWAP_prev2'] = df['VWAP'].shift(2)
    df['VWAP_prev3'] = df['VWAP'].shift(3)
    df['ma_5'] = df['VWAP'].rolling(window=5).mean()
    df['ma_10'] = df['VWAP'].rolling(window=10).mean()
    df['delta_t'] = df['DateTime'].diff().dt.total_seconds().fillna(180)

    df['target'] = (df['VWAP'].shift(-1) > df['VWAP']).astype(int)
    df.dropna(inplace=True)

    feature_cols = ['VWAP', 'VWAP_prev1', 'VWAP_prev2', 'VWAP_prev3', 'ma_5', 'ma_10', 'delta_t']
    return df[feature_cols], df['target']

def get_sqlalchemy_engine():
    uri = f"mysql+pymysql://{SQL_USER}:{SQL_PASSWORD}@{SQL_HOST}/{SQL_DATABASE}"
    return create_engine(uri)

def get_recent_data(symbol, lookback=20):
    engine = get_sqlalchemy_engine()
    query = f"""
        SELECT DateTime, VWAP FROM {symbol}
        ORDER BY DateTime DESC
        LIMIT %s
    """
    df = pd.read_sql(query, engine, params=(lookback,))
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df.sort_values('DateTime')


def fetch_latest_features(symbol):
    """
    Fetch the most recent rows and build a feature vector for prediction.
    """
    db = connection_to_database("crypto")
    query = f"""
        SELECT DateTime, VWAP
        FROM {symbol}
        ORDER BY DateTime DESC
        LIMIT 15
    """

    df = pd.read_sql(query, db)
    if df.empty or len(df) < 10:
        return None  # Not enough data

    df = df.sort_values('DateTime').copy()

    # --- Build Features (same as build_features) ---
    df['VWAP_prev1'] = df['VWAP'].shift(1)
    df['VWAP_prev2'] = df['VWAP'].shift(2)
    df['VWAP_prev3'] = df['VWAP'].shift(3)

    df['ma_5'] = df['VWAP'].rolling(window=5).mean()
    df['ma_10'] = df['VWAP'].rolling(window=10).mean()

    df['price_velocity'] = df['VWAP'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()

    df.dropna(inplace=True)

    if df.empty:
        return None

    # Select only the most recent row for prediction
    feature_cols = [
        'VWAP', 'VWAP_prev1', 'VWAP_prev2', 'VWAP_prev3',
        'ma_5', 'ma_10',
        'price_velocity', 'price_acceleration'
    ]

    latest_row = df[feature_cols].iloc[[-1]]
    return latest_row

def get_prediction(symbol):
    model, _ = load_model(symbol)
    if model is None:
        return 'hold', 0

    # You’d need to fetch or generate the latest row of features
    latest_df = fetch_latest_features(symbol)  # you’d write this
    if latest_df is None:
        return 'hold', 0

    pred = model.predict(latest_df)[0]

    decision = 'buy' if pred > 0 else 'sell' if pred < 0 else 'hold'
    return decision, pred
