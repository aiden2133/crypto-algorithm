from predictor import get_recent_data
import os
import joblib

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

def update_model(symbol):
    model, recent_perf = load_model(symbol)
    if model is None:
        print(f"No model found for {symbol}")
        return

    df = get_recent_data(symbol, lookback=30)
    X, y = build_features(df)

    if len(X) > 0:
        model.partial_fit(X, y, classes=[0,1])
        save_model(symbol, model, recent_perf)
        print(f"Model updated for {symbol}")
