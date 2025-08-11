import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from database import connection_to_database
from config import SQL_USER, SQL_PASSWORD, SQL_HOST, SQL_DATABASE

# ========== CONFIG ==========
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_model_path(symbol):
    return os.path.join(MODEL_DIR, f'{symbol}.pt')

def save_model(symbol, model):
    path = get_model_path(symbol)
    torch.save(model.state_dict(), path)

def load_model(symbol, input_size):
    path = get_model_path(symbol)
    if os.path.exists(path):
        model = DeepEnsemblePriceChangeNet(input_size).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    return None

# ========== DATABASE CONNECTIONS ==========
def get_sqlalchemy_engine():
    uri = f"mysql+pymysql://{SQL_USER}:{SQL_PASSWORD}@{SQL_HOST}/{SQL_DATABASE}"
    return create_engine(uri)


def fetch_price_data(symbol, start, end):
    engine = get_sqlalchemy_engine()
    print(symbol)
    table_name = symbol.split('-')[0] # e.g. 'BTC', 'ETH'
    print(f"Using table: {table_name}")
    query = f"""
        SELECT DateTime, VWAP FROM crypto.{table_name}
        WHERE DateTime BETWEEN %s AND %s
        ORDER BY DateTime ASC
    """
    df = pd.read_sql(query, engine, params=(start, end))
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    if 'VWAP' not in df.columns:
        raise ValueError(f"'VWAP' column not found in data for {table_name}")

    return df


# ========== MODEL DEFINITION ==========

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        out = self.bn2(self.fc2(out))
        out += identity
        return F.leaky_relu(out, negative_slope=0.01)

class DeepEnsemblePriceChangeNet(nn.Module):
    def __init__(self, input_size):
        super(DeepEnsemblePriceChangeNet, self).__init__()

        # Removed BatchNorm here to avoid suppressing early gradients
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3)
        )

        self.branch1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            ResidualBlock(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01)
        )

        self.branch2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            ResidualBlock(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_combined = torch.cat([x1, x2], dim=1)
        return self.output_layer(x_combined)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ========== FEATURES ==========
def build_features(df):
    df = df.sort_values('DateTime').copy()
    epsilon = 1e-8

    # Basic lag
    df['VWAP_prev1'] = df['VWAP'].shift(1)

    # Returns over multiple horizons
    df['return_1'] = df['VWAP'].pct_change(1)
    df['return_5'] = df['VWAP'].pct_change(5)
    df['return_10'] = df['VWAP'].pct_change(10)

    # Rolling volatility
    df['volatility_5'] = df['return_1'].rolling(window=5).std()
    df['volatility_10'] = df['return_1'].rolling(window=10).std()
    df['volatility_20'] = df['return_1'].rolling(window=20).std()

    # Momentum features
    df['momentum_5'] = df['VWAP'] - df['VWAP'].shift(5)
    df['momentum_10'] = df['VWAP'] - df['VWAP'].shift(10)

    # Price velocity (first diff)
    df['price_velocity'] = df['VWAP'].diff()

    # Target variable
    df['target'] = (df['VWAP'].shift(-1) > df['VWAP']).astype(int)

    # Clean up
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Select features to keep
    feature_cols = [
        'VWAP_prev1',
        'return_1', 'return_5', 'return_10',
        'volatility_5', 'volatility_10', 'volatility_20',
        'momentum_5', 'momentum_10',
        'price_velocity'
    ]

    X_raw = df[feature_cols].copy()

    # Use robust scaling (better for outliers)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=[col + '_scaled' for col in feature_cols],
        index=df.index
    )

    return X_scaled, df['target']


# ========== TRAINING ==========
def train_model(symbol, start_date, end_date, epochs=100, batch_size=64, lr=0.0003):
    print(f"\n🔧 Training model for {symbol}...")

    df = fetch_price_data(symbol, start_date, end_date)
    if df.empty or len(df) < 100:
        print(f"⚠️ Not enough data to train model for {symbol}")
        return None

    X_scaled, y = build_features(df)  # Your existing feature building
    analyze_features_vs_target(X_scaled, y)
    return None
    X, y = build_features(df)
    y = y.astype(int)

    pos_ratio = y.mean()
    print(f"📊 Class distribution: {pos_ratio:.2%} positive, {(1-pos_ratio):.2%} negative")

    split_idx = int(len(X) * 0.8)
    X_train_np, X_test_np = X.iloc[:split_idx].values, X.iloc[split_idx:].values
    y_train_np, y_test_np = y.iloc[:split_idx].values, y.iloc[split_idx:].values

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    model = DeepEnsemblePriceChangeNet(input_dim).to(device)
    model.apply(init_weights)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test_np, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test_np, y_pred))

    print("\n🔍 Sample predictions vs actuals:")
    for pred, actual in zip(y_pred[:10], y_test_np[:10]):
        print(f"Predicted: {pred}, Actual: {int(actual)}")

    save_model(symbol, model)

    return model

# ========== PREDICTION ==========
def predict(symbol, input_df):
    """
    Predict probabilities for price increase.

    input_df: DataFrame with scaled feature columns matching train features.
    """
    input_size = input_df.shape[1]
    model = load_model(symbol, input_size)
    if model is None:
        raise ValueError(f"No trained model found for {symbol}")

    model.eval()
    X_tensor = torch.tensor(input_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def analyze_features_vs_target(features_df: pd.DataFrame, target_series: pd.Series, top_n_corr=10):
    """
    Analyze and visualize the relationship between features and target.

    Parameters:
        features_df (pd.DataFrame): DataFrame of features (already processed/scaled or raw).
        target_series (pd.Series): Binary target (0/1).
        top_n_corr (int): Number of top correlated features to show.
    """

    print("\n=== Feature Descriptive Statistics by Target Class ===")
    stats = features_df.copy()
    stats['target'] = target_series.values
    grouped = stats.groupby('target').describe().transpose()
    print(grouped)

    # Calculate correlation with target
    corr = features_df.apply(lambda x: x.corr(target_series))
    corr = corr.dropna().sort_values(ascending=False)
    print(f"\n=== Top {top_n_corr} Features Correlated with Target ===")
    print(corr.head(top_n_corr))

    # Plot distributions of top correlated features
    top_features = corr.head(top_n_corr).index.tolist()

    for feature in top_features:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(features_df[feature][target_series == 0], label='Target=0', fill=True)
        sns.kdeplot(features_df[feature][target_series == 1], label='Target=1', fill=True)
        plt.title(f'Distribution of {feature} by Target Class')
        plt.legend()
        plt.tight_layout()
        plt.show()


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    # Replace with your real data range and symbol
    TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "AVAX-USD",
               "LINK-USD", "LTC-USD", "SHIB-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "XTZ-USD"]

    START_DATE = '2023-01-01'
    END_DATE = '2025-08-01'

    for ticker in TICKERS:
        model = train_model(ticker, START_DATE, END_DATE, epochs=50)

    # If you want to predict, prepare the input_df with the same feature transformations
    # Example: predict(SYMBOL, input_df)
