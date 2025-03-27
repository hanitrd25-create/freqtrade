import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas_ta as ta

# === CONFIG ===
PAIR = "BTC/USDT"
EXCHANGE = "binance"
TIMEFRAME = "5m"
DATA_PATH = f"user_data/data/{EXCHANGE}/{PAIR.replace('/', '_')}-{TIMEFRAME}.feather"
MODEL_PATH = "ml/trained_model.pkl"

# === Load real OHLCV data ===
print(f"Loading data from {DATA_PATH}...")
df = pd.read_feather(DATA_PATH)

# Basic cleaning
df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
df = df.sort_values('date')
df.reset_index(drop=True, inplace=True)

# === Feature Engineering ===
df['price_delta'] = df['close'].diff().fillna(0)
df['ema_10'] = df['close'].ewm(span=10).mean()
df['ema_50'] = df['close'].ewm(span=50).mean()
df['rsi'] = ta.rsi(df['close'], length=14)
df['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)

# === Labels ===
df['future_close'] = df['close'].shift(-1)
df['target'] = (df['future_close'] > df['close']).astype(int)
df.dropna(inplace=True)

features = ['price_delta', 'ema_10', 'ema_50', 'volume', 'rsi', 'atr']
X = df[features]
y = df['target']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Train model ===
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)

# === Save model ===
os.makedirs("ml", exist_ok=True)
joblib.dump(pipe, MODEL_PATH)
print(f"✅ Model trained and saved to {MODEL_PATH}")