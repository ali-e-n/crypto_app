import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import json

# ───────────────────────────── TECHNICAL INDICATORS ─────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def add_indicators(df):
    df['EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['Volume'] = df.get('Volume', 0)
    df.dropna(inplace=True)
    return df

# ───────────────────────────── DATA FETCHING ─────────────────────────────
def fetch_data(symbol="BTC-USD", interval="1d", start="2015-01-01", force_refresh=False):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol.replace('-', '_')}_{interval}.csv"

    if not os.path.exists(filename) or force_refresh:
        print(f"[INFO] Downloading fresh data for {symbol}...")
        if interval == "1h":
            df = yf.download(symbol, period="60d", interval="1h", auto_adjust=False)
        else:
            df = yf.download(symbol, start=start, interval=interval, auto_adjust=False)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = add_indicators(df)
    return df

# ───────────────────────────── DATA PREP ─────────────────────────────
def prepare_data(df, window_size=400, steps=7, train_split=0.8):
    features = ['Close', 'EMA', 'RSI', 'MACD', 'Volume']
    data = df[features].values
    split_index = int(len(data) * train_split)
    scaler = MinMaxScaler()
    scaler.fit(data[:split_index])
    scaled = scaler.transform(data)

    x, y = [], []
    for i in range(window_size, len(scaled) - steps):
        x.append(scaled[i - window_size:i])
        y.append(scaled[i:i + steps, 0])  # Predict Close only

    x, y = np.array(x), np.array(y)
    split = int(len(x) * train_split)
    return x[:split], y[:split], x[split:], y[split:], scaler

# ───────────────────────────── MODEL ─────────────────────────────
def build_model(input_shape, output_steps):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ───────────────────────────── TRAIN & PREDICT ─────────────────────────────
def run_forecast(symbol="BTC-USD", interval="1d",  label="7d", steps=7, window_size=400, epochs=10, log_callback=None):


    df = fetch_data(symbol, interval, force_refresh=True)
    if log_callback:
        log_callback(f"✔️ Data loaded: {len(df)} rows")
    else:
        print(f"[INFO] Loaded {len(df)} rows")
    x_train, y_train, x_test, y_test, scaler = prepare_data(df, window_size, steps)

    model = build_model((x_train.shape[1], x_train.shape[2]), steps)
    for epoch in range(1, epochs + 1):
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss = history.history['loss'][0]
        if log_callback:
            log_callback(f"✅ Epoch {epoch}/{epochs} — Loss: {loss:.6f}")
        else:
            print(f"[Epoch {epoch}/{epochs}] Loss: {loss:.6f}")



    predictions = model.predict(x_test)
    actual = y_test

    def inverse_scale_close_only(data, scaler, feature_count):
        reshaped = data.reshape(-1, 1)
        padded = np.zeros((reshaped.shape[0], feature_count))
        padded[:, 0] = reshaped[:, 0]
        unscaled = scaler.inverse_transform(padded)[:, 0]
        return unscaled.reshape(data.shape)

    feature_count = x_train.shape[2]
    pred_inverse = inverse_scale_close_only(predictions, scaler, feature_count)
    actual_inverse = inverse_scale_close_only(actual, scaler, feature_count)

    last_time = df.index[-1]
    delta = timedelta(days=1) if interval == "1d" else timedelta(hours=1)
    # future_dates = [last_time + delta * (i + 1) for i in range(steps)]
    # Use multiplier spacing based on label
    multiplier = {
        "1h": 1,
        "1d": 1,
        "7d": 7,
        "14d": 14,
        "30d": 30,
        "1y": 365
    }.get(label, 1)

    from dateutil.relativedelta import relativedelta

    if label == "7d":
        future_dates = [last_time + timedelta(days=7 * (i + 1)) for i in range(5)]  # 5 weeks
    elif label == "14d":
        future_dates = [last_time + timedelta(days=14 * (i + 1)) for i in range(4)]  # 2 months
    elif label == "30d":
        future_dates = [last_time + relativedelta(months=+i + 1) for i in range(5)]  # 5 months
    elif label == "1y":
        end_of_year = pd.Timestamp.now().replace(month=12, day=31)
        future_dates = pd.date_range(start=last_time + timedelta(days=1), end=end_of_year, freq="MS").to_list()
    else:
        future_dates = [last_time + delta * (i + 1) for i in range(steps)]



    output_dir = f"static/forecasts/{symbol}_{label}"  # label = 7d, 30d etc.

    os.makedirs(output_dir, exist_ok=True)

    # Historical part from test set
    hist_len = min(len(actual_inverse), len(pred_inverse))
    test_timestamps = df.index[-hist_len:]

    hist_df = pd.DataFrame({
        "Timestamp": test_timestamps,
        "Actual": actual_inverse[-hist_len:, 0],
        "Predicted": pred_inverse[-hist_len:, 0]
    })

    # Future part
    future_preds = pred_inverse[-1].flatten()
    if len(future_preds) != len(future_dates):
        future_preds = future_preds[:len(future_dates)]
        future_dates = future_dates[:len(future_preds)]

    future_df = pd.DataFrame({
        "Timestamp": future_dates,
        "Actual": [None] * len(future_preds),
        "Predicted": future_preds
    })

    full_df = pd.concat([hist_df, future_df])
    full_df.set_index("Timestamp", inplace=True)

    # ✅ Inject OHLC data for 1h interval
    if interval == "1h":
        try:
            ohlc_cols = df[['Open', 'High', 'Low', 'Close']].copy()
            ohlc_cols.index = pd.to_datetime(ohlc_cols.index)
            full_df = full_df.merge(ohlc_cols, how="left", left_index=True, right_index=True)
        except Exception as e:
            print(f"[WARN] Could not inject OHLC columns: {e}")

    # Save CSV
    full_df.to_csv(os.path.join(output_dir, "forecast.csv"))

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(full_df["Predicted"], label="Predicted")
    if full_df["Actual"].notna().sum() > 0:
        plt.plot(full_df["Actual"].dropna(), label="Actual", linestyle="--")
    plt.title(f"{symbol} - {interval} Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast.png"))
    plt.close()

    # Metrics — match latest values only
    valid_len = min(len(actual_inverse), len(pred_inverse))
    mae = mean_absolute_error(actual_inverse[-valid_len:, 0], pred_inverse[-valid_len:, 0])
    mse = mean_squared_error(actual_inverse[-valid_len:, 0], pred_inverse[-valid_len:, 0])
    rmse = np.sqrt(mse)
    # ✅ Directional accuracy (up/down movement match)
    try:
        direction_matches = np.sum(
            np.sign(np.diff(actual_inverse[-valid_len:, 0])) == np.sign(np.diff(pred_inverse[-valid_len:, 0]))
        )
        directional_accuracy = round((direction_matches / (valid_len - 1)) * 100, 2)
    except Exception:
        directional_accuracy = None


    # ✅ Calculate percent change for UI
    try:
        current_price = actual_inverse[-1, 0]
        predicted_price = future_preds[-1]
        percent_change = round(((predicted_price - current_price) / current_price) * 100, 2)
    except Exception:
        percent_change = None

    # ✅ Save all metrics (overwrite always)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "DirectionalAccuracy": directional_accuracy,
            "PercentChange": percent_change,
            "FeaturesUsed": ["Close", "EMA", "RSI", "MACD", "Volume"]
        }, f)




    print(f"[✅] Forecast complete — MAE: {round(mae, 2)} | RMSE: {round(rmse, 2)} | Δ {percent_change}%")
