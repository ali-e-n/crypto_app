import os
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, RepeatVector
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import timedelta
import matplotlib.pyplot as plt

def fetch_data(symbol="BTC-USD", start="2015-01-01", interval="1d", force_refresh=False):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol.replace('-', '_')}_{interval}.csv"

    if not os.path.exists(filename) or force_refresh:
        print(f"[INFO] Downloading fresh data for {symbol} from Yahoo Finance...")
        if interval == "1h":
            df = yf.download(symbol, period="60d", interval=interval, auto_adjust=False)
        else:
            df = yf.download(symbol, start=start, interval=interval, auto_adjust=False)

        if df.empty:
            raise ValueError(f"[ERROR] No data fetched for {symbol}. Check symbol, period, or interval.")

        df.to_csv(filename)
        print(f"[INFO] Saved data to {filename}")
    else:
        print(f"[INFO] Using cached file: {filename}")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df = df[["Close", "Volume", "MACD", "RSI"]].dropna()
    print(f"[INFO] Loaded data with {len(df)} rows. Latest Close: {df['Close'].iloc[-1]}")
    return df


def prepare_seq2seq_data(df, window_size=60, forecast_horizon=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    if len(scaled) < window_size + forecast_horizon:
        raise ValueError(f"Not enough data: {len(scaled)} rows for window {window_size} + horizon {forecast_horizon}")

    x, y = [], []
    for i in range(window_size, len(scaled) - forecast_horizon + 1):
        input_seq = scaled[i - window_size:i]
        target_seq = scaled[i:i + forecast_horizon, 0]

        if input_seq.shape[0] == window_size and target_seq.shape[0] == forecast_horizon:
            x.append(input_seq)
            y.append(target_seq)

    x, y = np.array(x), np.array(y)
    y = y.reshape((y.shape[0], y.shape[1], 1))
    return x, y, scaler


def build_seq2seq_model(input_shape, forecast_horizon):
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(RepeatVector(forecast_horizon))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    try:
        current_price = actual[-1][0]
        predicted_price = predicted[-1][0]
        percent_change = round(((predicted_price - current_price) / current_price) * 100, 2)
    except Exception:
        percent_change = None

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "PercentChange": percent_change
    }


def run_lstm_model_logs(symbol="BTC-USD", interval="ALL", window_size=60, forecast_horizon=30, epochs=10):
    forecast_targets = {
        "1h": {"interval": "1h", "steps": 24},
        "1d": {"interval": "1d", "steps": 7},
        "7d": {"interval": "1d", "steps": 14},
        "14d": {"interval": "1d", "steps": 30},
        "30d": {"interval": "1d", "steps": 90},
        "1y": {"interval": "1d", "steps": 365},
    }

    for label, cfg in forecast_targets.items():
        try:
            yf_interval = cfg["interval"]
            steps = cfg["steps"]

            yield f"data: ▶ Running model for {symbol} - {label} (data: {yf_interval})...\n\n"

            df = fetch_data(symbol=symbol, interval=yf_interval, force_refresh=True)
            yield f"data: ✔️ Data loaded: {len(df)} rows\n\n"

            x, y, scaler = prepare_seq2seq_data(df, window_size=window_size, forecast_horizon=forecast_horizon)
            model = build_seq2seq_model(input_shape=(x.shape[1], x.shape[2]), forecast_horizon=forecast_horizon)

            for epoch in range(1, epochs + 1):
                model.fit(x, y, epochs=1, batch_size=32, verbose=0)
                yield f"data: ✅ Epoch {epoch}/{epochs} completed for {label}\n\n"

            last_window = df[-window_size:].values
            last_scaled = scaler.transform(last_window).reshape(1, window_size, -1)
            forecast_scaled = model.predict(last_scaled)

            padded_forecast = np.concatenate([forecast_scaled[0], np.zeros((forecast_horizon, x.shape[2] - 1))], axis=1)
            future_forecast = scaler.inverse_transform(padded_forecast)[:, 0].flatten()[:steps]

            test_input = x[-1].reshape(1, window_size, -1)
            test_pred_scaled = model.predict(test_input)
            padded_test = np.concatenate([test_pred_scaled[0], np.zeros((forecast_horizon, x.shape[2] - 1))], axis=1)
            historical_pred = scaler.inverse_transform(padded_test)[:, 0].flatten()

            actual = df["Close"].values[-forecast_horizon:]
            test_timestamps = df.index[-forecast_horizon:]

            metrics = evaluate(actual.reshape(-1, 1), historical_pred.reshape(-1, 1))

            dir_key = f"{symbol}_{label}"
            forecast_dir = os.path.join("static/forecasts", dir_key)
            os.makedirs(forecast_dir, exist_ok=True)

            with open(os.path.join(forecast_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

            last_time = df.index[-1]
            time_delta = timedelta(hours=1) if yf_interval == "1h" else timedelta(days=1)
            future_timestamps = [last_time + time_delta * (i + 1) for i in range(len(future_forecast))]

            hist_df = pd.DataFrame({
                "Timestamp": test_timestamps,
                "Actual": actual.flatten(),
                "Predicted": historical_pred
            })

            future_df = pd.DataFrame({
                "Timestamp": future_timestamps,
                "Actual": [None] * len(future_forecast),
                "Predicted": future_forecast
            })

            full_df = pd.concat([hist_df, future_df])
            full_df.set_index("Timestamp", inplace=True)
            full_df.to_csv(os.path.join(forecast_dir, "forecast.csv"))

            plt.figure(figsize=(10, 4))
            plt.plot(full_df["Predicted"], label='Predicted')
            if full_df["Actual"].notna().sum() > 0:
                plt.plot(full_df["Actual"].dropna(), label='Actual')
            plt.legend()
            plt.title(f"Forecast for {symbol} ({label})")
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, "forecast.png"))
            plt.close()

            yield f"data: 📊 {label} — MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, Δ {metrics.get('PercentChange', '-')}%\n\n"
            yield f"data: ✅ Forecast for {label} completed.\n\n"

        except Exception as e:
            yield f"data: ❌ {label} Error: {str(e)}\n\n"
