import os
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import timedelta
import matplotlib.pyplot as plt


def fetch_data(symbol="BTC-USD", start="2015-01-01", interval="1d", force_refresh=False):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol.replace('-', '_')}_{interval}.csv"

    # Always re-fetch if admin triggers it
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
        df = pd.read_csv(filename, index_col=0)

    if "Close" not in df.columns:
        raise ValueError("[ERROR] 'Close' column not found in the dataset.")

    df = df[["Close"]].dropna()
    print(f"[INFO] Loaded data with {len(df)} rows. Latest Close: {df['Close'].iloc[-1]}")
    return df





def prepare_data(df, window_size=60):
    # Extract just the Close prices (should be a Series, not whole DataFrame)
    data = df["Close"].values.reshape(-1, 1)  # <-- this line must exist
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    x, y = [], []
    for i in range(window_size, len(scaled)):
        x.append(scaled[i - window_size:i])
        y.append(scaled[i])

    return np.array(x), np.array(y), scaler



def split_data(df, window_size=60, train_split=0.8):
    x, y, scaler = prepare_data(df, window_size)
    split = int(len(x) * train_split)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    return x_train, y_train, x_test, y_test, scaler


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



def train_model(model, x_train, y_train, epochs=20, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def predict(model, x_test, scaler):
    predictions = model.predict(x_test)
    return scaler.inverse_transform(predictions)


def evaluate(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

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



def run_lstm_model_logs(symbol="BTC-USD", interval="ALL", window_size=60, epochs=10):
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

            x_train, y_train, x_test, y_test, scaler = split_data(df, window_size=window_size)
            model = build_model((x_train.shape[1], 1))

            for epoch in range(1, epochs + 1):
                model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
                yield f"data: ✅ Epoch {epoch}/{epochs} completed for {label}\n\n"

            predictions = predict(model, x_test, scaler)
            actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            metrics = evaluate(actual, predictions)

            # Save forecast files
            dir_key = f"{symbol}_{label}"
            forecast_dir = os.path.join("static/forecasts", dir_key)
            os.makedirs(forecast_dir, exist_ok=True)

            # Save metrics
            with open(os.path.join(forecast_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

            # Generate future predictions
            future_input = x_test[-1]
            future_preds = []
            for _ in range(steps):
                pred = model.predict(future_input.reshape(1, *future_input.shape))[0][0]
                future_preds.append(pred)
                future_input = np.append(future_input[1:], [[pred]], axis=0)
            future_scaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

            last_time = df.index[-1]
            time_delta = timedelta(hours=1) if yf_interval == "1h" else timedelta(days=1)
            future_timestamps = [last_time + time_delta * (i + 1) for i in range(steps)]

            test_timestamps = df.index[-len(actual):]
            hist_df = pd.DataFrame({
                "Timestamp": test_timestamps,
                "Actual": actual.flatten(),
                "Predicted": predictions.flatten()
            })

            future_df = pd.DataFrame({
                "Timestamp": future_timestamps,
                "Actual": [None] * steps,
                "Predicted": future_scaled.flatten()
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
