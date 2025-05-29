from flask import render_template, request, redirect, url_for, flash, Response, stream_with_context, send_file, jsonify
from flask_login import login_user, logout_user, login_required
from model import lstm_model
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import timedelta
import numpy as np
import yfinance as yf
from model.lstm_model import run_lstm_model_logs
from model.lstm_seq2seq import run_forecast
from queue import Queue
from threading import Thread

BASE_FORECAST_PATH = "static/forecasts"

def setup_routes(app, users):
    @app.route("/")
    def index():
        metrics = None
        forecast_rows = []
        plot_data = {}

        symbol = request.args.get("symbol", "BTC-USD")
        # ✅ Map symbol prefix to icon filename
        symbol_icon_map = {
            "BTC": "bitcoin.png",
            "ETH": "etherium.png",
            "DOGE": "doge.png",
            "XRP": "ripple.png",
            "BNB": "bnb.png",  # Add more as needed
        }
        steps_map = {
            "1h": 24,
            "1d": 7,
            "7d": 5,
            "14d": 4,
            "30d": 5,
            "1y": 7  # or however many months left in year
        }


        symbol_prefix = symbol.split("-")[0]
        icon_filename = symbol_icon_map.get(symbol_prefix, "default.png")
        icon_url = f"/static/icons/{icon_filename}"
        # ✅ Fallback if file doesn't exist
        if not os.path.exists(os.path.join("static/icons", icon_filename)):
            icon_url = "/static/icons/default.png"

        intervals = ["1h", "1d", "7d", "14d", "30d", "1y"]

        forecast_summary = {}
        price_pred = {}

        # ✅ Get live price only once
        live_price = None
        try:
            live_data = yf.download(tickers=symbol, period="1d", interval="1m")
            if not live_data.empty:
                live_price = float(live_data["Close"].iloc[-1])
        except Exception as e:
            print(f"[ERROR] Could not fetch live price for {symbol}: {e}")
            live_price = None

        # 🔁 Loop through intervals to load predictions + changes
        for interval in intervals:
            dir_key = f"{symbol}_{interval}"
            forecast_dir = os.path.join(BASE_FORECAST_PATH, dir_key)

            metrics_path = os.path.join(forecast_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    try:
                        metrics_data = json.load(f)
                        change = metrics_data.get("PercentChange")
                        forecast_summary[interval] = round(change, 2) if change is not None else "-"
                    except:
                        forecast_summary[interval] = "-"
            else:
                forecast_summary[interval] = "-"

            csv_path = os.path.join(forecast_dir, "forecast.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df = df.set_index("Timestamp")
                    df["Predicted"] = df["Predicted"].astype(float)
                    last_pred = df["Predicted"].iloc[-1]
                    price_pred[interval] = round(last_pred, 2)
                except:
                    price_pred[interval] = "-"
            else:
                price_pred[interval] = "-"

        interval_selected = request.args.get("interval", "1d")
        default_dir = os.path.join(BASE_FORECAST_PATH, f"{symbol}_{interval_selected}")
        csv_path = os.path.join(default_dir, "forecast.csv")
        metrics_path = os.path.join(default_dir, "metrics.json")


        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                df_plot = df.copy()
                df_plot["Timestamp"] = pd.to_datetime(df_plot["Timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
                plot_data = {
                    "timestamps": df_plot["Timestamp"].tolist(),
                    "actual": df_plot["Actual"].fillna('').tolist() if "Actual" in df_plot.columns else [None] * len(df_plot),
                    "predicted": df_plot["Predicted"].fillna('').tolist()
                }

                if interval_selected == "1h" and all(col in df_plot.columns for col in ['Open', 'High', 'Low', 'Close']):
                    plot_data["candlestick"] = {
                    "timestamps": df_plot["Timestamp"].tolist(),
                    "open": df_plot["Open"].fillna('').tolist(),
                    "high": df_plot["High"].fillna('').tolist(),
                    "low": df_plot["Low"].fillna('').tolist(),
                    "close": df_plot["Close"].fillna('').tolist()
                }


                if interval_selected == "1h":
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
                else:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime('%Y-%m-%d')

                if interval_selected in ["7d", "14d", "30d", "1y"]:
                    df = df.tail(steps_map[interval_selected]).reset_index(drop=True)
                else:
                    df = df.tail(20).reset_index(drop=True)

                forecast_rows = df.to_dict(orient="records")
        else:
            flash("Forecast data not available yet for the selected currency.")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

        return render_template("user.html",
                        forecast_rows=forecast_rows,
                        metrics=metrics,
                        plot_data=plot_data,
                        symbol=symbol,
                        icon_url=icon_url,  # ✅ add this
                        interval_selected=interval_selected,
                        forecast_summary=forecast_summary,
                        price_pred=price_pred,
                        live_price=live_price)




    @app.route("/api/price/<symbol>")
    def get_live_price(symbol):
        try:
            data = yf.download(tickers=symbol, period="1d", interval="1m")
            if not data.empty:
                last_price = data['Close'].iloc[-1]
                return jsonify({"symbol": symbol, "price": round(last_price, 2)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"error": "Price data unavailable"}), 404

    @app.route("/download/<symbol>/<interval>")
    def download_csv(symbol, interval):
        path = os.path.join(BASE_FORECAST_PATH, f"{symbol}_{interval}", "forecast.csv")
        if os.path.exists(path):
            return send_file(
                path,
                as_attachment=True,
                download_name=f"{symbol}_{interval}_forecast.csv"
            )
        flash("CSV not found.")
        return redirect(url_for("index", symbol=symbol, interval=interval))


    @app.route("/admin", methods=["GET", "POST"])
    @login_required
    def admin():
        symbol = None
        if request.method == "POST":
            symbol = request.form.get("symbol", "BTC-USD")
            intervals = ["1h", "1d", "7d", "14d", "30d", "1y"]

            for interval in intervals:
                dir_key = f"{symbol}_{interval}"
                forecast_dir = os.path.join(BASE_FORECAST_PATH, dir_key)
                os.makedirs(forecast_dir, exist_ok=True)

                csv_path = os.path.join(forecast_dir, "forecast.csv")
                metrics_path = os.path.join(forecast_dir, "metrics.json")
                log_path = os.path.join(forecast_dir, "train.log")

                try:
                    with open(log_path, "w") as log:
                        log.write(f"▶ Running model for {symbol} - {interval}...\n")
                        log.flush()

                        df = lstm_model.fetch_data(symbol=symbol, interval=interval, force_refresh=True)
                        df.index = pd.to_datetime(df.index)
                        log.write(f"✔️ Data loaded: {len(df)} rows\n")
                        log.flush()

                        x_train, y_train, x_test, y_test, scaler = lstm_model.split_data(df)
                        log.write(f"✔️ Data split: x_train={x_train.shape}, x_test={x_test.shape}\n")
                        log.flush()

                        model = lstm_model.build_model((x_train.shape[1], 1))
                        log.write("✔️ Model built\n")
                        log.flush()

                        for epoch in range(1, 11):
                            history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
                            loss = history.history['loss'][0]
                            log.write(f"✅ Epoch {epoch}/10 completed — Loss: {loss:.4f}\n")
                            log.flush()

                        predictions = lstm_model.predict(model, x_test, scaler)
                        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                        metrics = lstm_model.evaluate(actual, predictions)
                        log.write(f"📊 MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}\n")
                        log.flush()

                        future_steps = 24 if interval == "1h" else 7
                        time_delta = timedelta(hours=1) if interval == "1h" else timedelta(days=1)
                        future_input = x_test[-1]
                        future_preds = []

                        for i in range(future_steps):
                            pred = model.predict(future_input.reshape(1, *future_input.shape))[0][0]
                            future_preds.append(pred)
                            future_input = np.append(future_input[1:], [[pred]], axis=0)
                            log.write(f"🔮 Future step {i+1}: {pred:.4f}\n")
                            log.flush()

                        future_scaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                        last_time = df.index[-1]
                        future_timestamps = [last_time + time_delta * (i + 1) for i in range(future_steps)]

                        hist_df = pd.DataFrame({
                            "Timestamp": df.index[-len(actual):],
                            "Actual": actual.flatten(),
                            "Predicted": predictions.flatten()
                        })

                        future_df = pd.DataFrame({
                            "Timestamp": future_timestamps,
                            "Actual": [None] * future_steps,
                            "Predicted": future_scaled.flatten()
                        })

                        full_df = pd.concat([hist_df, future_df])
                        full_df.set_index("Timestamp", inplace=True)
                        full_df.to_csv(csv_path)

                        with open(metrics_path, "w") as f:
                            json.dump(metrics, f)

                        log.write("✅ Forecast generation completed.\n")
                        log.flush()

                except Exception as e:
                    with open(log_path, "a") as log:
                        log.write(f"❌ Error: {str(e)}\n")
                        log.flush()
                    flash(str(e))

        return render_template("admin.html", symbol=symbol)

    @app.route("/logs/<symbol>/<interval>")
    @login_required
    def get_logs(symbol, interval):
        dir_key = f"{symbol}_{interval}"
        log_path = os.path.join(BASE_FORECAST_PATH, dir_key, "train.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                return f.read()
        return ""

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"]
            user = users.get(username)
            if user and password == user.password:
                login_user(user)
                return redirect(url_for("admin"))
            else:
                flash("Invalid credentials")
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    from flask import stream_with_context

    @app.route("/stream_logs")
    def stream_logs():
        symbol = request.args.get("symbol", "BTC-USD")

        def generate_logs():
            log_queue = Queue()

            def log_callback(msg):
                log_queue.put(f"data: {msg}\n\n")

            def run_forecast_thread():
                forecast_targets = {
                    "1h": {"interval": "1h", "steps": 24},
                    "1d": {"interval": "1d", "steps": 7},
                    "7d": {"interval": "1d", "steps": 14},
                    "14d": {"interval": "1d", "steps": 30},
                    "30d": {"interval": "1d", "steps": 90},
                    "1y": {"interval": "1d", "steps": 365},
                }

                for label, cfg in forecast_targets.items():
                    log_callback(f"▶ Running model for {symbol} - {label}...")
                    try:
                        from model.lstm_seq2seq import run_forecast
                        run_forecast(
                            symbol=symbol,
                            interval=cfg["interval"],
                            steps=cfg["steps"],
                            window_size=500,
                            epochs=100,
                            log_callback=log_callback,
                            label=label, 
                        )
                        log_callback(f"✅ Forecast for {label} completed.")
                    except Exception as e:
                        log_callback(f"❌ {label} Error: {str(e)}")

                log_queue.put(None)  # End signal

            Thread(target=run_forecast_thread).start()

            while True:
                msg = log_queue.get()
                if msg is None:
                    break
                yield msg

        return Response(generate_logs(), mimetype="text/event-stream")
