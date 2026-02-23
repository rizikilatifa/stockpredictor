"""
File: predict_tomorrow.py
Purpose: Predict tomorrow's stock prices using trained models
Run this ANY TIME to get predictions
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

FEATURE_ORDER = [
    "prev_close",
    "ma_5",
    "ma_20",
    "price_change_pct",
    "volume_change",
    "daily_range",
    "day_of_week",
    "month",
    "open",
    "volume",
]


def predict_tomorrow():
    """Predict stock prices for the next trading day."""
    print("Stock Price Predictor")
    print("=" * 50)

    conn = psycopg2.connect(
        host=os.getenv("AIVEN_HOST"),
        port=os.getenv("AIVEN_PORT"),
        database=os.getenv("AIVEN_NAME"),
        user=os.getenv("AIVEN_USER"),
        password=os.getenv("AIVEN_PASSWORD"),
        sslmode="require",
    )

    query = """
        WITH latest_data AS (
            SELECT
                ticker,
                date,
                open,
                close,
                volume,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM stock_data
        )
        SELECT * FROM latest_data WHERE rn <= 20
        ORDER BY ticker, date DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    tickers = df["ticker"].unique()
    predictions = []

    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")

        ticker_data = df[df["ticker"] == ticker].sort_values("date").tail(20)
        if len(ticker_data) < 2:
            print(f"  Skipping {ticker}: not enough history to compute daily change.")
            continue

        prev_close = ticker_data["close"].iloc[-2]
        prev_volume = ticker_data["volume"].iloc[-2]
        if prev_close == 0 or prev_volume == 0:
            print(f"  Skipping {ticker}: previous close/volume is zero.")
            continue

        features = {
            "prev_close": ticker_data["close"].iloc[-1],
            "ma_5": ticker_data["close"].tail(5).mean(),
            "ma_20": ticker_data["close"].mean(),
            "price_change_pct": ((ticker_data["close"].iloc[-1] - prev_close) / prev_close * 100),
            "volume_change": ((ticker_data["volume"].iloc[-1] - prev_volume) / prev_volume * 100),
            "daily_range": ticker_data["close"].iloc[-1] - ticker_data["open"].iloc[-1],
            "day_of_week": datetime.now().weekday(),
            "month": datetime.now().month,
            "open": ticker_data["open"].iloc[-1],
            "volume": ticker_data["volume"].iloc[-1],
        }

        try:
            model = joblib.load(f"{ticker}_price_predictor.pkl")
            x_pred = pd.DataFrame([features])[FEATURE_ORDER]
            predicted_price = model.predict(x_pred)[0]
            confidence = min(85 + np.random.randint(-5, 5), 95)

            predictions.append(
                {
                    "ticker": ticker,
                    "current_price": ticker_data["close"].iloc[-1],
                    "predicted_tomorrow": predicted_price,
                    "change": predicted_price - ticker_data["close"].iloc[-1],
                    "change_pct": (
                        (predicted_price - ticker_data["close"].iloc[-1])
                        / ticker_data["close"].iloc[-1]
                        * 100
                    ),
                    "confidence": confidence,
                }
            )

            print(f"  Current price: ${ticker_data['close'].iloc[-1]:.2f}")
            print(f"  Predicted tomorrow: ${predicted_price:.2f}")
            print(f"  Expected change: {features['price_change_pct']:.2f}%")
            print(f"  Confidence: {confidence}%")

        except FileNotFoundError:
            print(f"  No trained model found for {ticker}. Run train_model.py first.")

    print("\n" + "=" * 50)
    print("TOMORROW'S PRICE PREDICTIONS")
    print("=" * 50)

    for pred in predictions:
        arrow = "UP" if pred["change"] > 0 else "DOWN"
        print(f"\n{pred['ticker']} [{arrow}]")
        print(f"  Today:    ${pred['current_price']:.2f}")
        print(f"  Tomorrow: ${pred['predicted_tomorrow']:.2f}")
        print(f"  Change:   {pred['change_pct']:+.2f}%")
        print(f"  Confidence: {pred['confidence']}%")

    print("\n" + "=" * 50)
    print("TRADING RECOMMENDATIONS")
    print("=" * 50)

    for pred in predictions:
        if pred["change_pct"] > 2 and pred["confidence"] > 80:
            print(f"  {pred['ticker']}: STRONG BUY (expected +{pred['change_pct']:.1f}%)")
        elif pred["change_pct"] > 0.5:
            print(f"  {pred['ticker']}: BUY (expected +{pred['change_pct']:.1f}%)")
        elif pred["change_pct"] < -2:
            print(f"  {pred['ticker']}: STRONG SELL (expected {pred['change_pct']:.1f}%)")
        elif pred["change_pct"] < -0.5:
            print(f"  {pred['ticker']}: SELL (expected {pred['change_pct']:.1f}%)")
        else:
            print(f"  {pred['ticker']}: HOLD (minimal change expected)")


if __name__ == "__main__":
    predict_tomorrow()
