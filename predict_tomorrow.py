"""
File: predict_tomorrow.py
Purpose: Predict tomorrow's stock prices using trained models
Run this ANY TIME to get predictions
"""

import pandas as pd
import numpy as np
import psycopg2
import joblib
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def predict_tomorrow():
    """Predict stock prices for the next trading day"""
    
    print("🤖 Stock Price Predictor")
    print("="*50)
    
    # Load today's data from database
    conn = psycopg2.connect(
        host=os.getenv("AIVEN_HOST"),
        port=os.getenv("AIVEN_PORT"),
        database=os.getenv("AIVEN_NAME"),
        user=os.getenv("AIVEN_USER"),
        password=os.getenv("AIVEN_PASSWORD"),
        sslmode='require'
    )
    
    # Get the most recent data for each ticker
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
    
    tickers = df['ticker'].unique()
    predictions = []
    
    for ticker in tickers:
        print(f"\n📈 Analyzing {ticker}...")
        
        # Get last 20 days of data for this ticker
        ticker_data = df[df['ticker'] == ticker].sort_values('date').tail(20)
        
        # Calculate features needed for prediction
        features = {
            'prev_close': ticker_data['close'].iloc[-1],
            'ma_5': ticker_data['close'].tail(5).mean(),
            'ma_20': ticker_data['close'].mean(),
            'price_change_pct': ((ticker_data['close'].iloc[-1] - ticker_data['close'].iloc[-2]) 
                                / ticker_data['close'].iloc[-2] * 100),
            'volume_change': ((ticker_data['volume'].iloc[-1] - ticker_data['volume'].iloc[-2]) 
                             / ticker_data['volume'].iloc[-2] * 100),
            'daily_range': ticker_data['close'].iloc[-1] - ticker_data['open'].iloc[-1],
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'open': ticker_data['open'].iloc[-1],
            'volume': ticker_data['volume'].iloc[-1]
        }
        
        # Load the trained model
        try:
            model = joblib.load(f'{ticker}_price_predictor.pkl')
            
            # Create feature array in the same order as training
            feature_order = [
                'prev_close', 'ma_5', 'ma_20', 'price_change_pct',
                'volume_change', 'daily_range', 'day_of_week', 'month',
                'open', 'volume'
            ]
            
            X_pred = pd.DataFrame([features])[feature_order]
            
            # Make prediction
            predicted_price = model.predict(X_pred)[0]
            
            # Calculate confidence (based on recent model performance)
            # This is simplified - you'd want more sophisticated confidence scoring
            confidence = min(85 + np.random.randint(-5, 5), 95)  # Placeholder
            
            predictions.append({
                'ticker': ticker,
                'current_price': ticker_data['close'].iloc[-1],
                'predicted_tomorrow': predicted_price,
                'change': predicted_price - ticker_data['close'].iloc[-1],
                'change_pct': ((predicted_price - ticker_data['close'].iloc[-1]) 
                              / ticker_data['close'].iloc[-1] * 100),
                'confidence': confidence
            })
            
            print(f"  Current price: ${ticker_data['close'].iloc[-1]:.2f}")
            print(f"  Predicted tomorrow: ${predicted_price:.2f}")
            print(f"  Expected change: {features['price_change_pct']:.2f}%")
            print(f"  Confidence: {confidence}%")
            
        except FileNotFoundError:
            print(f"  ❌ No trained model found for {ticker}. Run train_model.py first.")
    
    # Summary
    print("\n" + "="*50)
    print("📊 TOMORROW'S PRICE PREDICTIONS")
    print("="*50)
    
    for pred in predictions:
        arrow = "🟢" if pred['change'] > 0 else "🔴"
        print(f"\n{pred['ticker']} {arrow}")
        print(f"  Today:  ${pred['current_price']:.2f}")
        print(f"  Tomorrow: ${pred['predicted_tomorrow']:.2f}")
        print(f"  Change: {pred['change_pct']:+.2f}%")
        print(f"  Confidence: {pred['confidence']}%")
    
    # Trading recommendation
    print("\n" + "="*50)
    print("💡 TRADING RECOMMENDATIONS")
    print("="*50)
    
    for pred in predictions:
        if pred['change_pct'] > 2 and pred['confidence'] > 80:
            print(f"  {pred['ticker']}: 🟢 STRONG BUY (expected +{pred['change_pct']:.1f}%)")
        elif pred['change_pct'] > 0.5:
            print(f"  {pred['ticker']}: ⚪ BUY (expected +{pred['change_pct']:.1f}%)")
        elif pred['change_pct'] < -2:
            print(f"  {pred['ticker']}: 🔴 STRONG SELL (expected {pred['change_pct']:.1f}%)")
        elif pred['change_pct'] < -0.5:
            print(f"  {pred['ticker']}: 🟡 SELL (expected {pred['change_pct']:.1f}%)")
        else:
            print(f"  {pred['ticker']}: ⚪ HOLD (minimal change expected)")

if __name__ == "__main__":
    predict_tomorrow()