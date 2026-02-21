#training a mchine learning model to predict stock prices

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import psycopg2
import joblib
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

load_dotenv()

# Load features from database
conn = psycopg2.connect(
    host=os.getenv("AIVEN_HOST"),
    port=os.getenv("AIVEN_PORT"),
    database=os.getenv("AIVEN_NAME"),
    user=os.getenv("AIVEN_USER"),
    password=os.getenv("AIVEN_PASSWORD"),
    sslmode='require'
   )

query = "SELECT * FROM stock_features ORDER BY ticker, date"
df = pd.read_sql(query, conn)
print(f"Loaded{len(df)} rows with features")

# We'll build a separate model for each ticker
tickers = df['ticker'].unique()
models = {}

for ticker in tickers:
   print(f"\n{'='*50}")
   print(f"Training model for {ticker}")

   # Get data for this ticker
   ticker_data = df[df['ticker'] == ticker].copy()
   
   # Select features (X) and target (y)
   feature_columns = [
      'prev_close', 'ma_5', 'ma_20', 'price_change_pct',
        'volume_change', 'daily_range', 'day_of_week', 'month',
        'open', 'volume'
    ]
   
   X = ticker_data[feature_columns]
   y = ticker_data['next_day_close']

   # split into training (80%) and testing (20%)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42, shuffle = False
                                                       )
   
   print(f"Training on {len(X_train)} days, Testing on { len(X_test)} days")

   # Train Random Forest model
   model = RandomForestRegressor(
      n_estimators=100,
      max_depth= 10,
      random_state=42
   )
   model.fit(X_train, y_train)

   # Make predictions
   y_pred = model.predict(X_test)

   #Evaluate model
   mae = mean_absolute_error(y_test, y_pred)
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   r2 = r2_score(y_test, y_pred)

   print(f"\n📊 Model Performance for {ticker}:")
   print(f"  Mean Absolute Error: ${mae:.2f}")
   print(f"  Root Mean Square Error: ${rmse:.2f}")
   print(f"  R² Score: {r2:.3f}")

   # Feature importance
   importance = pd.DataFrame({
      'feature' : feature_columns,
      'importance' : model.feature_importances_
   }).sort_values('importance', ascending=False)

   print("\n🔍 Top 5 Most Important Features:")
   print(importance.head())
    
    # Save model
   joblib.dump(model, f'{ticker}_price_predictor.pkl')
   print(f"✅ Model saved as {ticker}_price_predictor.pkl")
    
    # Store model for later use
   models[ticker] = model
    
    # Plot predictions vs actual
   plt.figure(figsize=(12, 6))
   plt.plot(y_test.values[-30:], label='Actual', marker='o')
   plt.plot(y_pred[-30:], label='Predicted', marker='x')
   plt.title(f'{ticker} - Actual vs Predicted Prices (Last 30 test days)')
   plt.xlabel('Days')
   plt.ylabel('Stock Price ($)')
   plt.legend()
   plt.grid(True)
   plt.savefig(f'{ticker}_predictions.png')
   plt.show()

conn.close()
print("\n All models trained and saved")