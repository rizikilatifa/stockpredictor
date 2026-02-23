##creating feature that help predict stock prices

import pandas as pd
import psycopg2
from datetime import datetime, timedelta

import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("AIVEN_HOST")
port = os.getenv("AIVEN_PORT")
database = os.getenv("AIVEN_NAME")
user = os.getenv("AIVEN_USER")
password = os.getenv("AIVEN_PASSWORD")


#Database connection
conn = psycopg2.connect(
   host = os.getenv("AIVEN_HOST"),
   port = os.getenv("AIVEN_PORT"),
   database = os.getenv("AIVEN_NAME"),
   user = os.getenv("AIVEN_USER"),
   password = os.getenv("AIVEN_PASSWORD"),
   sslmode = 'require'
   )

# read data from database
query = "SELECT * FROM stock_data ORDER BY ticker, date"
df = pd.read_sql(query, conn)
print(f"Read{len(df)} rows from database")

# create features for each ticker separately
tickers= df['ticker'].unique()
all_features = []

for ticker in tickers:
   ticker_data = df[df['ticker'] == ticker].copy()
   ticker_data = ticker_data.sort_values('date')

   # FEATURE 1 : Previous day's close
   ticker_data['prev_close'] = ticker_data['close'].shift(1)

   # FEATURE 2: 5-day moving average
   ticker_data['ma_5'] = ticker_data['close'].rolling(window =5).mean()

   # FEATURE 3: 20-day moving average
   ticker_data['ma_20'] = ticker_data['close'].rolling(window = 20).mean()

   # FEATURE 4: Price change percentage
   ticker_data['price_change_pct'] = ticker_data['close'].pct_change() *100

   # FEATURE 5 volume change
   ticker_data['volume_change'] = ticker_data['volume'].pct_change() *100

   # FEATURE 6: High-low range
   ticker_data['daily_range'] = ticker_data['close'] - ticker_data['open']

   # FEATURE 7: Day of weel( 0= Monday, 4= Friday)
   ticker_data['day_of_week'] = pd.to_datetime(ticker_data['date']).dt.dayofweek

   # FEATURE 8: Month
   ticker_data['month'] = pd.to_datetime(ticker_data['date']).dt.month

   # TARGET : Next day's price (what we want to predict)
   ticker_data['next_day_close'] = ticker_data['close'].shift(-1)

   all_features.append(ticker_data)

#combine all tickers
feature_df = pd.concat(all_features, ignore_index=True)

# Remove rows with NaN(From shifting/rolling)
feature_df = feature_df.dropna()

print(f"Created {len(feature_df.columns) - 5} features for {len(feature_df)} rows")
print("\nFeatures created:")
for col in feature_df.columns:
    print(f"  - {col}")
   
# save features to a new table in postgreSQL
from sqlalchemy import create_engine

Database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require"

engine = create_engine(Database_url)

feature_df.to_sql("stock_features", engine, if_exists ='replace', index = False)

print("\n Features saved to stock_features table")

conn.close()