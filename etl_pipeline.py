import yfinance as yf
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL connection info
DB_HOST = os.getenv("AIVEN_HOST")
DB_PORT = os.getenv("AIVEN_PORT")
DB_NAME = os.getenv("AIVEN_NAME")
DB_USER = os.getenv("AIVEN_USER")
DB_PASSWORD = os.getenv("AIVEN_PASSWORD")

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    sslmode='require'
)
cur = conn.cursor()

# Create table if it doesn't exist (kept for future runs)
cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        open NUMERIC(10, 2),
        close NUMERIC(10, 2),
        volume BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date)
    );
""")
conn.commit()

# Stock info
TICKERS = ["AAPL", "MSFT", "GOOGL"]

for ticker in TICKERS:
    df = yf.download(ticker, start="2026-01-01", end="2026-02-20")
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    df = df[['Ticker', 'Date', 'Open', 'Close', 'Volume']]
    df.columns = ['ticker', 'date', 'open', 'close', 'volume']
    
    # Insert into PostgreSQL
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO stock_data (ticker, date, open, close, volume)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO NOTHING
        """, (row['ticker'], row['date'], row['open'], row['close'], row['volume']))

conn.commit()
cur.close()
conn.close()

print("Data inserted successfully!")