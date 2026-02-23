# Stock Predictor

End-to-end stock prediction project with:
- ETL from Yahoo Finance
- PostgreSQL storage
- Feature engineering
- Per-ticker ML model training
- Next-day prediction script
- Streamlit dashboard

## Project Structure

- `etl_pipeline.py`: pulls market data and writes to `stock_data`
- `feature_engineering.py`: builds features into `stock_features`
- `train_model.py`: trains and saves per-ticker models
- `predict_tomorrow.py`: loads models and predicts next-day prices
- `run_Pipeline.py`: runs the full pipeline (single run or scheduled)
- `dashboard.py`: interactive Streamlit dashboard
- `.env.example`: required environment variables template

## Requirements

Recommended Python: 3.10+

Install dependencies:

```bash
pip install streamlit pandas plotly psycopg2-binary joblib python-dotenv numpy scikit-learn matplotlib yfinance schedule
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Fill in your DB credentials:

```env
AIVEN_HOST=your-host
AIVEN_PORT=your-port
AIVEN_NAME=your-database
AIVEN_USER=your-username
AIVEN_PASSWORD=your-password
```

## Run Pipeline

Run full pipeline once:

```bash
python run_Pipeline.py --once
```

Run and keep scheduler active (daily at 18:00):

```bash
python run_Pipeline.py
```

## Run Dashboard

```bash
streamlit run dashboard.py
```

## Notes

- ETL date range is dynamic: last 365 days up to current date.
- `run_Pipeline.py` fails fast and reports which step failed.
- If training is skipped for a ticker, it usually means insufficient clean rows.

## Security

- Do not commit `.env`
- Secret-safety guard is included:
  - `scripts/precommit_secret_check.py`
  - `.githooks/pre-commit`

To ensure hook is active locally:

```bash
git config core.hooksPath .githooks
```
