"""
File: run_pipeline.py
Purpose: Run the complete pipeline from data extraction to predictions
Run this daily to keep everything updated
"""

import subprocess
import schedule
import time
from datetime import datetime

def run_daily_pipeline():
    """Run the complete ETL + ML pipeline"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Daily Pipeline at {datetime.now()}")
    print('='*60)
    
    # Step 1: Extract, Transform, Load new data
    print("\n📥 Step 1: Running ETL Pipeline...")
    result = subprocess.run(['python', 'etl_pipeline.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    # Step 2: Feature Engineering
    print("\n🛠️ Step 2: Creating Features...")
    result = subprocess.run(['python', 'feature_engineering.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    # Step 3: Retrain Model
    print("\n🤖 Step 3: Retraining Model...")
    result = subprocess.run(['python', 'train_model.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    # Step 4: Get Predictions
    print("\n🔮 Step 4: Generating Predictions...")
    result = subprocess.run(['python', 'predict_tomorrow.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline Complete at {datetime.now()}")
    print('='*60)

# Run once immediately
run_daily_pipeline()

# Schedule to run daily at 6 PM
schedule.every().day.at("18:00").do(run_daily_pipeline)

print("\n⏰ Scheduler started. Pipeline will run daily at 6 PM.")
print("Press Ctrl+C to stop")

while True:
    schedule.run_pending()
    time.sleep(60)