"""
File: run_pipeline.py
Purpose: Run the complete pipeline from data extraction to predictions
Run this daily to keep everything updated
"""

import argparse
import subprocess
import schedule
import time
from datetime import datetime


def run_step(step_name, script_name):
    """Run one pipeline step and raise if it fails."""
    print(f"\n{step_name}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def run_daily_pipeline():
    """Run the complete ETL + ML pipeline."""
    print(f"\n{'=' * 60}")
    print(f"Starting Daily Pipeline at {datetime.now()}")
    print("=" * 60)

    run_step("Step 1: Running ETL Pipeline...", "etl_pipeline.py")
    run_step("Step 2: Creating Features...", "feature_engineering.py")
    run_step("Step 3: Retraining Model...", "train_model.py")
    run_step("Step 4: Generating Predictions...", "predict_tomorrow.py")

    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete at {datetime.now()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run stock ETL/ML pipeline")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    args = parser.parse_args()

    run_daily_pipeline()

    if args.once:
        return

    schedule.every().day.at("18:00").do(run_daily_pipeline)
    print("\nScheduler started. Pipeline will run daily at 6 PM.")
    print("Press Ctrl+C to stop")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
