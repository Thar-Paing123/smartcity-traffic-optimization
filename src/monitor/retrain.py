"""Evaluates the current model against fresh data and retrains it on drift.

Meant to be run on a schedule (cron, Airflow, etc.) so the model stays
responsive to changing traffic patterns without someone manually noticing
degraded predictions and re-triggering training by hand.
"""

import argparse
import os
import subprocess
import sys

import tensorflow as tf

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ingestion.pipeline import load_historical, make_datasets  # noqa: E402
from ingestion.simulate import generate_intersection_data  # noqa: E402

DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "model", "model.keras")
DEFAULT_CSV = os.path.join(ROOT_DIR, "data", "historical_traffic.csv")
DEFAULT_MAE_THRESHOLD = 0.35


def evaluate(model_path: str, csv_path: str) -> float:
    model = tf.keras.models.load_model(model_path)
    df = load_historical(csv_path) if os.path.exists(csv_path) else generate_intersection_data()
    _, val_ds, _ = make_datasets(df)
    _, mae = model.evaluate(val_ds, verbose=0)
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--mae-threshold", type=float, default=DEFAULT_MAE_THRESHOLD)
    args = parser.parse_args()

    mae = evaluate(args.model_path, args.csv)
    print(f"Current validation MAE: {mae:.4f} (threshold: {args.mae_threshold})")

    if mae > args.mae_threshold:
        print("Model performance has drifted past threshold, retraining...")
        subprocess.run([sys.executable, os.path.join(SRC_DIR, "train.py"), "--csv", args.csv], check=True, cwd=SRC_DIR)
    else:
        print("Model within acceptable performance, no retraining needed.")


if __name__ == "__main__":
    main()
