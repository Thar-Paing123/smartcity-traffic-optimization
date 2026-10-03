"""Trains the traffic forecasting model and prepares it for serving.

Supports multi-GPU scale-out via tf.distribute.MirroredStrategy: the same
training loop runs unchanged on a single CPU/GPU machine today or across
multiple GPUs as more compute is added, with no code change at that point.
"""

import argparse
import os

import tensorflow as tf

from ingestion.pipeline import load_historical, make_datasets, save_stats
from ingestion.simulate import generate_intersection_data
from models.traffic_model import build_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT_DIR, "data", "historical_traffic.csv")
DEFAULT_MODEL_DIR = os.path.join(ROOT_DIR, "artifacts", "model")
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")


def get_data(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"No historical data at {csv_path}, generating synthetic dataset...")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = generate_intersection_data()
        df.to_csv(csv_path, index=False)
    else:
        df = load_historical(csv_path)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    df = get_data(args.csv)
    train_ds, val_ds, stats = make_datasets(df, window=args.window, horizon=args.horizon, batch_size=args.batch_size)
    stats["csv_path"] = os.path.abspath(args.csv)

    strategy = tf.distribute.MirroredStrategy()
    print(f"Training with {strategy.num_replicas_in_sync} device(s)")

    with strategy.scope():
        model = build_model(
            window=stats["window"],
            horizon=stats["horizon"],
            n_intersections=stats["n_intersections"],
            n_features=stats["n_features"],
        )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "checkpoint.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    keras_model_path = os.path.join(args.model_dir, "model.keras")
    model.save(keras_model_path)
    print(f"Saved model (for retraining/evaluation) to {keras_model_path}")

    stats_path = os.path.join(args.model_dir, "stats.json")
    save_stats(stats, stats_path)
    print(f"Saved normalization stats to {stats_path}")

    saved_model_path = os.path.join(args.model_dir, "saved_model")
    model.export(saved_model_path)
    print(f"Exported serving-ready SavedModel to {saved_model_path}")


if __name__ == "__main__":
    main()
