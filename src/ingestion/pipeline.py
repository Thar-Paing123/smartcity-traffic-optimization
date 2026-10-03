"""Real-time ingestion and preprocessing pipeline.

Wraps per-intersection traffic time series into sliding windows for sequence
forecasting and exposes them as tf.data.Dataset pipelines, so the same
windowing logic can be swapped between simulated/historical CSV data today
and a live streaming source (cameras, GPS, loop detectors) later without
touching the model code.
"""

import json
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

FEATURE_COLUMNS = ["volume", "avg_speed_kph", "occupancy"]


def load_historical(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["timestamp"])


def intersection_ids(df: pd.DataFrame) -> list:
    """Intersection ids in the same order pivot()/_pivot_wide() produces columns in."""
    return sorted(df["intersection_id"].unique())


def latest_window(df: pd.DataFrame, window: int) -> np.ndarray:
    """The most recent `window` time steps, shaped (window, intersections, features)."""
    data = _impute(_pivot_wide(df))
    return data[-window:]


def save_stats(stats: dict, path: str) -> None:
    """Persist normalization stats so inference code can reuse the exact training-time scale."""
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_stats(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    raw["mean"] = np.array(raw["mean"])
    raw["std"] = np.array(raw["std"])
    return raw


def _pivot_wide(df: pd.DataFrame) -> np.ndarray:
    """Pivot long-format data into shape (time, intersections, features)."""
    arrays = [df.pivot(index="timestamp", columns="intersection_id", values=col).to_numpy() for col in FEATURE_COLUMNS]
    return np.stack(arrays, axis=-1)  # (time, intersections, features)


def _impute(data: np.ndarray) -> np.ndarray:
    """Fill gaps with the per-feature mean rather than 0.

    Some real-world sources (see ingestion/real_sources/) only measure a
    subset of features (e.g. volume but not speed/occupancy) and some have
    gaps between recording campaigns; zero-filling those would read as
    "road closed" or "sensor broken" when it really just means "not
    measured", so the mean is a less misleading placeholder.
    """
    data = data.astype(float)
    feature_means = np.nan_to_num(np.nanmean(data, axis=(0, 1), keepdims=True), nan=0.0)
    missing = np.isnan(data)
    data[missing] = np.broadcast_to(feature_means, data.shape)[missing]
    return data


def make_datasets(
    df: pd.DataFrame,
    window: int = 12,
    horizon: int = 3,
    batch_size: int = 64,
    val_split: float = 0.15,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """Build windowed train/val tf.data pipelines from long-format traffic data.

    `window` past steps are used to predict `horizon` steps ahead, per
    intersection, across all feature columns.
    """
    data = _impute(_pivot_wide(df))

    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (data - mean) / std

    t, intersections, features = normalized.shape
    flat = normalized.reshape(t, intersections * features)

    n_windows = t - window - horizon + 1
    split = int(n_windows * (1 - val_split))

    def make_ds(start: int, end: int) -> tf.data.Dataset:
        xs = np.stack([flat[i : i + window] for i in range(start, end)])
        ys = np.stack([flat[i + window : i + window + horizon] for i in range(start, end)])
        return tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(0, split)
    val_ds = make_ds(split, n_windows)

    stats = {
        "mean": mean,
        "std": std,
        "n_intersections": intersections,
        "n_features": features,
        "window": window,
        "horizon": horizon,
    }
    return train_ds, val_ds, stats
