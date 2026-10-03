"""Shared helpers for fetching and normalizing real-world public traffic datasets.

Every fetcher in this package returns a DataFrame in the same long-format
schema `ingestion/pipeline.py` expects: timestamp, intersection_id, volume,
avg_speed_kph, occupancy. Fields a given source doesn't measure are left as
NaN rather than invented, and documented as such in that source's module.
"""

import json
import os

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "intersection_id", "volume", "avg_speed_kph", "occupancy"]


def mph_to_kph(mph: pd.Series) -> pd.Series:
    return mph * 1.60934


def density_proxy_occupancy(volume: pd.Series, speed_kph: pd.Series) -> pd.Series:
    """Estimate relative occupancy (0-1) from flow/speed when a source doesn't measure it directly.

    Traffic flow theory treats density as roughly proportional to flow/speed.
    This is min-max normalized across the series, so it reads as a relative
    congestion indicator for that location, not a calibrated occupancy %.
    """
    density = (volume / speed_kph.replace(0, np.nan)).fillna(0)
    lo, hi = density.min(), density.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=density.index)
    return (density - lo) / (hi - lo)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the pipeline's expected column set, order, and sort."""
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["intersection_id", "timestamp"]).reset_index(drop=True)


def save_dataset(df: pd.DataFrame, root_dir: str, country: str, city: str) -> str:
    """Write a fetched dataset to data/real/<country>/<city>/traffic.csv."""
    out_dir = os.path.join(root_dir, country.replace(" ", "_"), city.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "traffic.csv")
    df.to_csv(out_path, index=False)
    return out_path


def save_locations(locations: dict, root_dir: str, country: str, city: str) -> str:
    """Write {intersection_id: (lat, lon)} to data/real/<country>/<city>/locations.json for map display."""
    out_dir = os.path.join(root_dir, country.replace(" ", "_"), city.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "locations.json")
    with open(out_path, "w") as f:
        json.dump(locations, f)
    return out_path
