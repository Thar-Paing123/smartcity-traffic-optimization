"""Synthetic traffic data generator.

Stands in for real traffic-camera, GPS, and historical-record feeds until
live ingestion endpoints are wired up. Produces per-intersection time series
of vehicle volume, average speed, and occupancy with daily seasonality and
randomly injected congestion incidents, in the same long-format shape a real
feed would need to arrive in for the rest of the pipeline to consume it.
"""

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60


def generate_locations(ids: list, center: tuple = (39.5, -98.35), spread: float = 0.06) -> dict:
    """Fabricated lat/lon grid around `center`, purely for map display of the demo dataset.

    Keyed by the exact `ids` values passed in (not re-derived names), so
    lookups against the dataset's actual intersection_id values always hit.
    Default center is the geographic center of the contiguous US — a neutral
    placeholder, not a claim about where this "city" is. Callers should label
    it clearly as simulated.
    """
    rng = np.random.default_rng(7)
    side = int(np.ceil(np.sqrt(len(ids))))
    offsets = [(r - side / 2, c - side / 2) for r in range(side) for c in range(side)][: len(ids)]
    return {
        intersection_id: (
            center[0] + dr * spread + rng.normal(0, spread * 0.1),
            center[1] + dc * spread + rng.normal(0, spread * 0.1),
        )
        for intersection_id, (dr, dc) in zip(ids, offsets)
    }


def _daily_pattern(timestamps_s: np.ndarray) -> np.ndarray:
    time_of_day = (timestamps_s % SECONDS_PER_DAY) / SECONDS_PER_DAY
    morning_peak = np.exp(-((time_of_day - 0.33) ** 2) / (2 * 0.03**2))
    evening_peak = np.exp(-((time_of_day - 0.75) ** 2) / (2 * 0.04**2))
    return morning_peak + evening_peak


def generate_intersection_data(
    n_intersections: int = 10,
    days: int = 30,
    interval_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a long-format DataFrame of simulated traffic data.

    Columns: timestamp, intersection_id, volume, avg_speed_kph, occupancy
    """
    rng = np.random.default_rng(seed)
    n_steps = int(days * SECONDS_PER_DAY / (interval_minutes * 60))
    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq=f"{interval_minutes}min")
    timestamps_s = timestamps.values.astype("int64") // 10**9

    congestion = _daily_pattern(timestamps_s)
    decay_len = max(1, int(30 / interval_minutes))

    frames = []
    for intersection_id in range(n_intersections):
        base_volume = rng.uniform(20, 60)

        incident_mask = rng.random(n_steps) < 0.002
        incident_decay = np.zeros(n_steps)
        for idx in np.where(incident_mask)[0]:
            end = min(n_steps, idx + decay_len)
            incident_decay[idx:end] += np.linspace(1.0, 0.0, end - idx)

        noise = rng.normal(0, 0.05, n_steps)
        congestion_level = np.clip(congestion + incident_decay + noise, 0, 1.5)

        volume = base_volume * (1 + 2.5 * congestion_level) + rng.normal(0, 2, n_steps)
        avg_speed_kph = 60 * np.exp(-1.8 * congestion_level) + rng.normal(0, 1.5, n_steps)
        occupancy = np.clip(congestion_level / 1.5 + rng.normal(0, 0.03, n_steps), 0, 1)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "intersection_id": intersection_id,
                    "volume": np.clip(volume, 0, None),
                    "avg_speed_kph": np.clip(avg_speed_kph, 0, None),
                    "occupancy": occupancy,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import os

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "historical_traffic.csv",
    )
    df = generate_intersection_data()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
