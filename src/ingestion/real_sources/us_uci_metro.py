"""Minneapolis-Saint Paul interstate traffic volume (UCI Machine Learning Repository, free, no API key).

Source: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
Hourly westbound traffic volume on I-94 at a single location, 2012-2018.
Volume is directly measured; this source doesn't report speed or occupancy,
so those fields are left as NaN (see ingestion/pipeline._impute for how
that's handled downstream).
"""

import gzip
import io
import zipfile

import pandas as pd
import requests

from .common import finalize

COUNTRY = "United States"
CITY = "Minneapolis-Saint Paul"
DATASET_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"

# The dataset doesn't publish exact sensor geolocation; this is an approximate
# point on I-94 near downtown Minneapolis, for map display only.
LOCATIONS = {"I-94 Westbound": (44.9740, -93.2277)}


def fetch() -> pd.DataFrame:
    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner_name = next(name for name in zf.namelist() if name.endswith(".csv.gz"))
        raw_bytes = gzip.decompress(zf.read(inner_name))

    raw = pd.read_csv(io.BytesIO(raw_bytes))

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["date_time"]),
            "intersection_id": "I-94 Westbound",
            "volume": pd.to_numeric(raw["traffic_volume"], errors="coerce"),
            "avg_speed_kph": float("nan"),
            "occupancy": float("nan"),
        }
    )
    # The source has a known quirk of duplicated hourly rows; keep one per timestamp.
    df = df.drop_duplicates(subset=["timestamp", "intersection_id"])
    return finalize(df)
