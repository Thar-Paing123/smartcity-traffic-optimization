"""New York City traffic volume counts (NYC Open Data, Socrata API, free, no API key).

Source: https://data.cityofnewyork.us/resource/7ym2-wayt.json ("Traffic
Volume Counts"). NYC DOT's counts are periodic campaigns rather than
continuous monitoring; this fetches one ~3-month campaign (request id
20922, segment 83624 on 111th Street, Queens) that recorded both travel
directions concurrently, giving two time-aligned real series. Volume is
directly measured; this source doesn't report speed or occupancy, so those
fields are left as NaN.
"""

import pandas as pd
import requests

from .common import finalize

COUNTRY = "United States"
CITY = "New York City"
BASE_URL = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
REQUEST_ID = 20922
SEGMENT_ID = 83624
PAGE_SIZE = 5000

# Segment 83624 is 111th St between 45th & 46th Ave, Corona, Queens (looked up
# manually; the dataset's own "wktgeom" field is in NY State Plane feet, not
# lat/lon, so this is an approximate point for map display only).
LOCATIONS = {"111 Street (NB)": (40.7527, -73.8603), "111 Street (SB)": (40.7527, -73.8603)}


def fetch() -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        resp = requests.get(
            BASE_URL,
            params={
                "requestid": REQUEST_ID,
                "segmentid": SEGMENT_ID,
                "$limit": PAGE_SIZE,
                "$offset": offset,
                "$order": "yr,m,d,hh,mm",
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not rows:
        raise RuntimeError(f"No NYC Open Data rows returned for request {REQUEST_ID} / segment {SEGMENT_ID}")

    raw = pd.DataFrame(rows)
    timestamp = pd.to_datetime(
        pd.DataFrame(
            {
                "year": raw["yr"].astype(int),
                "month": raw["m"].astype(int),
                "day": raw["d"].astype(int),
                "hour": raw["hh"].astype(int),
                "minute": raw["mm"].astype(int),
            }
        )
    )
    street = raw["street"].iloc[0].title()

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "intersection_id": street + " (" + raw["direction"] + ")",
            "volume": pd.to_numeric(raw["vol"], errors="coerce"),
            "avg_speed_kph": float("nan"),
            "occupancy": float("nan"),
        }
    )
    return finalize(df)
