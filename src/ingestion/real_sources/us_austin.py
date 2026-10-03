"""Austin, TX traffic data via the city's Radar Traffic Counts dataset (Socrata, free, no API key).

Source: https://data.austintexas.gov/resource/i626-g7ub.json ("Radar Traffic
Counts"). Unlike the other US sources in this package, Austin's radar
detectors report volume, occupancy, AND speed natively per named
intersection/direction — no derived fields needed here.

Coordinates for each intersection are not part of the dataset and are
manually identified (approximate) for map display.
"""

import pandas as pd
import requests

from .common import finalize

COUNTRY = "United States"
CITY = "Austin"
BASE_URL = "https://data.austintexas.gov/resource/i626-g7ub.json"
YEAR = 2018
PAGE_SIZE = 5000

# (intersection name as stored in the source, detector/direction) -> label
DETECTORS = [
    ("LAMARSHOALCREEK", "NB_in", "Lamar & Shoal Creek (NB)"),
    ("LAMARSHOALCREEK", "SB_in", "Lamar & Shoal Creek (SB)"),
    ("CONGRESSBARTON SPRINGS", "NB_in", "Congress & Barton Springs (NB)"),
    ("CONGRESSBARTON SPRINGS", "SB_in", "Congress & Barton Springs (SB)"),
]

LOCATIONS = {
    "Lamar & Shoal Creek (NB)": (30.2975, -97.7636),
    "Lamar & Shoal Creek (SB)": (30.2975, -97.7636),
    "Congress & Barton Springs (NB)": (30.2610, -97.7495),
    "Congress & Barton Springs (SB)": (30.2610, -97.7495),
}


def _fetch_detector(intname: str, detname: str, label: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        resp = requests.get(
            BASE_URL,
            params={
                "$where": f"year={YEAR} AND intname='{intname}' AND detname='{detname}'",
                "$select": "curdatetime,volume,occupancy,speed",
                "$order": "curdatetime",
                "$limit": PAGE_SIZE,
                "$offset": offset,
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
        raise RuntimeError(f"No Austin data returned for {intname}/{detname} ({YEAR})")

    raw = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["curdatetime"]),
            "intersection_id": label,
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
            "avg_speed_kph": pd.to_numeric(raw["speed"], errors="coerce") * 1.60934,
            "occupancy": pd.to_numeric(raw["occupancy"], errors="coerce").clip(0, 100) / 100,
        }
    )


def fetch() -> pd.DataFrame:
    frames = [_fetch_detector(intname, detname, label) for intname, detname, label in DETECTORS]
    return finalize(pd.concat(frames, ignore_index=True))
