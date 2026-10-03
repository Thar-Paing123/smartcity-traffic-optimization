"""UK traffic data via National Highways' WebTRIS Traffic Flow API (free, no API key).

Source: https://webtris.nationalhighways.co.uk (England's strategic road
network — motorways/trunk roads, not local signal-controlled intersections).
Two MIDAS monitoring sites per city, chosen because that motorway is
uniquely associated with the city (M25 = London orbital, M60 = Manchester
ring road, M32 = Bristol spur, M62 at the Leeds end of the Leeds-Manchester
corridor) and manually checked against their GPS coordinates.

Both volume and average speed are directly measured by the roadside sensors.
WebTRIS does not report occupancy, so it's estimated via a flow/speed
density proxy (see common.density_proxy_occupancy).
"""

import time

import pandas as pd
import requests

from .common import density_proxy_occupancy, finalize, mph_to_kph

BASE_URL = "https://webtris.nationalhighways.co.uk/api/v1"
COUNTRY = "United Kingdom"

SITES = {
    "London": [
        {"id": 5, "name": "M25/5764B", "lat": 51.5756168053165, "lon": 0.283161593410359},
        {"id": 8, "name": "M25/4876A", "lat": 51.4337486439773, "lon": -0.538795788123459},
    ],
    "Manchester": [
        {"id": 100, "name": "M60/9462B", "lat": 53.4789693723244, "lon": -2.11907034674187},
        {"id": 134, "name": "M60/9492A", "lat": 53.4576935244612, "lon": -2.1360583677519},
    ],
    "Bristol": [
        {"id": 243, "name": "M32/5069B", "lat": 51.5035941828822, "lon": -2.5280962103214},
        {"id": 616, "name": "M32/5077A", "lat": 51.5104588395728, "lon": -2.52318986317759},
    ],
    "Leeds": [
        {"id": 18, "name": "M62/2099A", "lat": 53.7317663847111, "lon": -1.60804240457756},
        {"id": 13, "name": "M62/2328B", "lat": 53.7084340056232, "lon": -1.28311525912878},
    ],
}

LOCATIONS = {site["name"]: (site["lat"], site["lon"]) for sites in SITES.values() for site in sites}

DEFAULT_START_DATE = "01012020"
DEFAULT_END_DATE = "31012020"


def _fetch_site(site_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    rows = []
    page = 1
    while True:
        resp = requests.get(
            f"{BASE_URL}/reports/daily",
            params={
                "sites": site_id,
                "start_date": start_date,
                "end_date": end_date,
                "page": page,
                "page_size": 1000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            break
        payload = resp.json()
        rows.extend(payload.get("Rows", []))
        links = payload.get("Header", {}).get("links", [])
        if not any(link.get("rel") == "nextPage" for link in links):
            break
        page += 1
        time.sleep(0.2)
    return pd.DataFrame(rows)


def fetch_city(city: str, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    """Fetch 15-minute volume/speed data for every configured site in `city`.

    Dates are DDMMYYYY strings, per the WebTRIS API.
    """
    if city not in SITES:
        raise ValueError(f"Unknown city '{city}'. Available: {list(SITES)}")

    frames = []
    for site in SITES[city]:
        raw = _fetch_site(site["id"], start_date, end_date)
        if raw.empty:
            raise RuntimeError(f"No WebTRIS data returned for {city} site {site['name']} ({start_date}-{end_date})")

        timestamp = pd.to_datetime(raw["Report Date"]).dt.normalize() + pd.to_timedelta(raw["Time Period Ending"])
        volume = pd.to_numeric(raw["Total Volume"], errors="coerce").fillna(0)
        speed_kph = mph_to_kph(pd.to_numeric(raw["Avg mph"], errors="coerce").ffill().fillna(0))

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "intersection_id": site["name"],
                    "volume": volume,
                    "avg_speed_kph": speed_kph,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df["occupancy"] = density_proxy_occupancy(df["volume"], df["avg_speed_kph"])
    return finalize(df)


def fetch_all(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> dict:
    """Fetch every configured UK city. Returns {city: DataFrame}."""
    return {city: fetch_city(city, start_date, end_date) for city in SITES}
