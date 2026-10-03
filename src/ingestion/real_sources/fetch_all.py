"""Fetches free, real-world public traffic datasets, organized by country and city.

Writes each city's data to data/real/<country>/<city>/traffic.csv (plus a
locations.json of intersection lat/lon for map display) in the schema
ingestion/pipeline.py expects, so it can be trained on directly:

    cd src && python train.py --csv ../data/real/United_Kingdom/London/traffic.csv

See each module in this package for the source, license, and which fields
are directly measured vs. unavailable for that dataset.
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ingestion.real_sources import uk_webtris, us_austin, us_nyc, us_uci_metro  # noqa: E402
from ingestion.real_sources.common import save_dataset, save_locations  # noqa: E402

DATA_REAL_DIR = os.path.join(ROOT_DIR, "data", "real")

# source key -> (country, {city: fetch_fn}, module used for LOCATIONS lookup)
SOURCES = {
    "uk": (uk_webtris.COUNTRY, {city: (lambda c=city: uk_webtris.fetch_city(c)) for city in uk_webtris.SITES}, uk_webtris),
    "us-minneapolis": (us_uci_metro.COUNTRY, {us_uci_metro.CITY: us_uci_metro.fetch}, us_uci_metro),
    "us-nyc": (us_nyc.COUNTRY, {us_nyc.CITY: us_nyc.fetch}, us_nyc),
    "us-austin": (us_austin.COUNTRY, {us_austin.CITY: us_austin.fetch}, us_austin),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    args = parser.parse_args()

    sources = list(SOURCES) if args.source == "all" else [args.source]
    for source in sources:
        print(f"Fetching {source}...")
        country, city_fetchers, module = SOURCES[source]
        for city, fetch_fn in city_fetchers.items():
            try:
                df = fetch_fn()
            except Exception as exc:  # network/source hiccups shouldn't abort the other cities/sources
                print(f"  {city} FAILED: {exc}")
                continue
            path = save_dataset(df, DATA_REAL_DIR, country, city)
            locations = {iid: module.LOCATIONS[iid] for iid in df["intersection_id"].unique() if iid in module.LOCATIONS}
            save_locations(locations, DATA_REAL_DIR, country, city)
            print(f"  {country} / {city}: {len(df)} rows, {df['intersection_id'].nunique()} intersection(s) -> {path}")


if __name__ == "__main__":
    main()
