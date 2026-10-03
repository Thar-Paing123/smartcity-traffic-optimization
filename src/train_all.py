"""Trains one model per available dataset: the synthetic demo city plus every
city fetched by ingestion/real_sources/fetch_all.py.

Lets the dashboard switch between country/city selections instantly, reading
a pre-trained model per city, instead of retraining on each selection. Run
this after (re)fetching real data:

    python -m ingestion.real_sources.fetch_all
    python train_all.py
"""

import glob
import os
import subprocess
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_REAL_DIR = os.path.join(ROOT_DIR, "data", "real")
MODELS_DIR = os.path.join(ROOT_DIR, "artifacts", "models")

EPOCHS = 15
DEMO_COUNTRY = "Demo"
DEMO_CITY = "Synthetic_City"


def discover_datasets():
    datasets = [(DEMO_COUNTRY, DEMO_CITY, os.path.join(ROOT_DIR, "data", "historical_traffic.csv"))]
    for csv_path in sorted(glob.glob(os.path.join(DATA_REAL_DIR, "*", "*", "traffic.csv"))):
        city_dir = os.path.dirname(csv_path)
        country_dir = os.path.dirname(city_dir)
        datasets.append((os.path.basename(country_dir), os.path.basename(city_dir), csv_path))
    return datasets


def main():
    for country, city, csv_path in discover_datasets():
        model_dir = os.path.join(MODELS_DIR, country, city)
        log_dir = os.path.join(model_dir, "logs")
        print(f"=== Training {country}/{city} ({csv_path}) ===")
        subprocess.run(
            [
                sys.executable,
                os.path.join(SRC_DIR, "train.py"),
                "--csv",
                csv_path,
                "--epochs",
                str(EPOCHS),
                "--model-dir",
                model_dir,
                "--log-dir",
                log_dir,
            ],
            check=True,
            cwd=SRC_DIR,
        )


if __name__ == "__main__":
    main()
