# SmartCity Traffic Optimization

A machine learning system for optimizing traffic flow across a city using real-time and historical data from sensors, cameras, and connected infrastructure.

**Further reading:** [Requirements](REQUIREMENTS.md) · [Skills Demonstrated](SKILLS.md) · [Business System Design](BUSINESS_SYSTEM_DESIGN.md) · [Technical Architecture](ARCHITECTURE.md)

## Overview

As cities grow, traffic congestion, signal inefficiency, and incident response times become harder to manage with static, rule-based systems. This project applies machine learning to continuously analyze traffic data and recommend or automate optimizations — such as adaptive signal timing, congestion prediction, and incident detection — across a citywide network of intersections and roadways.

## Goals

- Reduce average commute times and congestion at key intersections
- Predict traffic buildup before it happens, rather than reacting to it
- Scale from a single district pilot to full city-wide coverage without a redesign
- Integrate with existing traffic control hardware and city systems rather than replacing them

## Why TensorFlow

The system is built on TensorFlow for two main reasons tied directly to the project's long-term requirements:

**Scalability.** As more data sources (sensors, cameras, connected vehicles) come online across the city and beyond, the system needs to grow without architectural rework. TensorFlow's distributed computing support, along with native compatibility with cloud platforms like Google Cloud and AWS, allows training and inference to scale horizontally across multiple GPUs or TPUs. This means the same modeling pipeline that works for a pilot district can scale to city-wide deployment as data volume grows.

**Integration with existing systems.** A traffic optimization model is only useful if it can act on real infrastructure. TensorFlow provides extensive APIs and tooling for integrating ML models with existing software and hardware. TensorFlow Serving in particular enables production deployment of trained models with low-friction integration into the city's traffic management systems — including legacy software and real-time signal control systems — without requiring those systems to be rebuilt first.

## Implementation Plan

1. **Data ingestion.** Set up data pipelines to collect and preprocess data from traffic cameras, GPS devices, and historical records in real-time.

2. **Model development.** Use TensorFlow — particularly through its Keras API — to build a predictive model that can forecast traffic patterns and recommend optimal signal timings.

3. **Model training.** Train the model on historical and real-time data, leveraging TensorFlow's distributed training capabilities for scalability.

4. **Model deployment.** Deploy the model using TensorFlow Serving, ensuring it integrates with the city's existing traffic management systems.

5. **Monitoring and maintenance.** Continuously monitor the model's performance using TensorBoard, and retrain the model as needed to adapt to changing traffic patterns.

## Project Structure

```text
src/
  ingestion/
    simulate.py       # synthetic camera/GPS/historical data generator (stand-in until live feeds are wired up)
    pipeline.py        # tf.data windowing + normalization pipeline
  models/
    traffic_model.py   # Keras LSTM forecaster + signal-timing recommendation
  train.py              # distributed training (tf.distribute.MirroredStrategy) + TensorBoard
  train_all.py           # trains one model per dataset (demo + every fetched city), for the dashboard's selector
  serve/
    export_model.py    # promotes a trained model into TF Serving's versioned layout
    Dockerfile           # tensorflow/serving container definition
  monitor/
    retrain.py          # evaluates the live model on fresh data, retrains on drift
  dashboard/
    app.py               # Streamlit UI: country/city selector, map, forecasts + signal-timing recommendations
  ingestion/real_sources/ # fetchers for free, real-world public traffic datasets (see below)
data/
  real/                  # real datasets + locations.json, organized data/real/<country>/<city>/
artifacts/
  models/                # one pre-trained model per country/city (see train_all.py), used by the dashboard
```

## Real-World Datasets

`src/ingestion/real_sources/` fetches free, publicly available traffic data and organizes it by country and city under `data/real/<country>/<city>/traffic.csv`, in the same schema the rest of the pipeline expects:

| Country | City | Source | Measured fields |
|---|---|---|---|
| United Kingdom | London, Manchester, Bristol, Leeds | [National Highways WebTRIS](https://webtris.nationalhighways.co.uk) (2 motorway sensor sites per city) | Volume + speed real; occupancy estimated from flow/speed (density proxy) |
| United States | Austin | [Austin Radar Traffic Counts](https://data.austintexas.gov/resource/i626-g7ub.json) (2 intersections, 2 directions each) | Volume + occupancy + speed all real — no derived fields |
| United States | Minneapolis-Saint Paul | [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | Volume real; speed/occupancy not measured by this source |
| United States | New York City | [NYC Open Data — Traffic Volume Counts](https://data.cityofnewyork.us/resource/7ym2-wayt.json) | Volume real; speed/occupancy not measured by this source |

Each city's folder also gets a `locations.json` (intersection → lat/lon) for the dashboard's map — real coordinates for the UK sites (from WebTRIS) and Austin intersections (manually identified, since the source doesn't publish them), approximate single points for the single-location US sources.

Fetch one or all of them, then pre-train a model for each so the dashboard's selector has something to show instantly:

```bash
cd src
python -m ingestion.real_sources.fetch_all --source all   # or: uk / us-austin / us-minneapolis / us-nyc
python train_all.py
```

You can still train directly on one city's data by hand:

```bash
python train.py --csv ../data/real/United_Kingdom/London/traffic.csv
```

No source here provides every field (volume/speed/occupancy) a real deployment would use (see [REQUIREMENTS.md](REQUIREMENTS.md#data-requirements)); fields a source doesn't measure are left missing and imputed to the feature mean rather than invented (see `ingestion/pipeline._impute`), and each fetcher module documents exactly what's real vs. derived for that source. All datasets above are small sites/segments/windows chosen to be free and fast to fetch — swap in a full feed per [BUSINESS_SYSTEM_DESIGN.md](BUSINESS_SYSTEM_DESIGN.md#rollout-strategy) as real sensor coverage grows.

## Getting Started

```bash
pip install -r requirements.txt
```

**1. Train** (generates a synthetic dataset on first run if `data/historical_traffic.csv` doesn't exist yet):

```bash
cd src
python train.py --epochs 20
```

**2. Monitor training** with TensorBoard:

```bash
tensorboard --logdir artifacts/logs
```

**3. Export for serving** (promotes the trained model into TF Serving's versioned layout):

```bash
cd src/serve
python export_model.py
```

**4. Serve** the exported model with TensorFlow Serving (requires Docker):

```bash
docker build -t traffic-forecaster -f src/serve/Dockerfile .
docker run -p 8500:8500 -p 8501:8501 traffic-forecaster
```

**5. Monitor for drift and retrain automatically** (run on a schedule in production):

```bash
cd src/monitor
python retrain.py
```

**6. View the dashboard** — pick a country and city, see them on a map (colored by predicted congestion), and get per-intersection forecasts and signal-timing recommendations. Requires `train_all.py` (or at least one `train.py` run) to have populated `artifacts/models/<country>/<city>/`:

```bash
streamlit run src/dashboard/app.py --server.port 8502
```

Then open `http://localhost:8502`. Use a different port than 8501 so it doesn't collide with TensorFlow Serving's default REST port if both are running on the same machine. The dashboard scans `artifacts/models/` for every pre-trained city and `data/real/<country>/<city>/locations.json` for its map coordinates — selecting a different country/city is instant (no retraining).

> **Note:** `simulate.py` generates realistic but synthetic traffic data (daily congestion patterns + random incidents) so the full pipeline is runnable before real camera/GPS/sensor feeds are connected. Swap `ingestion/pipeline.load_historical` for a real data source when ready — the model and training code don't need to change.

## Status

Core pipeline implemented and verified end-to-end: data ingestion (synthetic for now) → Keras model training with distributed-strategy support → TensorBoard monitoring → TensorFlow Serving export → drift-based retraining. Next steps: connect real camera/GPS data sources, and wire signal-timing recommendations into actual traffic controllers.
