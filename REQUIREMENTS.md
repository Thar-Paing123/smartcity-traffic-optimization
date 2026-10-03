# Requirements

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | Ingest traffic data (volume, speed, occupancy) from cameras, GPS/connected vehicles, and historical records, keyed by intersection and timestamp. |
| FR2 | Preprocess raw readings into normalized, time-windowed sequences suitable for sequence forecasting. |
| FR3 | Forecast per-intersection traffic volume, average speed, and occupancy over a configurable horizon (default: next 3 intervals). |
| FR4 | Translate forecasted occupancy into a recommended signal green-light duration, within configurable safety bounds. |
| FR5 | Train models using distributed compute (multi-GPU today, multi-worker/TPU as volume grows) without changing training code. |
| FR6 | Deploy trained models for low-latency inference via TensorFlow Serving. |
| FR7 | Expose predictions/recommendations in a form consumable by the city's traffic control systems, including legacy signal controllers. |
| FR8 | Continuously evaluate model performance against fresh data in production. |
| FR9 | Automatically retrain and re-export the model when evaluation error exceeds a defined drift threshold. |
| FR10 | Support versioned, zero-downtime model rollout (new model version served alongside/replacing the previous one without interrupting traffic requests). |

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | **Scalability** — scale horizontally across GPUs/TPUs and cloud platforms (GCP, AWS) as intersections and data sources are added, from a single pilot district to city-wide and beyond. |
| NFR2 | **Latency** — inference must be fast enough to inform signal timing decisions on a near-real-time cycle (seconds, not minutes). |
| NFR3 | **Availability** — the serving layer must support rolling model updates without downtime, since signal control cannot tolerate extended outages. |
| NFR4 | **Extensibility** — new intersections, sensor types, or data sources must be addable without redesigning the pipeline. |
| NFR5 | **Observability** — training and serving behavior must be visible (metrics, logs) for debugging and performance tracking. |
| NFR6 | **Data privacy & compliance** — camera and GPS data must be handled in a way that avoids storing or exposing personally identifiable information (e.g., license plates, vehicle-level tracking) beyond what's needed for aggregate traffic metrics. |
| NFR7 | **Portability** — the system must run both on-prem (near existing traffic control infrastructure) and in the cloud, via containerized deployment. |
| NFR8 | **Maintainability** — ingestion, modeling, serving, and monitoring must be independently modifiable/replaceable components. |

## Data Requirements

- **Traffic camera feeds** — aggregate counts/metrics per intersection (not raw video processing in the current phase).
- **GPS / connected-vehicle data** — speed and location signals aggregated per intersection.
- **Historical traffic records** — used for initial model training and backtesting.
- **Intersection metadata** — location, lane configuration, existing signal timing plan (needed to bound recommended timing changes).

Expected schema for ingested data (see `src/ingestion/pipeline.py`): long-format rows of `timestamp, intersection_id, volume, avg_speed_kph, occupancy`.

## Out of Scope (Current Phase)

- Raw video/image object detection — the system assumes cameras (or an upstream system) already produce aggregate metrics.
- Direct integration with specific vendor signal-controller hardware/protocols (e.g., NTCIP) — treated as a future integration point.
- Multi-modal transport (pedestrians, cyclists, transit priority) — future extension.

## Assumptions & Constraints

- Model development uses TensorFlow/Keras, per [README.md](README.md#why-tensorflow).
- Until live feeds are connected, the system uses a synthetic data generator (`src/ingestion/simulate.py`) that reproduces realistic daily congestion patterns and random incidents, so the full pipeline is testable end-to-end.
- `src/ingestion/real_sources/` additionally fetches free public datasets for several real cities (see [README.md](README.md#real-world-datasets)) for development/testing against real traffic patterns. These are historical, low-rate, or partial-field (e.g., volume without speed) by nature of being free public sources — not a substitute for FR1's live per-intersection camera/GPS/historical feed in production.
- Signal timing recommendations are advisory outputs with enforced min/max bounds; final authority to apply them rests with existing traffic control systems/operators, not this model.
