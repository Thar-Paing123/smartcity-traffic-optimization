# Technical Architecture

Technical companion to [BUSINESS_SYSTEM_DESIGN.md](BUSINESS_SYSTEM_DESIGN.md). Describes how the system in `src/` implements each business-level layer.

## Component Diagram

```text
┌─────────────────────┐
│  ingestion/simulate  │  synthetic data generator (stand-in for cameras/GPS/historical feeds)
└──────────┬───────────┘
           │ long-format CSV: timestamp, intersection_id, volume, avg_speed_kph, occupancy
           ▼
┌─────────────────────┐
│  ingestion/pipeline  │  pivot → normalize → sliding-window → tf.data.Dataset
└──────────┬───────────┘
           │ (batch, window=12, n_intersections*n_features)
           ▼
┌─────────────────────┐
│  models/traffic_model│  Keras LSTM forecaster + recommend_signal_timing()
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐        ┌──────────────────────┐
│       train.py        │ ───▶ │ artifacts/model/*.keras│ (retrainable checkpoint)
│ tf.distribute.Mirrored │ ───▶ │ artifacts/model/saved_ │ (serving-ready SavedModel)
│ Strategy + TensorBoard │       │ model                 │
└─────────────────────┘        └──────────┬───────────┘
                                            ▼
                                 ┌──────────────────────┐
                                 │  serve/export_model.py │  versions into TF Serving layout
                                 └──────────┬───────────┘
                                            ▼
                                 ┌──────────────────────┐
                                 │ serve/Dockerfile       │  tensorflow/serving container
                                 │ (REST :8501 / gRPC :8500)│
                                 └──────────┬───────────┘
                                            ▼
                                 Traffic management center / signal controllers
                                            ▲
                                            │ evaluates + triggers retrain on drift
                                 ┌──────────────────────┐
                                 │ monitor/retrain.py     │
                                 └──────────────────────┘
```

## Data Flow

1. `ingestion/simulate.py` (or a future live feed) produces long-format rows per intersection.
2. `ingestion/pipeline.py` pivots to `(time, intersection, feature)`, normalizes with global mean/std, and slices into `(window=12, horizon=3)` sequence pairs, exposed as a shuffled, batched, prefetched `tf.data.Dataset`.
3. `train.py` builds the model inside a `tf.distribute.MirroredStrategy` scope, fits it with `TensorBoard`, `ModelCheckpoint`, and `EarlyStopping` callbacks, then writes both a retrainable `.keras` model and a serving-ready `SavedModel` export.
4. `serve/export_model.py` copies the SavedModel into `artifacts/serving/<model_name>/<unix_timestamp>/`, the directory layout TensorFlow Serving auto-discovers by highest version number.
5. The Docker image in `serve/Dockerfile` runs `tensorflow/serving` against that directory, exposing REST (8501) and gRPC (8500) endpoints.
6. `monitor/retrain.py` periodically evaluates the `.keras` model against fresh data; if MAE exceeds a threshold, it invokes `train.py` again and the cycle repeats, producing a new versioned export without taking the serving endpoint down.

## Model Architecture

`models/traffic_model.build_model()`:

- **Input**: `(window=12, n_intersections * n_features)` — 12 past time steps, all intersections and features flattened per step
- **LSTM(128, return_sequences=True) → LSTM(64)** — sequence encoding
- **Dense(128, relu) → Dropout(0.2)** — regularized representation
- **Dense(horizon * output_dim) → Reshape((horizon, output_dim))** — multi-step, multi-intersection forecast
- **Loss/metric**: MSE / MAE, Adam optimizer

`models/traffic_model.recommend_signal_timing()` maps a single intersection's predicted occupancy (0–1) to a green-light duration via linear interpolation between `min_green_s` (15s) and `max_green_s` (90s) — a deliberately simple, bounded decision rule so the model's output can't request an unsafe signal timing.

## Scaling Strategy

- **Today**: `tf.distribute.MirroredStrategy` — synchronous data-parallel training across all GPUs on one machine; degrades to single-device (CPU or 1 GPU) automatically.
- **Next**: swap to `tf.distribute.MultiWorkerMirroredStrategy` or `TPUStrategy` for multi-machine/TPU pod training as data volume grows — the surrounding training code in `train.py` does not need to change, only the strategy object.
- **Serving**: TensorFlow Serving scales horizontally by running multiple container replicas behind a load balancer; model versioning already supports rolling updates with no downtime.
- **Cloud**: both training (GCP AI Platform/Vertex AI, AWS SageMaker, or raw GPU/TPU VMs) and serving (containers on GKE/EKS or a managed serving endpoint) map directly onto this design without architectural changes.

## Technology Stack

| Concern | Technology |
|---|---|
| Modeling | TensorFlow / Keras (LSTM) |
| Data pipeline | `tf.data` |
| Distributed training | `tf.distribute.MirroredStrategy` |
| Experiment tracking | TensorBoard |
| Serving | TensorFlow Serving (REST + gRPC) |
| Deployment unit | Docker |
| Scheduling (retraining) | External scheduler (cron/Airflow) invoking `monitor/retrain.py` |

## Known Gaps / Future Work

- No real ingestion connector yet — `ingestion/simulate.py` must be replaced by a live source implementing the same long-format schema consumed by `ingestion/pipeline.load_historical`.
- No API gateway/auth layer in front of TensorFlow Serving — needed before exposing it beyond an internal network.
- No automated scheduler is included for `monitor/retrain.py`; it's designed to be invoked by one (cron, Airflow, etc.), not to self-schedule.
