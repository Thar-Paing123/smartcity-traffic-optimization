# Skills Demonstrated

This project applies skills across machine learning, data engineering, MLOps, and systems design, as implemented in `src/`.

## Machine Learning & Deep Learning

- Time-series forecasting with sequence models (`src/models/traffic_model.py`)
- Keras (Functional API): `Input`, stacked `LSTM` layers, `Dense`, `Dropout`, `Reshape`
- Multi-variate, multi-entity forecasting (joint modeling of volume, speed, and occupancy across many intersections at once)
- Translating model output into a decision/action (predicted occupancy → recommended signal green-light duration), not just a raw prediction
- Model evaluation with MSE/MAE and validation-based early stopping

## Data Engineering

- Designing a data schema for streaming/time-series sensor data (`timestamp, intersection_id, volume, avg_speed_kph, occupancy`)
- Building `tf.data.Dataset` pipelines: sliding-window sequence construction, normalization, batching, shuffling, prefetching
- Synthetic data generation for development and testing before real data sources are available (`src/ingestion/simulate.py`) — seasonal patterns, noise, and injected anomalies (incidents)
- Separating ingestion/preprocessing logic from model code so either can change independently
- Integrating heterogeneous real-world public data APIs (REST/JSON, Socrata, zipped CSV) and reconciling mismatched schemas — different sources measure different subsets of fields — into one common schema (`src/ingestion/real_sources/`)
- Domain-informed feature derivation: estimating occupancy from flow and speed via a traffic-density proxy when a source doesn't measure it directly, rather than fabricating a value
- Honest handling of missing data: imputing to the feature mean instead of zero-filling, so "not measured" doesn't get misread as "no traffic"

## MLOps / Production ML

- Distributed training via `tf.distribute.MirroredStrategy`, with a training loop that scales from one device to many without code changes
- Model export for production serving (`model.export()` → TensorFlow SavedModel) distinct from the retrainable checkpoint format (`model.save()` → `.keras`)
- Versioned model deployment layout for TensorFlow Serving (`src/serve/export_model.py`), enabling zero-downtime rollout
- Containerized serving with Docker (`src/serve/Dockerfile`, `tensorflow/serving` base image)
- Training observability via TensorBoard callbacks
- Automated drift detection and retraining (`src/monitor/retrain.py`) — evaluate → compare against a threshold → retrain → re-export
- Batch-training a model per dataset (`src/train_all.py`) so a multi-city UI can switch instantly instead of retraining on demand
- Interactive geospatial visualization: a country/city-selectable map (pydeck `ScatterplotLayer`) colored and sized by predicted congestion, alongside per-intersection forecasts (`src/dashboard/app.py`)

## Software Engineering

- Modular project structure separating ingestion, modeling, training, serving, and monitoring concerns
- CLI-configurable scripts (`argparse`) for reproducible runs with different hyperparameters
- Path handling robust to the script's working directory (so scripts run correctly whether invoked from the repo root or their own subdirectory)
- End-to-end verification of a multi-stage pipeline in an isolated environment before declaring it complete

## Cloud & Infrastructure Thinking

- Designing for horizontal scale-out (multi-GPU/TPU, cloud platforms) as a first-class requirement rather than an afterthought
- Containerization as the deployment unit, enabling portability between on-prem and cloud environments

## Domain Knowledge

- Intelligent Transportation Systems (ITS) concepts: congestion modeling, signal timing trade-offs, incident impact on flow
- Translating a city-operations problem (congestion, signal inefficiency) into a well-scoped ML problem (sequence forecasting + bounded decision rule)
