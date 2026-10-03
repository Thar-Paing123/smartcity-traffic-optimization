FROM tensorflow/serving:latest

ENV MODEL_NAME=traffic_forecaster
ENV MODEL_BASE_PATH=/models

COPY artifacts/serving/traffic_forecaster /models/traffic_forecaster

EXPOSE 8500 8501
