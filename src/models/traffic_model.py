"""Keras model for multi-intersection traffic forecasting.

Built with the Keras API specifically for fast iteration on architecture,
and because it gives direct access to TensorFlow's distributed training
strategies and TensorFlow Serving export without extra glue code.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(window: int, horizon: int, n_intersections: int, n_features: int) -> keras.Model:
    input_dim = n_intersections * n_features
    output_dim = n_intersections * n_features

    inputs = keras.Input(shape=(window, input_dim), name="traffic_window")
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(horizon * output_dim)(x)
    outputs = layers.Reshape((horizon, output_dim), name="forecast")(x)

    model = keras.Model(inputs, outputs, name="traffic_forecaster")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def recommend_signal_timing(predicted_occupancy: float, min_green_s: int = 15, max_green_s: int = 90) -> int:
    """Map a predicted occupancy value (0-1) at an intersection to a green-light duration.

    Linear interpolation between min/max green time: higher predicted
    occupancy (more queued traffic) gets more green time, within fixed
    safety bounds enforced by the signal controller.
    """
    occupancy = max(0.0, min(1.0, predicted_occupancy))
    return int(round(min_green_s + occupancy * (max_green_s - min_green_s)))
