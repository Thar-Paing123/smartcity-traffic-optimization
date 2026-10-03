"""Live dashboard: pick a country/city and see traffic forecasts, a map, and
signal-timing recommendations per intersection.

Scans artifacts/models/<country>/<city>/ for pre-trained models (see
train_all.py) and data/real/<country>/<city>/locations.json for intersection
coordinates, so switching the selection is instant — no retraining or
geocoding happens in the dashboard itself.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import tensorflow as tf

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ingestion.pipeline import FEATURE_COLUMNS, intersection_ids, latest_window, load_stats  # noqa: E402
from ingestion.simulate import generate_intersection_data, generate_locations  # noqa: E402
from models.traffic_model import recommend_signal_timing  # noqa: E402

MODELS_DIR = os.path.join(ROOT_DIR, "artifacts", "models")
DATA_REAL_DIR = os.path.join(ROOT_DIR, "data", "real")


def pretty(name: str) -> str:
    return name.replace("_", " ")


def discover_models() -> dict:
    """{country_display: {city_display: model_dir}} for every pre-trained model found."""
    registry = {}
    for model_dir in sorted(glob.glob(os.path.join(MODELS_DIR, "*", "*"))):
        if not os.path.exists(os.path.join(model_dir, "stats.json")):
            continue
        city_slug = os.path.basename(model_dir)
        country_slug = os.path.basename(os.path.dirname(model_dir))
        registry.setdefault(pretty(country_slug), {})[pretty(city_slug)] = model_dir
    return registry


st.set_page_config(page_title="SmartCity Traffic Optimization", layout="wide")
st.title("SmartCity Traffic Optimization — Live Dashboard")

registry = discover_models()
if not registry:
    st.warning("No trained models found yet. Fetch real data and train, then reload this page:")
    st.code("python -m ingestion.real_sources.fetch_all\npython train_all.py", language="bash")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", sorted(registry), index=sorted(registry).index("United Kingdom") if "United Kingdom" in registry else 0)
with col2:
    cities = sorted(registry[country])
    city = st.selectbox("City", cities)

model_dir = registry[country][city]


@st.cache_resource
def load_model_and_stats(model_dir: str):
    return tf.keras.models.load_model(os.path.join(model_dir, "model.keras")), load_stats(os.path.join(model_dir, "stats.json"))


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    return generate_intersection_data()


@st.cache_data
def load_locations(country: str, city: str, ids: tuple) -> dict:
    locations_path = os.path.join(DATA_REAL_DIR, country.replace(" ", "_"), city.replace(" ", "_"), "locations.json")
    if os.path.exists(locations_path):
        with open(locations_path) as f:
            return json.load(f)
    return generate_locations(list(ids))  # demo/synthetic city: fabricated grid, not a real place


model, stats = load_model_and_stats(model_dir)
df = load_data(stats.get("csv_path"))

window, horizon = stats["window"], stats["horizon"]
mean, std = stats["mean"][0], stats["std"][0]  # (1, features) — broadcasts over intersections
ids = intersection_ids(df)
locations = load_locations(country, city, tuple(ids))

raw_window = latest_window(df, window)  # (window, intersections, features)
model_input = ((raw_window - mean) / std).reshape(1, window, -1).astype("float32")

prediction = model.predict(model_input, verbose=0).reshape(horizon, len(ids), len(FEATURE_COLUMNS))
forecast = prediction * std + mean  # denormalized, shape (horizon, intersections, features)

occ_idx = FEATURE_COLUMNS.index("occupancy")
vol_idx = FEATURE_COLUMNS.index("volume")
speed_idx = FEATURE_COLUMNS.index("avg_speed_kph")
next_step = forecast[0]


def occupancy_color(occ: float) -> list:
    """Green (free-flowing) -> red (congested), for the map markers."""
    occ = max(0.0, min(1.0, occ))
    return [int(255 * occ), int(255 * (1 - occ)), 40, 200]


rows = []
for i, intersection_id in enumerate(ids):
    predicted_occupancy = float(np.clip(next_step[i, occ_idx], 0, 1))
    lat, lon = locations.get(intersection_id, (None, None))
    rows.append(
        {
            "Intersection": intersection_id,
            "lat": lat,
            "lon": lon,
            "Current Occupancy": round(float(raw_window[-1, i, occ_idx]), 3),
            "Predicted Occupancy (next interval)": round(predicted_occupancy, 3),
            "Predicted Volume": round(float(next_step[i, vol_idx]), 1),
            "Predicted Speed (kph)": round(float(next_step[i, speed_idx]), 1),
            "Recommended Green Time (s)": recommend_signal_timing(predicted_occupancy),
            "color": occupancy_color(predicted_occupancy),
            "radius": 80 + 400 * predicted_occupancy,
        }
    )

summary = pd.DataFrame(rows).sort_values("Predicted Occupancy (next interval)", ascending=False)

st.caption(
    f"Source data: `{stats.get('csv_path', 'synthetic (generated on the fly)')}`"
    + (" — simulated, not a real place." if country == "Demo" else "")
)

st.subheader("Map")
map_df = summary.dropna(subset=["lat", "lon"])
if not map_df.empty:
    view_state = pdk.ViewState(
        latitude=float(map_df["lat"].mean()),
        longitude=float(map_df["lon"].mean()),
        zoom=10 if (map_df["lat"].max() - map_df["lat"].min()) < 0.3 else 6,
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=8,
        radius_max_pixels=40,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None,
            tooltip={"text": "{Intersection}\nPredicted occupancy: {Predicted Occupancy (next interval)}\nRecommended green: {Recommended Green Time (s)}s"},
        )
    )
    st.caption("Marker color/size scale with predicted congestion for the next interval (green = free-flowing, red = congested).")
else:
    st.info("No coordinates available for this dataset's intersections.")

st.subheader("Per-Intersection Forecast & Signal Recommendation")
st.dataframe(summary.drop(columns=["lat", "lon", "color", "radius"]), use_container_width=True, hide_index=True)

st.subheader("Intersection Detail")
selected = st.selectbox("Select an intersection", ids)
sel_idx = ids.index(selected)

history = df[df["intersection_id"] == selected].tail(window * 4)[["timestamp", "occupancy"]]
freq = pd.infer_freq(history["timestamp"]) or "5min"
forecast_times = pd.date_range(history["timestamp"].iloc[-1], periods=horizon + 1, freq=freq)[1:]

chart_df = pd.concat(
    [
        history.assign(series="history"),
        pd.DataFrame({"timestamp": forecast_times, "occupancy": forecast[:, sel_idx, occ_idx], "series": "forecast"}),
    ]
)
st.line_chart(chart_df.pivot(index="timestamp", columns="series", values="occupancy"))

csv_path = stats.get("csv_path") or ""
if os.path.join("data", "real") in csv_path.replace(os.sep, "/"):
    source_note = "Data is from a free public real-world source (see src/ingestion/real_sources/)."
elif os.path.exists(csv_path):
    source_note = "Data is historical/previously-generated (see ingestion/simulate.py)."
else:
    source_note = "Data is simulated until live camera/GPS feeds are connected (see ingestion/simulate.py)."

st.caption(f"{source_note} Recommended green time is advisory — see BUSINESS_SYSTEM_DESIGN.md.")

if st.button("Refresh"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
