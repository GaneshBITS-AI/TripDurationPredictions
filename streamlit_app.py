"""
ETA Prediction — Streamlit UI
==============================

Thin client over the FastAPI service in eta_pipeline/serving/api.py. It does
no feature engineering or model loading itself - every prediction is a plain
HTTP call to the API's /predict endpoint, so the UI and the model server can
be started, restarted, and scaled independently.

Usage
-----
    python serve.py                              # start the API (separate terminal)
    streamlit run streamlit_app.py                # start this UI
"""

from __future__ import annotations

from datetime import date, time as dtime

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(page_title="ETA Prediction", page_icon="🚕", layout="centered")

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


# ----------------------------- API helpers -----------------------------

def _api_get(path: str, timeout: float = 5.0):
    resp = requests.get(f"{st.session_state.api_url}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _api_post(path: str, payload: dict, timeout: float = 15.0):
    resp = requests.post(f"{st.session_state.api_url}{path}", json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text) if resp.headers.get(
            "content-type", ""
        ).startswith("application/json") else resp.text
        raise RuntimeError(detail)
    return resp.json()


# ----------------------------- Sidebar: API status -----------------------------

with st.sidebar:
    st.header("API connection")
    st.session_state.api_url = st.text_input("API base URL", value=st.session_state.api_url)

    if st.button("Check health", use_container_width=True):
        st.session_state["_health_check"] = True

    if st.session_state.get("_health_check"):
        try:
            health = _api_get("/health")
            st.success(f"API healthy — model v{health['model_version']}")
            st.caption(f"dataset: {health['dataset_version']}")
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach {st.session_state.api_url}. Is `python serve.py` running?")
        except requests.exceptions.HTTPError as exc:
            st.error(f"API unhealthy: {exc.response.json().get('detail', exc)}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    st.divider()
    st.caption(
        "Start the API first:\n\n`python serve.py`\n\n"
        "then run this app with:\n\n`streamlit run streamlit_app.py`"
    )


# ----------------------------- Main form -----------------------------

st.title("🚕 NYC Taxi ETA Predictor")
st.write("Enter trip details to predict ride duration using the trained Production model.")

NYC_PRESETS = {
    "Midtown → JFK Airport": (40.7549, -73.9840, 40.6413, -73.7781),
    "Times Square → Central Park": (40.7580, -73.9855, 40.7812, -73.9665),
    "Wall St → Brooklyn Bridge": (40.7074, -74.0113, 40.7061, -73.9969),
    "Custom": None,
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Route")
    preset = st.selectbox("Quick preset", list(NYC_PRESETS.keys()))
    preset_coords = NYC_PRESETS[preset]

    pickup_lat = st.number_input(
        "Pickup latitude", value=preset_coords[0] if preset_coords else 40.7614,
        min_value=40.50, max_value=40.92, format="%.4f",
    )
    pickup_lon = st.number_input(
        "Pickup longitude", value=preset_coords[1] if preset_coords else -73.9776,
        min_value=-74.26, max_value=-73.68, format="%.4f",
    )
    dropoff_lat = st.number_input(
        "Dropoff latitude", value=preset_coords[2] if preset_coords else 40.6413,
        min_value=40.50, max_value=40.92, format="%.4f",
    )
    dropoff_lon = st.number_input(
        "Dropoff longitude", value=preset_coords[3] if preset_coords else -73.7781,
        min_value=-74.26, max_value=-73.68, format="%.4f",
    )

    st.map(
        pd.DataFrame(
            {"lat": [pickup_lat, dropoff_lat], "lon": [pickup_lon, dropoff_lon]}
        ),
        size=40,
    )

with col2:
    st.subheader("Trip details")
    pickup_date = st.date_input("Pickup date", value=date(2016, 6, 1))
    pickup_time = st.time_input("Pickup time", value=dtime(8, 30))
    passenger_count = st.slider("Passenger count", min_value=1, max_value=6, value=1)
    vendor_id = st.selectbox("Vendor ID", [1, 2], index=1)
    store_and_fwd_flag = st.selectbox("Store & forward flag", ["N", "Y"])

    with st.expander("Weather (optional — defaults to none)"):
        use_weather = st.checkbox("Specify weather", value=False)
        avg_temp_f = st.number_input("Avg temperature (°F)", value=55.0) if use_weather else None
        precipitation = st.number_input("Precipitation (in)", value=0.0, min_value=0.0) if use_weather else None
        snow_depth = st.number_input("Snow depth (in)", value=0.0, min_value=0.0) if use_weather else None

st.divider()

if st.button("Predict ETA", type="primary", use_container_width=True):
    payload = {
        "pickup_datetime": f"{pickup_date.isoformat()} {pickup_time.strftime('%H:%M:%S')}",
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "passenger_count": passenger_count,
        "vendor_id": vendor_id,
        "store_and_fwd_flag": store_and_fwd_flag,
        "avg_temp_f": avg_temp_f,
        "precipitation": precipitation,
        "snow_depth": snow_depth,
    }

    try:
        with st.spinner("Calling prediction API..."):
            result = _api_post("/predict", payload)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted ETA", f"{result['eta_minutes']:.1f} min")
        m2.metric("In seconds", f"{result['eta_seconds']:,} s")
        m3.metric("Latency", f"{result['latency_ms']:.0f} ms")

        st.caption(
            f"model: **{result['model_name']}** v{result['model_version']} · "
            f"dataset: **{result['dataset_version']}** · "
            f"features: {result['input_features']}"
        )

    except requests.exceptions.ConnectionError:
        st.error(
            f"Can't reach the API at {st.session_state.api_url}. "
            "Start it in another terminal with `python serve.py`."
        )
    except RuntimeError as exc:
        st.error(f"Prediction failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
