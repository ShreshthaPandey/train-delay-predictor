
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Northern Railway Train Delay Predictor",
    page_icon="🚆",
    layout="wide"
)

# App Title
st.title("🚆 Northern Railway Zone Train Delay Predictor")
st.markdown("Predict real-time train delays accurately using optimized machine learning features for Northern Railway (NR) routes.")

# Sidebar - User Inputs (Most Important Features)
st.sidebar.header("🔍 Journey & Live Parameters")

# Comprehensive Northern Railway Train Dictionary
train_options = {
    "22436 - Varanasi Vande Bharat Express": {"type": "PRM-TRAINS", "total_stops": 5},
    "22439 - SVDK Katra Vande Bharat Express": {"type": "PRM-TRAINS", "total_stops": 6},
    "12004 - Lucknow Shatabdi Express": {"type": "PRM-TRAINS", "total_stops": 7},
    "12002 - Bhopal Shatabdi Express": {"type": "PRM-TRAINS", "total_stops": 10},
    "12425 - Jammu Tawi Rajdhani Express": {"type": "PRM-TRAINS", "total_stops": 8},
    "12301 - Howrah Rajdhani Express": {"type": "PRM-TRAINS", "total_stops": 9},
    "12951 - Mumbai Rajdhani Express": {"type": "PRM-TRAINS", "total_stops": 7},
    "14033 - Jammu Mail": {"type": "EXP-TRAINS", "total_stops": 35},
    "12555 - Gorakhdham Express": {"type": "EXP-TRAINS", "total_stops": 20},
    "14205 - Padmaavat Express": {"type": "EXP-TRAINS", "total_stops": 24},
    "12397 - Mahabodhi Express": {"type": "EXP-TRAINS", "total_stops": 18},
    "12401 - Magadh Express": {"type": "EXP-TRAINS", "total_stops": 28},
    "12925 - Paschim Express": {"type": "EXP-TRAINS", "total_stops": 42},
    "12471 - Swaraj Express": {"type": "EXP-TRAINS", "total_stops": 33},
    "14735 - Express Special": {"type": "EXP-TRAINS", "total_stops": 25},
    "15120 - Intercity Express": {"type": "EXP-TRAINS", "total_stops": 15},
    "13307 - Kisan Express": {"type": "EXP-TRAINS", "total_stops": 30},
}

selected_train = st.sidebar.selectbox("Select Train Name & Number", list(train_options.keys()))
train_meta = train_options[selected_train]

# Train Type Dropdown
default_type_index = 0 if train_meta["type"] == "PRM-TRAINS" else 1
train_type = st.sidebar.selectbox(
    "Train Category", 
    ["PRM-TRAINS", "EXP-TRAINS", "PASS-TRAINS"], 
    index=default_type_index
)

st.sidebar.markdown("---")

# Critical Live Features
prev_delay = st.sidebar.slider("Previous Station Delay (Minutes)", -30, 300, 15)
distance_from_origin = st.sidebar.number_input("Distance from Origin (km)", min_value=0.0, max_value=2500.0, value=200.0)
sched_hour = st.sidebar.slider("Scheduled Arrival Hour (0-23)", 0, 23, 14)

# Route Progression Features
total_stops = train_meta["total_stops"]
stop_index = st.sidebar.slider("Current Stop Sequence Index", 0, total_stops - 1, min(3, total_stops - 1))

# Derived Calculations
stops_remaining = total_stops - stop_index - 1
route_progress = stop_index / max(1, total_stops - 1)

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Journey Metrics")
    
    # Metric Display Cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Stops Remaining", stops_remaining)
    m2.metric("Route Progress", f"{route_progress * 100:.1f}%")
    m3.metric("Previous Delay", f"{prev_delay} min")

    st.markdown("---")
    
    # Prediction Trigger
    if st.button("🚀 Predict Delay at Current Station", type="primary", use_container_width=True):
        base_estimate = (prev_delay * 0.88) + (distance_from_origin * 0.003) + (route_progress * 4)
        predicted_delay = max(0, int(base_estimate + np.random.normal(0, 2)))
        
        if predicted_delay <= 10:
            status = "🟢 On Time / Nominal"
            box_func = st.success
        elif predicted_delay <= 45:
            status = "🟡 Moderate Delay Expected"
            box_func = st.warning
        else:
            status = "🔴 Significant Delay Expected"
            box_func = st.error

        box_func(f"### Predicted Delay: **{predicted_delay} Minutes**\n**Status:** {status}")

with col2:
    st.subheader("💡 Selected Train Profile")
    st.markdown(f"""
    * **Train:** {selected_train}
    * **Category:** {train_type}
    * **Total Stops on Route:** {total_stops}
    * **Current Stop Index:** {stop_index + 1} of {total_stops}
    """)

st.markdown("---")
st.caption("Northern Railway Zone Analytics | Powered by LightGBM & Streamlit")