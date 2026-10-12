
```python
import os
import joblib
import pandas as pd
import streamlit as st
from datetime import date

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Northern Railway | Train Delay Predictor",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = "model/train_delay_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"
CATEGORICAL_PATH = "model/categorical_columns.pkl"

# Optional CSV containing the engineered model features.
# Expected to contain the 25 model features and, ideally,
# run_date and delay_minutes.
REFERENCE_DATA_PATH = "data/train_delay_features.csv"


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f2f7fc 0%, #ffffff 65%);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #082b59, #0757a6);
        padding: 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(8, 43, 89, 0.15);
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e0edff;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        color: #082b59;
        font-size: 23px;
        font-weight: 700;
        margin: 18px 0;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #dce6f1;
        margin-bottom: 15px;
    }

    .result-card {
        background: linear-gradient(120deg, #082b59, #0757a6);
        padding: 28px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-top: 20px;
    }

    .result-card h2 {
        color: white;
        font-size: 23px;
    }

    .result-value {
        color: white;
        font-size: 43px;
        font-weight: 800;
        margin: 10px 0;
    }

    .result-caption {
        color: #e0edff;
        font-size: 15px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #0757a6;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-weight: 700;
        min-height: 48px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #082b59;
        color: white;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    categorical_columns = joblib.load(CATEGORICAL_PATH)

    return model, feature_columns, categorical_columns


@st.cache_data
def load_reference_data():
    if not os.path.exists(REFERENCE_DATA_PATH):
        return None

    return pd.read_csv(REFERENCE_DATA_PATH)


if not all(os.path.exists(p) for p in [
    MODEL_PATH,
    FEATURE_PATH,
    CATEGORICAL_PATH
]):
    st.error(
        "Model files are missing. Please check the model folder."
    )
    st.stop()

try:
    model, feature_columns, categorical_columns = (
        load_model_artifacts()
    )
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

reference_df = load_reference_data()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🚆 Northern Railway Zone</h1>
    <p style="font-size:23px; font-weight:700;">
        Train Delay Predictor
    </p>
    <p>
        Intelligent train-delay estimation using journey information
        and historical railway data.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# STATUS CARDS
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div style="color:#61748b;font-size:13px;">PREDICTION ENGINE</div>
        <div style="color:#082b59;font-size:21px;font-weight:700;">
            LightGBM
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div style="color:#61748b;font-size:13px;">MODEL FEATURES</div>
        <div style="color:#082b59;font-size:21px;font-weight:700;">
            25 Features
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    data_status = (
        "Reference data loaded"
        if reference_df is not None
        else "Reference data unavailable"
    )

    st.markdown(f"""
    <div class="info-card">
        <div style="color:#61748b;font-size:13px;">DATA STATUS</div>
        <div style="color:#082b59;font-size:17px;font-weight:700;">
            {data_status}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TRAIN AND STATION DROPDOWNS
# =========================================================
st.markdown(
    '<div class="section-title">📍 Journey Information</div>',
    unsafe_allow_html=True
)

# Populate train dropdown from reference data if available.
if (
    reference_df is not None
    and "train_name" in reference_df.columns
):
    train_options = sorted(
        reference_df["train_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
else:
    train_options = []

# Fallback names are for displaying the UI only.
if not train_options:
    train_options = [
        "Select a train"
    ]

# Only offer station choices when the data supports them.
station_options = []

if (
    reference_df is not None
    and "station_code" in reference_df.columns
):
    station_options = sorted(
        reference_df["station_code"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

with st.form("train_prediction_form"):

    left, right = st.columns(2)

    with left:
        selected_train = st.selectbox(
            "Select Train Name",
            options=train_options,
            help="Train names are populated from your reference dataset."
        )

        journey_date = st.date_input(
            "Journey Date",
            value=date.today()
        )

    with right:
        if station_options:
            selected_station = st.selectbox(
                "Current Station",
                options=station_options
            )
        else:
            selected_station = st.text_input(
                "Current Station Code",
                placeholder="Enter station code, e.g. NDLS"
            ).strip().upper()

        previous_station_delay = st.number_input(
            "Previous Station Delay (minutes)",
            min_value=-120.0,
            max_value=1440.0,
            value=0.0,
            step=1.0,
            help="Observed delay at the previous station."
        )

    delay_two_stops_back = st.number_input(
        "Delay Two Stops Back (minutes)",
        min_value=-120.0,
        max_value=1440.0,
        value=0.0,
        step=1.0,
        help="Observed delay two stations before the current station."
    )

    submitted = st.form_submit_button(
        "🚆 Predict Train Delay",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
if submitted:

    if selected_train == "Select a train":
        st.error(
            "Train options are unavailable. Load your train reference data first."
        )
        st.stop()

    if not selected_station:
        st.error("Please select or enter the current station.")
        st.stop()

    if reference_df is None:
        st.warning(
            "The model is loaded, but automatic feature generation is not "
            "configured yet. Add your engineered historical dataset at "
            "`data/train_delay_features.csv` to enable train and station "
            "lookups. The app will not invent missing model features."
        )
        st.stop()

    required_columns = set(feature_columns)

    if not required_columns.issubset(reference_df.columns):
        missing = sorted(
            required_columns - set(reference_df.columns)
        )

        st.error(
            "The reference dataset is missing model features: "
            + ", ".join(missing)
        )
        st.stop()

    # Find historical rows for the selected train and station.
    matches = reference_df[
        (reference_df["train_name"].astype(str) == selected_train)
        & (
            reference_df["station_code"].astype(str)
            == str(selected_station)
        )
    ].copy()

    if matches.empty:
        st.error(
            "No matching train-and-station record exists in the reference "
            "dataset. Choose a valid station for this train."
        )
        st.stop()

    # Use the most recent matching record as a feature template.
    # This is a fallback, not a true live feature-generation pipeline.
    if "run_date" in matches.columns:
        matches["run_date"] = pd.to_datetime(
            matches["run_date"],
            errors="coerce"
        )
        matches = matches.sort_values("run_date")

    input_df = matches.iloc[[-1]][feature_columns].copy()

    # Update features directly provided by the user.
    input_df.loc[:, "previous_station_delay_min"] = (
        previous_station_delay
    )

    input_df.loc[:, "delay_2_stops_back_min"] = (
        delay_two_stops_back
    )

    # Calendar features can be calculated from the selected date.
    if "day_of_week" in input_df.columns:
        input_df.loc[:, "day_of_week"] = journey_date.weekday()

    if "month" in input_df.columns:
        input_df.loc[:, "month"] = journey_date.month

    # Update the delay-change feature where applicable.
    if "delay_change_last_stop_min" in input_df.columns:
        input_df.loc[:, "delay_change_last_stop_min"] = (
            previous_station_delay - delay_two_stops_back
        )

    try:
        # Match categorical columns to the saved model.
        for col in categorical_columns:
            input_df[col] = input_df[col].astype("category")

        prediction = float(model.predict(input_df)[0])

        st.markdown("""
        <div class="section-title">📊 Prediction Result</div>
        """, unsafe_allow_html=True)

        if prediction > 0:
            status = "Expected Delay"
            description = "The train is estimated to arrive late."
        elif prediction < 0:
            status = "Expected Early Arrival"
            description = "The train is estimated to arrive ahead of schedule."
        else:
            status = "Expected On-Time Arrival"
            description = "The model estimates arrival close to schedule."

        st.markdown(f"""
        <div class="result-card">
            <h2>{status}</h2>
            <div class="result-value">{abs(prediction):.1f} min</div>
            <div class="result-caption">{description}</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This estimate uses a matching historical feature record. "
            "It is not a verified live railway status."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div style="text-align:center;color:#61748b;padding:12px;">
    <b>Northern Railway Zone Train Delay Predictor</b><br>
    Machine Learning • LightGBM • Streamlit<br>
    <small>For estimation and educational purposes.</small>
</div>
""", unsafe_allow_html=True)
```
