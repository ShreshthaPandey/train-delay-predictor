"""
Indian Railways - Train Delay Predictor (Streamlit UI)

Run:
    pip install -r requirements.txt
    streamlit run app.py

Expected files (created by your notebook):
    model/train_delay_model.pkl
    model/feature_columns.pkl
    model/categorical_columns.pkl
Optional (auto-fills train name / type):
    railway_data/train_details.csv
"""

import datetime as dt
import io
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page + theme
# --------------------------------------------------------------------------
st.set_page_config(page_title="Train Delay Predictor", page_icon="🚆", layout="wide")

NAVY = "#14325C"
SIGNAL_AMBER = "#F2A900"
SIGNAL_GREEN = "#2E8B57"
SIGNAL_RED = "#C8372D"
STEEL = "#E9EEF3"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox {{ font-family: 'Barlow', sans-serif; }}
h1, h2, h3, .board-title {{ font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .01em; }}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}

.board {{
  background: {NAVY}; color: #fff; border-radius: 6px; padding: 20px 26px;
  display: flex; gap: 28px; align-items: center; flex-wrap: wrap;
  border-bottom: 6px solid {SIGNAL_AMBER};
}}
.board .big {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 76px; line-height: 1;
  color: {SIGNAL_AMBER}; min-width: 190px;
}}
.board .big small {{ font-size: 26px; color: #cfd9e8; margin-left: 6px; font-weight: 500; }}
.board .meta {{ flex: 1; min-width: 220px; }}
.board .meta .status {{ font-family: 'Barlow Condensed'; font-size: 30px; font-weight: 600; }}
.board .meta .line {{ color: #cfd9e8; font-size: 15px; margin-top: 4px; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:8px; }}

.hero-title {{ font-family:'Barlow Condensed'; font-size: 44px; font-weight:700; color:{NAVY}; margin:0; }}
.hero-sub {{ color:#4a5a70; margin:2px 0 18px 0; font-size:16px; }}
.note {{ background:{STEEL}; border-left:4px solid {NAVY}; padding:10px 14px; border-radius:4px; font-size:14px; }}
div[data-testid="stMetricValue"] {{ font-family:'Barlow Condensed'; font-weight:700; color:{NAVY}; }}
button[data-baseweb="tab"] {{ font-family:'Barlow Condensed'; font-size:19px; font-weight:600; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Small compatibility helpers (Streamlit changed width args across versions)
# --------------------------------------------------------------------------
def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_editor(df, **kw):
    try:
        return st.data_editor(df, width="stretch", **kw)
    except TypeError:
        return st.data_editor(df, use_container_width=True, **kw)


# --------------------------------------------------------------------------
# Load model artifacts
# --------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "model")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "train_delay_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    cat_cols = joblib.load(os.path.join(MODEL_DIR, "categorical_columns.pkl"))

    cat_in_order = [c for c in features if c in cat_cols]
    levels = {}
    try:
        pc = model.booster_.pandas_categorical or []
        if len(pc) == len(cat_in_order):
            levels = {c: [str(x) for x in lv] for c, lv in zip(cat_in_order, pc)}
    except Exception:
        pass

    gain = None
    try:
        gain = pd.DataFrame(
            {
                "feature": model.booster_.feature_name(),
                "gain": model.booster_.feature_importance("gain"),
                "splits": model.booster_.feature_importance("split"),
            }
        ).sort_values("gain", ascending=False)
    except Exception:
        pass
    return model, list(features), list(cat_cols), levels, gain


try:
    MODEL, FEATURES, CAT_COLS, LEVELS, IMPORTANCE = load_artifacts()
except Exception as e:
    st.markdown('<p class="hero-title">Train Delay Predictor</p>', unsafe_allow_html=True)
    st.error(
        f"Could not load the model from the `{MODEL_DIR}/` folder.\n\n"
        "Make sure these three files exist next to app.py:\n"
        "`model/train_delay_model.pkl`, `model/feature_columns.pkl`, `model/categorical_columns.pkl`.\n\n"
        f"Details: {e}"
    )
    st.stop()


# --------------------------------------------------------------------------
# Optional reference data (train name / type lookup)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_reference(raw: bytes):
    d = pd.read_csv(io.BytesIO(raw), dtype={"train_no": str})
    d["train_no"] = d["train_no"].str.zfill(5)
    keep = [c for c in ["train_no", "train_name", "type_code"] if c in d.columns]
    return d[keep].drop_duplicates("train_no")


def find_reference(uploaded):
    if uploaded is not None:
        return read_reference(uploaded.getvalue())
    for p in ["railway_data/train_details.csv", "train_details.csv"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return read_reference(f.read())
    return None


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
def to_frame(rows):
    """Turn a list of feature dicts into a model-ready DataFrame."""
    df = pd.DataFrame(rows)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FEATURES].copy()
    for c in FEATURES:
        if c in CAT_COLS:
            vals = df[c].astype(object).where(df[c].notna(), None)
            vals = vals.map(lambda v: None if v is None else str(v))
            if c in LEVELS:
                df[c] = pd.Categorical(vals, categories=LEVELS[c])
            else:
                df[c] = pd.Categorical(vals)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def predict_rows(rows):
    return MODEL.predict(to_frame(rows))


def build_row(ctx, hist):
    """
    Build the model input for one stop.
    ctx  : fixed info about the train/stop (see callers)
    hist : delays (minutes) already known at earlier stops of this run, in order
    """
    i = ctx["stop_index"]
    total = ctx["total_stops"]
    prev = hist[-1] if len(hist) >= 1 else np.nan
    prev2 = hist[-2] if len(hist) >= 2 else np.nan
    row = {
        "train_number": ctx["train_number"],
        "train_name": ctx.get("train_name"),
        "train_type": ctx.get("train_type"),
        "station_code": ctx["station_code"],
        "stop_sequence_no": i + 1,
        "stop_index": i,
        "total_stops_in_route": total,
        "stops_remaining": total - i - 1,
        "route_progress_ratio": i / max(total - 1, 1),
        "distance_from_origin_km": ctx["distance_km"],
        "scheduled_arrival_hour": ctx["sched_hour"],
        "journey_day_of_arrival": ctx.get("journey_day", 1),
        "day_of_week": ctx["date"].weekday(),
        "month": ctx["date"].month,
        "is_fog_season": int(ctx["fog"]),
        "has_schedule_match": 1,
        "avg_delay_last_7_runs_min": ctx["avg7"],
        "previous_station_delay_min": prev,
        "delay_2_stops_back_min": prev2,
        "delay_change_last_stop_min": prev - prev2 if len(hist) >= 2 else np.nan,
        "origin_delay_min": hist[0] if (i > 0 and len(hist) >= 1) else np.nan,
        "max_delay_so_far_min": max(hist) if (i > 0 and len(hist) >= 1) else np.nan,
        "avg_delay_so_far_min": (sum(hist) / i) if (i > 0 and len(hist) >= 1) else np.nan,
    }
    return row


def status_of(d):
    if d <= -1:
        return "Running early", SIGNAL_GREEN
    if d <= 5:
        return "On time", SIGNAL_GREEN
    if d <= 15:
        return "Minor delay", SIGNAL_AMBER
    if d <= 60:
        return "Delayed", SIGNAL_AMBER
    if d <= 180:
        return "Heavily delayed", SIGNAL_RED
    return "Severely delayed", SIGNAL_RED


def fmt_duration(m):
    m = int(round(m))
    sign = "-" if m < 0 else ""
    m = abs(m)
    h, mm = divmod(m, 60)
    return f"{sign}{h} h {mm:02d} min" if h else f"{sign}{mm} min"


def departure_board(delay, sched_dt, title):
    label, color = status_of(delay)
    eta = sched_dt + dt.timedelta(minutes=float(delay))
    sign = "+" if delay > 0 else ""
    st.markdown(
        f"""
<div class="board">
  <div class="big">{sign}{delay:.0f}<small>min</small></div>
  <div class="meta">
    <div class="status"><span class="dot" style="background:{color}"></span>{label}</div>
    <div class="line">{title}</div>
    <div class="line">Scheduled {sched_dt:%d %b, %H:%M} &nbsp;|&nbsp; Expected {eta:%d %b, %H:%M} ({fmt_duration(delay)})</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def gauge(delay, mae):
    top = max(120, abs(delay) * 1.4)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(delay),
            number={"suffix": " min", "font": {"family": "Barlow Condensed", "size": 40}},
            gauge={
                "axis": {"range": [-30, top]},
                "bar": {"color": NAVY, "thickness": 0.28},
                "steps": [
                    {"range": [-30, 5], "color": "#cfe8d9"},
                    {"range": [5, 60], "color": "#fbe7b0"},
                    {"range": [60, top], "color": "#f2c6c1"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
    return fig


def season_default(d):
    return d.month in (12, 1)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model")
    st.caption(f"LightGBM | {len(FEATURES)} input features")
    mae = st.number_input(
        "Test MAE (minutes) from your notebook",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Optional. If you enter your model's test MAE, predictions show a +/- band.",
    )
    ref_file = st.file_uploader("train_details.csv (optional)", type="csv",
                                help="Fills train name and type automatically.")
    REF = find_reference(ref_file)
    st.divider()
    st.markdown("**Two ways to use the model**")
    st.caption(
        "Before departure: leave live delays off. The model uses history only.\n\n"
        "While running: switch live delays on and enter the delay at earlier stops."
    )
    with st.expander("Features the model expects"):
        st.write(FEATURES)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="hero-title">Train Delay Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Predict how late a train will be at any stop, forecast a whole journey, '
    "and see what drives the delay.</p>",
    unsafe_allow_html=True,
)

tab_pred, tab_journey, tab_what, tab_batch, tab_model = st.tabs(
    ["Predict a stop", "Journey forecast", "What-if", "Batch (CSV)", "Model insights"]
)


# Shared pickers ------------------------------------------------------------
def pick_train(key):
    if "train_number" in LEVELS:
        tn = st.selectbox("Train number", LEVELS["train_number"], key=f"{key}_tn")
    else:
        tn = st.text_input("Train number", "12001", key=f"{key}_tn").zfill(5)

    name_guess, type_guess = None, None
    if REF is not None:
        hit = REF[REF["train_no"] == str(tn)]
        if len(hit):
            name_guess = hit.iloc[0].get("train_name")
            type_guess = hit.iloc[0].get("type_code")

    if "train_name" in FEATURES:
        if "train_name" in LEVELS:
            opts = LEVELS["train_name"]
            idx = opts.index(name_guess) if name_guess in opts else 0
            tname = st.selectbox("Train name", opts, index=idx, key=f"{key}_nm")
        else:
            tname = st.text_input("Train name", name_guess or "", key=f"{key}_nm")
    else:
        tname = name_guess
    if "train_type" in FEATURES:
        if "train_type" in LEVELS:
            opts = LEVELS["train_type"]
            idx = opts.index(str(type_guess)) if str(type_guess) in opts else 0
            ttype = st.selectbox("Train type", opts, index=idx, key=f"{key}_ty")
        else:
            ttype = st.text_input("Train type", str(type_guess or ""), key=f"{key}_ty")
    else:
        ttype = type_guess
    return tn, tname, ttype


def pick_station(key, label="Station code"):
    if "station_code" in LEVELS:
        return st.selectbox(label, LEVELS["station_code"], key=key)
    return st.text_input(label, "NDLS", key=key).upper()


# --------------------------------------------------------------------------
# TAB 1 - single stop
# --------------------------------------------------------------------------
with tab_pred:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Trip details")
        c1, c2 = st.columns(2)
        with c1:
            tn, tname, ttype = pick_train("p")
        with c2:
            station = pick_station("p_station")
            run_date = st.date_input("Run date", dt.date.today(), key="p_date")
            sched_time = st.time_input("Scheduled arrival", dt.time(9, 0), key="p_time")

        c3, c4, c5 = st.columns(3)
        total_stops = c3.number_input("Stops in route", 2, 200, 30, key="p_total")
        stop_index = c4.number_input("This is stop # (0 = origin)", 0, int(total_stops) - 1,
                                     min(10, int(total_stops) - 1), key="p_idx")
        dist = c5.number_input("Distance from origin (km)", 0.0, 4000.0, 450.0, step=10.0, key="p_dist")

        c6, c7 = st.columns(2)
        avg7 = c6.number_input("Train's avg delay, last 7 runs (min)", -60.0, 600.0, 20.0, step=1.0, key="p_avg7")
        fog = c7.toggle("Fog season (Dec - Jan)", value=season_default(run_date), key="p_fog")

        st.subheader("Live delays")
        live = st.toggle("The train is running - I know its earlier delays", value=True, key="p_live")
        prev = prev2 = origin = np.nan
        if live:
            l1, l2, l3 = st.columns(3)
            prev = l1.number_input("Delay at previous stop (min)", -60.0, 1440.0, 25.0, key="p_prev")
            prev2 = l2.number_input("Delay 2 stops back (min)", -60.0, 1440.0, 20.0, key="p_prev2")
            origin = l3.number_input("Delay at origin (min)", -60.0, 1440.0, 10.0, key="p_origin")
            if stop_index == 0:
                st.warning("Stop 0 is the origin, so there are no earlier delays. Live inputs are ignored.")
        else:
            st.markdown('<div class="note">Pre-departure mode: only schedule, season and history are used.</div>',
                        unsafe_allow_html=True)

    ctx = dict(train_number=tn, train_name=tname, train_type=ttype, station_code=station,
               stop_index=int(stop_index), total_stops=int(total_stops), distance_km=dist,
               sched_hour=sched_time.hour, date=run_date, fog=fog, avg7=avg7)

    # Build a history list from the three live numbers the user gave us
    if live and stop_index > 0:
        known = [origin, prev2, prev] if stop_index >= 2 else [prev]
        hist = [float(x) for x in known if not np.isnan(x)]
        row = build_row(ctx, hist)
        # With only 3 delays we can't know the exact running stats - use what we have
        row["previous_station_delay_min"] = prev
        row["delay_2_stops_back_min"] = prev2 if stop_index >= 2 else np.nan
        row["delay_change_last_stop_min"] = prev - prev2 if stop_index >= 2 else np.nan
        row["origin_delay_min"] = origin
        row["max_delay_so_far_min"] = max(x for x in [origin, prev2, prev] if not np.isnan(x))
        row["avg_delay_so_far_min"] = float(np.nanmean([origin, prev2, prev]))
    else:
        row = build_row(ctx, [])

    pred = float(predict_rows([row])[0])
    sched_dt = dt.datetime.combine(run_date, sched_time)

    with right:
        st.subheader("Prediction")
        departure_board(pred, sched_dt, f"Train {tn} at {station}")
        show_plot(gauge(pred, mae))
        if mae > 0:
            st.caption(f"Typical error of this model is about +/- {mae:.1f} min, so the likely range is "
                       f"{pred - mae:.0f} to {pred + mae:.0f} min.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted delay", fmt_duration(pred))
        m2.metric("Route progress", f"{row['route_progress_ratio'] * 100:.0f}%")
        m3.metric("Stops remaining", int(row["stops_remaining"]))
        with st.expander("Exact values sent to the model"):
            show_df(to_frame([row]).T.rename(columns={0: "value"}).astype(str))

# --------------------------------------------------------------------------
# TAB 2 - journey forecast
# --------------------------------------------------------------------------
with tab_journey:
    st.subheader("Forecast the rest of a journey")
    st.caption("Fill in the delays you have already seen. Leave a delay empty and the model forecasts it, "
               "using its own earlier forecasts as live data.")

    a, b = st.columns([1, 1], gap="large")
    with a:
        jtn, jname, jtype = pick_train("j")
    with b:
        jdate = st.date_input("Run date", dt.date.today(), key="j_date")
        javg7 = st.number_input("Train's avg delay, last 7 runs (min)", -60.0, 600.0, 20.0, key="j_avg7")
        jfog = st.toggle("Fog season (Dec - Jan)", value=season_default(jdate), key="j_fog")
        n = st.slider("Stops in route", 4, 40, 12, key="j_n")

    first_station = LEVELS["station_code"][0] if "station_code" in LEVELS else "NDLS"
    seed = pd.DataFrame(
        {
            "station_code": [first_station] * n,
            "distance_km": np.linspace(0, 800, n).round(0),
            "sched_hour": [int((6 + i * 1.2) % 24) for i in range(n)],
            "delay_min": [5.0, 12.0, 20.0] + [None] * (n - 3),
        }
    )
    colcfg = {
        "station_code": (
            st.column_config.SelectboxColumn("Station", options=LEVELS["station_code"])
            if "station_code" in LEVELS
            else st.column_config.TextColumn("Station")
        ),
        "distance_km": st.column_config.NumberColumn("Distance (km)", min_value=0),
        "sched_hour": st.column_config.NumberColumn("Scheduled hour", min_value=0, max_value=23, step=1),
        "delay_min": st.column_config.NumberColumn("Observed delay (min)", help="Leave empty to forecast"),
    }
    edited = show_editor(seed, column_config=colcfg, hide_index=True, key=f"j_table_{n}")

    if st.button("Forecast journey", type="primary"):
        hist, rows_out = [], []
        for i, r in edited.reset_index(drop=True).iterrows():
            c = dict(train_number=jtn, train_name=jname, train_type=jtype,
                     station_code=r["station_code"], stop_index=i, total_stops=n,
                     distance_km=float(r["distance_km"]), sched_hour=int(r["sched_hour"]),
                     date=jdate, fog=jfog, avg7=javg7)
            if pd.notna(r["delay_min"]):
                d, kind = float(r["delay_min"]), "Observed"
            else:
                d, kind = float(predict_rows([build_row(c, hist)])[0]), "Forecast"
            hist.append(d)
            rows_out.append({"stop": i, "station": r["station_code"], "distance_km": r["distance_km"],
                             "delay_min": round(d, 1), "type": kind})
        res = pd.DataFrame(rows_out)
        st.session_state["journey_result"] = res

    res = st.session_state.get("journey_result")
    if res is not None and len(res):
        fig = go.Figure()
        obs = res[res["type"] == "Observed"]
        fc = res[res["type"] == "Forecast"]
        fig.add_trace(go.Scatter(x=obs["stop"], y=obs["delay_min"], mode="lines+markers",
                                 name="Observed", line=dict(color=NAVY, width=3)))
        if len(fc):
            join = pd.concat([obs.tail(1), fc]) if len(obs) else fc
            fig.add_trace(go.Scatter(x=join["stop"], y=join["delay_min"], mode="lines+markers",
                                     name="Forecast", line=dict(color=SIGNAL_AMBER, width=3, dash="dash")))
        fig.add_hline(y=0, line_color="#999", line_width=1)
        fig.update_layout(height=360, xaxis_title="Stop number", yaxis_title="Delay (min)",
                          margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
        show_plot(fig)
        last = res.iloc[-1]
        k1, k2, k3 = st.columns(3)
        k1.metric("Delay at final stop", fmt_duration(last["delay_min"]))
        k2.metric("Worst delay on route", fmt_duration(res["delay_min"].max()))
        k3.metric("Forecast stops", int((res["type"] == "Forecast").sum()))
        show_df(res)
        st.download_button("Download forecast (CSV)", res.to_csv(index=False).encode(),
                           "journey_forecast.csv", "text/csv")

# --------------------------------------------------------------------------
# TAB 3 - what-if
# --------------------------------------------------------------------------
with tab_what:
    st.subheader("What if the delay at the previous stop was different?")
    st.caption("Uses the trip from the first tab. Only the delay at the previous stop changes.")
    if stop_index == 0:
        st.info("Pick a stop after the origin in the first tab to run this.")
    else:
        xs = np.arange(-10, 301, 5)
        rows = []
        for x in xs:
            r = dict(row)
            r["previous_station_delay_min"] = float(x)
            if not np.isnan(r["delay_2_stops_back_min"]):
                r["delay_change_last_stop_min"] = float(x) - r["delay_2_stops_back_min"]
            vals = [v for v in [r["origin_delay_min"], r["delay_2_stops_back_min"], float(x)]
                    if not np.isnan(v)]
            r["max_delay_so_far_min"] = max(vals)
            r["avg_delay_so_far_min"] = float(np.mean(vals))
            rows.append(r)
        ys = predict_rows(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=NAVY, width=3), name="Predicted"))
        fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color="#999", dash="dot"),
                                 name="Same as previous stop"))
        fig.update_layout(height=380, xaxis_title="Delay at previous stop (min)",
                          yaxis_title="Predicted delay here (min)", margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.1))
        show_plot(fig)
        st.markdown(
            '<div class="note">If the blue line sits below the dotted line, the model expects the train to '
            "recover some time. Above the dotted line means it expects the delay to grow.</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Fog season vs. normal season")
        r_fog, r_ok = dict(row), dict(row)
        r_fog["is_fog_season"], r_ok["is_fog_season"] = 1, 0
        p_fog, p_ok = predict_rows([r_fog, r_ok])
        f1, f2, f3 = st.columns(3)
        f1.metric("Fog season", fmt_duration(p_fog))
        f2.metric("Normal season", fmt_duration(p_ok))
        f3.metric("Difference", fmt_duration(p_fog - p_ok))

# --------------------------------------------------------------------------
# TAB 4 - batch
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Predict many rows from a CSV")
    st.caption("Upload a file with the same column names the model was trained on. "
               "Missing columns are treated as unknown. If the file contains delay_minutes, "
               "accuracy is calculated too.")
    tmpl = pd.DataFrame(columns=FEATURES)
    st.download_button("Download empty template", tmpl.to_csv(index=False).encode(),
                       "batch_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv", key="batch_csv")
    if up is not None:
        raw = pd.read_csv(up, dtype={"train_number": str})
        if "train_number" in raw.columns:
            raw["train_number"] = raw["train_number"].astype(str).str.zfill(5)
        missing = [c for c in FEATURES if c not in raw.columns]
        if missing:
            st.warning(f"{len(missing)} model columns are missing and will be treated as unknown: "
                       + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
        out = raw.copy()
        out["predicted_delay_min"] = predict_rows(raw.to_dict("records")).round(2)
        show_df(out.head(500))
        st.download_button("Download predictions", out.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
        if "delay_minutes" in out.columns:
            err = (out["delay_minutes"] - out["predicted_delay_min"]).abs()
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("MAE", f"{err.mean():.2f} min")
            q2.metric("Median error", f"{err.median():.2f} min")
            q3.metric("Within 10 min", f"{(err <= 10).mean() * 100:.1f}%")
            q4.metric("Rows", f"{len(out):,}")
            samp = out.sample(min(5000, len(out)), random_state=1)
            fig = px.scatter(samp, x="delay_minutes", y="predicted_delay_min", opacity=0.35,
                             labels={"delay_minutes": "Actual (min)", "predicted_delay_min": "Predicted (min)"})
            lim = [float(samp["delay_minutes"].min()), float(samp["delay_minutes"].max())]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=SIGNAL_RED, dash="dash"),
                                     name="Perfect"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            show_plot(fig)

# --------------------------------------------------------------------------
# TAB 5 - model insights
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("What the model relies on")
    if IMPORTANCE is None:
        st.info("Feature importance is not available for this model object.")
    else:
        top = IMPORTANCE.head(15).sort_values("gain")
        top = top.assign(share=(top["gain"] / IMPORTANCE["gain"].sum() * 100).round(1))
        fig = px.bar(top, x="share", y="feature", orientation="h",
                     labels={"share": "Share of total gain (%)", "feature": ""},
                     color_discrete_sequence=[NAVY])
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
        show_plot(fig)
        st.caption("Gain shows how much each feature improved the model's splits. "
                   "It describes the model, not cause and effect.")

    st.subheader("Reading the results")
    st.markdown(
        """
- **Live mode is far stronger than pre-departure mode.** Delay at the previous stop is the strongest signal, so a
  train that is already late will almost always still be late at the next stop.
- **Rare, very long delays are hard to predict.** The model sees few examples of 6+ hour delays, so it tends to
  predict smaller values for them.
- **Only trains and stations seen in training are supported.** Unknown codes become "unknown" for the model.
"""
    )
