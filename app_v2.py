"""
Uttar Rail Pulse - Northern Railway train delay predictor (multi-page Streamlit app)

Run:
    pip install -r requirements.txt
    streamlit run app_v2.py

Needs (from your notebook):
    model/train_delay_model.pkl
    model/feature_columns.pkl
    model/categorical_columns.pkl

Strongly recommended (makes inputs much simpler and predictions more accurate):
    railway_data/combined_schedule.csv   -> route, distance, scheduled time filled automatically
    railway_data/train_details.csv       -> train name and type filled automatically
You can also upload both from the sidebar.
"""

import datetime as dt
import io
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_NAME = "Uttar Rail Pulse"
APP_TAGLINE = "Delay predictions for Northern Railway (NR) trains"
AUTHOR = "Shreshtha Pandey"

st.set_page_config(page_title=f"{APP_NAME} | Northern Railway delays", page_icon="🚆", layout="wide")

NAVY, AMBER, GREEN, RED, STEEL = "#14325C", "#F2A900", "#2E8B57", "#C8372D", "#E9EEF3"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Barlow', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', sans-serif !important; }}
.block-container {{ padding-top: 4.2rem; max-width: 1200px; }}
.hero-title {{ font-family:'Barlow Condensed'; font-size:46px; font-weight:700; color:inherit; margin:0; line-height:1.1; }}
.hero-sub {{ color:inherit; opacity:.75; margin:4px 0 20px 0; font-size:16px; }}
.step {{ font-family:'Barlow Condensed'; font-size:24px; font-weight:700; color:inherit;
        border-bottom:2px solid {STEEL}; padding-bottom:4px; margin:6px 0 12px 0; }}
.board {{ background:{NAVY}; color:#fff; border-radius:6px; padding:20px 24px; border-bottom:6px solid {AMBER}; }}
.board .big {{ font-family:'Barlow Condensed'; font-weight:700; font-size:78px; line-height:1; color:{AMBER}; }}
.board .big small {{ font-size:26px; color:#cfd9e8; margin-left:6px; font-weight:500; }}
.board .status {{ font-family:'Barlow Condensed'; font-size:30px; font-weight:600; margin-top:6px; }}
.board .line {{ color:#cfd9e8; font-size:15px; margin-top:3px; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:8px; }}
.note {{ background:{STEEL}; color:#14325C; border-left:4px solid {NAVY}; padding:10px 14px; border-radius:4px; font-size:14px; }}
.brand {{ background:{NAVY}; border-bottom:6px solid {AMBER}; border-radius:6px; padding:16px 24px;
          display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; margin-bottom:22px; }}
.brand .name {{ font-family:'Barlow Condensed'; font-size:48px; font-weight:700; color:#FFFFFF; line-height:1; letter-spacing:.01em; }}
.brand .tag {{ color:{AMBER}; font-size:17px; font-weight:600; }}
.footer {{ margin-top:48px; padding:14px 0; border-top:1px solid #d5dde6; text-align:center; color:inherit; opacity:.8; font-size:14px; }}
div[data-testid="stMetricValue"] {{ font-family:'Barlow Condensed'; font-weight:700; color:inherit; }}
</style>
""",
    unsafe_allow_html=True,
)


def hero(title, sub):
    st.markdown(f'<p class="hero-title">{title}</p><p class="hero-sub">{sub}</p>', unsafe_allow_html=True)


def step(text):
    st.markdown(f'<div class="step">{text}</div>', unsafe_allow_html=True)


# ---- Streamlit-version-safe wrappers --------------------------------------
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


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "model")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "train_delay_model.pkl"))
    features = list(joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl")))
    cat_cols = list(joblib.load(os.path.join(MODEL_DIR, "categorical_columns.pkl")))
    cat_in_order = [c for c in features if c in cat_cols]
    levels = {}
    try:
        pc = model.booster_.pandas_categorical or []
        if len(pc) == len(cat_in_order):
            levels = {c: [str(x) for x in lv] for c, lv in zip(cat_in_order, pc)}
    except Exception:
        pass
    imp = None
    try:
        imp = pd.DataFrame({
            "feature": model.booster_.feature_name(),
            "gain": model.booster_.feature_importance("gain"),
        }).sort_values("gain", ascending=False)
    except Exception:
        pass
    return model, features, cat_cols, levels, imp


try:
    MODEL, FEATURES, CAT_COLS, LEVELS, IMPORTANCE = load_artifacts()
except Exception as e:
    hero(APP_NAME, "")
    st.error(
        f"Could not load the model from the `{MODEL_DIR}/` folder. Put `train_delay_model.pkl`, "
        f"`feature_columns.pkl` and `categorical_columns.pkl` there and restart.\n\nDetails: {e}"
    )
    st.stop()


# --------------------------------------------------------------------------
# Optional data: schedule + train details
# --------------------------------------------------------------------------
def _clean_ref(d):
    d["train_no"] = d["train_no"].astype(str).str.zfill(5)
    keep = [c for c in ["train_no", "train_name", "type_code"] if c in d.columns]
    return d[keep].drop_duplicates("train_no")


def _clean_sched(d):
    need = {"train_no", "station_no", "station_name"}
    if not need.issubset(d.columns):
        return None
    d["train_no"] = d["train_no"].astype(str).str.zfill(5)
    d["station_no"] = pd.to_numeric(d["station_no"], errors="coerce")
    for c in ["distance_from_origin", "arrival_day"]:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    if "arrival_time" not in d.columns:
        d["arrival_time"] = None
    d = d.drop_duplicates(["train_no", "station_name"])
    return d[["train_no", "station_no", "station_name", "distance_from_origin", "arrival_day", "arrival_time"]]


@st.cache_data(show_spinner=False)
def ref_from_bytes(raw: bytes):
    return _clean_ref(pd.read_csv(io.BytesIO(raw), dtype={"train_no": str}))


@st.cache_data(show_spinner=False)
def ref_from_path(path: str, mtime: float):
    return _clean_ref(pd.read_csv(path, dtype={"train_no": str}))


@st.cache_data(show_spinner="Reading schedule...")
def sched_from_bytes(raw: bytes):
    return _clean_sched(pd.read_csv(io.BytesIO(raw), dtype={"train_no": str}))


@st.cache_data(show_spinner="Reading schedule...")
def sched_from_path(path: str, mtime: float):
    return _clean_sched(pd.read_csv(path, dtype={"train_no": str}))


def find_file(names):
    for p in names:
        if os.path.exists(p):
            return p
    return None


with st.sidebar:
    st.markdown("### Data files")
    st.caption("Optional, but they remove most of the typing.")
    up_sched = st.file_uploader("combined_schedule.csv", type="csv")
    up_ref = st.file_uploader("train_details.csv", type="csv")

if up_ref is not None:
    REF = ref_from_bytes(up_ref.getvalue())
else:
    _p = find_file(["railway_data/train_details.csv", "train_details.csv"])
    REF = ref_from_path(_p, os.path.getmtime(_p)) if _p else None

if up_sched is not None:
    SCHED = sched_from_bytes(up_sched.getvalue())
else:
    _p = find_file(["railway_data/combined_schedule.csv", "combined_schedule.csv"])
    SCHED = sched_from_path(_p, os.path.getmtime(_p)) if _p else None

REF_MAP = {}
if REF is not None:
    for r in REF.itertuples(index=False):
        REF_MAP[r.train_no] = (getattr(r, "train_name", None), getattr(r, "type_code", None))

with st.sidebar:
    if SCHED is not None:
        st.success("Schedule loaded: route details are automatic")
    else:
        st.warning("No schedule file: you will enter route details by hand")

    if REF is not None:
        st.success("Train names loaded")
    else:
        st.warning("No train_details.csv: accuracy may drop (name and type unknown)")


# --------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------
def to_frame(rows):
    df = pd.DataFrame(rows)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FEATURES].copy()
    for c in FEATURES:
        if c in CAT_COLS:
            vals = df[c].astype(object).where(df[c].notna(), None)
            vals = vals.map(lambda v: None if v is None else str(v))
            df[c] = pd.Categorical(vals, categories=LEVELS[c]) if c in LEVELS else pd.Categorical(vals)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def predict_rows(rows):
    return MODEL.predict(to_frame(rows))


def build_row(ctx, hist):
    """Model input for one stop. hist = known delays at earlier stops, in order."""
    i, total = ctx["stop_index"], ctx["total_stops"]
    prev = hist[-1] if len(hist) >= 1 else np.nan
    prev2 = hist[-2] if len(hist) >= 2 else np.nan
    return {
        "train_number": ctx["train_number"], "train_name": ctx.get("train_name"),
        "train_type": ctx.get("train_type"), "station_code": ctx.get("station_code"),
        "stop_sequence_no": i + 1, "stop_index": i, "total_stops_in_route": total,
        "stops_remaining": total - i - 1, "route_progress_ratio": i / max(total - 1, 1),
        "distance_from_origin_km": ctx["distance_km"], "scheduled_arrival_hour": ctx["sched_hour"],
        "journey_day_of_arrival": ctx.get("journey_day", 1),
        "day_of_week": ctx["date"].weekday(), "month": ctx["date"].month,
        "is_fog_season": int(ctx["date"].month in (12, 1)), "has_schedule_match": 1,
        "avg_delay_last_7_runs_min": ctx["avg7"],
        "previous_station_delay_min": prev, "delay_2_stops_back_min": prev2,
        "delay_change_last_stop_min": prev - prev2 if len(hist) >= 2 else np.nan,
        "origin_delay_min": hist[0] if (i > 0 and hist) else np.nan,
        "max_delay_so_far_min": max(hist) if (i > 0 and hist) else np.nan,
        "avg_delay_so_far_min": (sum(hist) / i) if (i > 0 and hist) else np.nan,
    }


def live_row(ctx, prev=None, prev2=None, origin=None):
    """Row when the user only knows a few live delays (prev required)."""
    row = build_row(ctx, [])
    i = ctx["stop_index"]
    if i == 0 or prev is None:
        return row
    if i == 1:
        origin, prev2 = prev, None
    known = [float(x) for x in (origin, prev2, prev) if x is not None]
    row["previous_station_delay_min"] = float(prev)
    if prev2 is not None:
        row["delay_2_stops_back_min"] = float(prev2)
        row["delay_change_last_stop_min"] = float(prev) - float(prev2)
    if origin is not None:
        row["origin_delay_min"] = float(origin)
    row["max_delay_so_far_min"] = max(known)
    row["avg_delay_so_far_min"] = float(np.mean(known))
    return row


def status_of(d):
    if d <= -1:
        return "Running early", GREEN
    if d <= 5:
        return "On time", GREEN
    if d <= 15:
        return "Minor delay", AMBER
    if d <= 60:
        return "Delayed", AMBER
    if d <= 180:
        return "Heavily delayed", RED
    return "Severely delayed", RED


def fmt_dur(m):
    m = int(round(m))
    sign, m = ("-" if m < 0 else ""), abs(m)
    h, mm = divmod(m, 60)
    return f"{sign}{h} h {mm:02d} min" if h else f"{sign}{mm} min"


def board(delay, sched_dt, title):
    label, color = status_of(delay)
    eta = sched_dt + dt.timedelta(minutes=float(delay))
    sign = "+" if delay > 0 else ""
    st.markdown(
        f"""<div class="board">
<div class="big">{sign}{delay:.0f}<small>min</small></div>
<div class="status"><span class="dot" style="background:{color}"></span>{label}</div>
<div class="line">{title}</div>
<div class="line">Scheduled {sched_dt:%d %b, %H:%M} &nbsp;|&nbsp; Expected {eta:%d %b, %H:%M}</div>
</div>""",
        unsafe_allow_html=True,
    )


def gauge(delay):
    top = max(120, abs(delay) * 1.4)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(delay),
        number={"suffix": " min", "font": {"family": "Barlow Condensed", "size": 36}},
        gauge={"axis": {"range": [-30, top]}, "bar": {"color": NAVY, "thickness": 0.3},
               "steps": [{"range": [-30, 5], "color": "#cfe8d9"}, {"range": [5, 60], "color": "#fbe7b0"},
                         {"range": [60, top], "color": "#f2c6c1"}]}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=20, b=0))
    return fig


# ---- pickers / route -------------------------------------------------------
def train_label(no):
    nm = REF_MAP.get(str(no), (None, None))[0]
    return f"{no} - {nm}" if nm else str(no)


def train_picker(key, default=None):
    if "train_number" in LEVELS:
        opts = LEVELS["train_number"]
        idx = opts.index(default) if default in opts else 0
        return st.selectbox("Train", opts, index=idx, format_func=train_label, key=key,
                            help="Type to search by number or name.")
    return st.text_input("Train number", default or "12001", key=key).strip().zfill(5)


def get_route(tn):
    if SCHED is None:
        return None
    r = SCHED[SCHED["train_no"] == str(tn)]
    if r.empty:
        return None
    return r.sort_values("station_no").reset_index(drop=True)


def parse_time(s):
    try:
        return dt.datetime.strptime(str(s)[:5], "%H:%M").time()
    except Exception:
        return dt.time(9, 0)


def train_meta(tn):
    nm, ty = REF_MAP.get(str(tn), (None, None))
    return nm, (str(ty) if ty is not None and not pd.isna(ty) else None)


def sched_datetime(date, sched_time, day):
    day = 1 if (day is None or pd.isna(day)) else int(day)
    return dt.datetime.combine(date, sched_time) + dt.timedelta(days=max(day - 1, 0))


# ==========================================================================
# PAGE 1 - Predict
# ==========================================================================
def page_predict():
    hero("Predict a delay", "Pick a Northern Railway train and a stop. Say whether it is already running. The rest is automatic.")
    prev_trip = st.session_state.get("trip", {})
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        step("1. Train and stop")
        tn = train_picker("pf_train", prev_trip.get("train_number"))
        route = get_route(tn)
        c1, c2 = st.columns([1, 1.3])
        run_date = c1.date_input("Travel date", dt.date.today(), key="pf_date")

        if route is not None:
            def lab(i):
                r = route.iloc[i]
                dist = r.distance_from_origin
                return f"{i + 1}. {r.station_name}" + (f" ({dist:.0f} km)" if pd.notna(dist) else "")
            stop_index = c2.selectbox("Stop", range(len(route)), index=len(route) // 2,
                                      format_func=lab, key=f"pf_stop_{tn}")
            r = route.iloc[stop_index]
            station, total = r.station_name, len(route)
            dist = float(r.distance_from_origin) if pd.notna(r.distance_from_origin) else 0.0
            sched_time = parse_time(r.arrival_time)
            day = r.arrival_day
            st.caption(f"Scheduled arrival {sched_time:%H:%M}. This stop is {stop_index + 1} of {total}.")
        else:
            if "station_code" in LEVELS:
                station = c2.selectbox("Station", LEVELS["station_code"], key="pf_station")
            else:
                station = c2.text_input("Station code", "NDLS", key="pf_station").upper()
            st.caption("No schedule file loaded, so please fill in the route details.")
            d1, d2, d3 = st.columns(3)
            total = int(d1.number_input("Stops on route", 2, 200, 30, key="pf_total"))
            stop_index = int(d2.number_input("Stop number (0 = start)", 0, total - 1, min(10, total - 1), key="pf_idx"))
            dist = float(d3.number_input("Km from start", 0.0, 4000.0, 450.0, step=10.0, key="pf_dist"))
            sched_time = st.time_input("Scheduled arrival", dt.time(9, 0), key="pf_time")
            day = 1

        step("2. Is the train already running?")
        running = st.radio("Status", ["Not started yet", "Already running"], horizontal=True,
                           label_visibility="collapsed", key="pf_status") == "Already running"
        prev = prev2 = origin = None
        if running and stop_index == 0:
            st.info("This is the first stop, so there are no earlier delays to use.")
        elif running:
            prev = st.slider("Delay at the last stop it passed (minutes)", -30, 600, 15, key="pf_prev",
                             help="The single most useful number for the prediction.")
        else:
            st.markdown('<div class="note">Using schedule, date and the train\'s history only. '
                        "Predictions are rougher before the train starts.</div>", unsafe_allow_html=True)

        with st.expander("Fine-tune (optional)"):
            avg7 = st.slider("Train's usual delay over its last 7 runs (minutes)", -10, 240, 15, key="pf_avg7")
            if running and stop_index >= 1:
                o1, o2 = st.columns(2)
                origin = o1.number_input("Delay at start station", value=None, placeholder="unknown", key="pf_origin")
                if stop_index >= 2:
                    prev2 = o2.number_input("Delay 2 stops back", value=None, placeholder="unknown", key="pf_prev2")

    nm, ty = train_meta(tn)
    ctx = dict(train_number=tn, train_name=nm, train_type=ty, station_code=station,
               stop_index=int(stop_index), total_stops=int(total), distance_km=dist,
               sched_hour=sched_time.hour, journey_day=1 if pd.isna(day) else int(day),
               date=run_date, avg7=float(avg7))
    row = live_row(ctx, prev, prev2, origin) if prev is not None else build_row(ctx, [])
    pred = float(predict_rows([row])[0])
    sched_dt = sched_datetime(run_date, sched_time, day)

    st.session_state["trip"] = dict(train_number=tn, ctx=ctx, row=row, pred=pred, sched_dt=sched_dt,
                                    running=prev is not None)

    with right:
        step("3. Prediction")
        board(pred, sched_dt, f"{train_label(tn)} at {station}")
        show_plot(gauge(pred))
        m1, m2, m3 = st.columns(3)
        m1.metric("Delay", fmt_dur(pred))
        m2.metric("Route done", f"{row['route_progress_ratio'] * 100:.0f}%")
        m3.metric("Stops left", int(row["stops_remaining"]))
        if st.button("Save to history"):
            st.session_state.setdefault("history", []).append(
                {"Train": train_label(tn), "Stop": station, "Date": str(run_date),
                 "Mode": "Live" if prev is not None else "Pre-departure", "Predicted delay (min)": round(pred, 1)})
            st.toast("Saved")
        hist = st.session_state.get("history", [])
        if hist:
            with st.expander(f"Saved predictions ({len(hist)})"):
                show_df(pd.DataFrame(hist))


# ==========================================================================
# PAGE 2 - Journey forecast
# ==========================================================================
def page_journey():
    hero("Journey forecast", "Type the delays you have seen so far. The model forecasts the rest of the route.")
    trip = st.session_state.get("trip", {})
    c1, c2 = st.columns(2)
    with c1:
        tn = train_picker("jf_train", trip.get("train_number"))
    run_date = c2.date_input("Travel date", trip.get("ctx", {}).get("date", dt.date.today()), key="jf_date")

    txt = st.text_input("Delays so far, in minutes, one per stop in order", "5, 12, 20", key="jf_txt",
                        help="Example: 0, 8, 15, 22 means 0 min at stop 1, 8 min at stop 2, and so on.")
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", txt)]
    route = get_route(tn)
    if route is not None:
        n = len(route)
    else:
        n = st.slider("Stops on route", max(len(nums) + 1, 4), 60, max(len(nums) + 6, 12), key="jf_n")
        st.caption("No schedule file loaded: using evenly spaced stops.")

    with st.expander("Fine-tune (optional)"):
        avg7 = st.slider("Train's usual delay over its last 7 runs (minutes)", -10, 240, 15, key="jf_avg7")

    if not nums:
        st.info("Enter at least one delay to start.")
        return
    if len(nums) >= n:
        st.warning(f"This route has {n} stops, so only the first {n} delays are used.")
        nums = nums[:n]

    nm, ty = train_meta(tn)
    hist, out = [], []
    for i in range(n):
        if route is not None:
            r = route.iloc[i]
            station = r.station_name
            dist = float(r.distance_from_origin) if pd.notna(r.distance_from_origin) else 0.0
            hour = parse_time(r.arrival_time).hour
            day = 1 if pd.isna(r.arrival_day) else int(r.arrival_day)
        else:
            station, dist, hour, day = None, float(np.linspace(0, 800, n)[i]), int((6 + i * 1.2) % 24), 1
        ctx = dict(train_number=tn, train_name=nm, train_type=ty, station_code=station, stop_index=i,
                   total_stops=n, distance_km=dist, sched_hour=hour, journey_day=day,
                   date=run_date, avg7=float(avg7))
        if i < len(nums):
            d, kind = nums[i], "Seen"
        else:
            d, kind = float(predict_rows([build_row(ctx, hist)])[0]), "Forecast"
        hist.append(d)
        out.append({"Stop": i + 1, "Station": station or f"Stop {i + 1}", "Delay (min)": round(d, 1), "Type": kind})
    res = pd.DataFrame(out)

    seen, fc = res[res["Type"] == "Seen"], res[res["Type"] == "Forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seen["Station"], y=seen["Delay (min)"], mode="lines+markers", name="Seen",
                             line=dict(color=NAVY, width=3)))
    if len(fc):
        join = pd.concat([seen.tail(1), fc])
        fig.add_trace(go.Scatter(x=join["Station"], y=join["Delay (min)"], mode="lines+markers", name="Forecast",
                                 line=dict(color=AMBER, width=3, dash="dash")))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(height=360, yaxis_title="Delay (min)", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12), xaxis=dict(type="category"))
    show_plot(fig)

    k1, k2, k3 = st.columns(3)
    k1.metric("At final stop", fmt_dur(res.iloc[-1]["Delay (min)"]))
    k2.metric("Worst on route", fmt_dur(res["Delay (min)"].max()))
    k3.metric("Stops forecast", len(fc))
    with st.expander("See the table"):
        show_df(res)
        st.download_button("Download CSV", res.to_csv(index=False).encode(), "journey_forecast.csv", "text/csv")


# ==========================================================================
# PAGE 3 - What-if
# ==========================================================================
def page_whatif():
    hero("What if...", "Change one thing about your last prediction and see what the model does.")
    trip = st.session_state.get("trip")
    if not trip:
        st.info("Make a prediction first, then come back here.")
        st.page_link(PG_PREDICT, label="Go to Predict a delay", icon="🎯")
        return
    ctx, base = trip["ctx"], trip["row"]
    st.caption(f"Based on {train_label(trip['train_number'])} at {ctx['station_code']} on {ctx['date']:%d %b %Y}.")
    if ctx["stop_index"] == 0:
        st.info("The trip's stop is the first one, so there are no earlier delays to change. "
                "Pick a later stop on the Predict page.")
    else:
        default = base["previous_station_delay_min"]
        default = 15 if pd.isna(default) else int(round(default))
        x = st.slider("Delay at the last stop it passed (minutes)", -30, 600, int(np.clip(default, -30, 600)))

        def with_prev(v):
            r = dict(base)
            known = [t for t in (r["origin_delay_min"], r["delay_2_stops_back_min"]) if not pd.isna(t)] + [float(v)]
            r["previous_station_delay_min"] = float(v)
            if not pd.isna(r["delay_2_stops_back_min"]):
                r["delay_change_last_stop_min"] = float(v) - r["delay_2_stops_back_min"]
            if ctx["stop_index"] == 1:
                r["origin_delay_min"] = float(v)
            r["max_delay_so_far_min"] = max(known)
            r["avg_delay_so_far_min"] = float(np.mean(known))
            return r

        xs = np.arange(-30, 601, 10)
        ys = predict_rows([with_prev(v) for v in xs])
        here = float(predict_rows([with_prev(x)])[0])
        c1, c2 = st.columns([1.6, 1], gap="large")
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=NAVY, width=3), name="Predicted"))
            fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color="#999", dash="dot"),
                                     name="Same as last stop"))
            fig.add_trace(go.Scatter(x=[x], y=[here], mode="markers", marker=dict(size=14, color=AMBER,
                                     line=dict(color=NAVY, width=2)), name="Your value"))
            fig.update_layout(height=360, xaxis_title="Delay at last stop (min)", yaxis_title="Predicted delay here (min)",
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
            show_plot(fig)
        with c2:
            board(here, trip["sched_dt"], "With your changed value")
            st.caption("Below the dotted line: the model expects the train to recover time. "
                       "Above it: the delay is expected to grow.")

    st.markdown("#### Which day of the week is best?")
    day_rows = []
    for d in range(7):
        r = dict(base)
        r["day_of_week"] = d
        day_rows.append(r)
    dp = predict_rows(day_rows)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = px.bar(x=names, y=dp, labels={"x": "", "y": "Predicted delay (min)"})
    fig.update_traces(marker_color=[AMBER if i == int(np.argmin(dp)) else NAVY for i in range(7)])
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    show_plot(fig)
    st.caption(f"Lowest predicted delay: {names[int(np.argmin(dp))]} (highlighted). "
               "This isolates the weekday effect only.")

    st.markdown("#### Fog season vs normal season")
    rf, rn = dict(base), dict(base)
    rf["is_fog_season"], rn["is_fog_season"] = 1, 0
    pf, pn = predict_rows([rf, rn])
    a, b, c = st.columns(3)
    a.metric("Fog season", fmt_dur(pf))
    b.metric("Normal season", fmt_dur(pn))
    c.metric("Difference", fmt_dur(pf - pn))


# ==========================================================================
# PAGE 4 - Batch
# ==========================================================================
def page_batch():
    hero("Batch predictions", "Upload a CSV, get a prediction for every row.")
    st.download_button("Download an empty template", pd.DataFrame(columns=FEATURES).to_csv(index=False).encode(),
                       "batch_template.csv", "text/csv")
    up = st.file_uploader("Your CSV", type="csv", key="batch_csv")
    st.caption("Columns must use the model's names. Missing columns count as unknown. "
               "If the file has delay_minutes, accuracy is calculated too.")
    if up is None:
        return
    raw = pd.read_csv(up, dtype={"train_number": str})
    if "train_number" in raw.columns:
        raw["train_number"] = raw["train_number"].astype(str).str.zfill(5)
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        st.warning(f"{len(missing)} columns missing, treated as unknown: " + ", ".join(missing[:10])
                   + (" ..." if len(missing) > 10 else ""))
    out = raw.copy()
    out["predicted_delay_min"] = predict_rows(raw.to_dict("records")).round(2)
    show_df(out.head(500))
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
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
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=RED, dash="dash"), name="Perfect"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        show_plot(fig)


# ==========================================================================
# PAGE 5 - Model insights
# ==========================================================================
def page_insights():
    hero("How the model thinks", "What matters most, in plain language.")
    if IMPORTANCE is None:
        st.info("Feature importance is not available for this model object.")
    else:
        top = IMPORTANCE.head(12).sort_values("gain").copy()
        top["share"] = (top["gain"] / IMPORTANCE["gain"].sum() * 100).round(1)
        fig = px.bar(top, x="share", y="feature", orientation="h", color_discrete_sequence=[NAVY],
                     labels={"share": "Share of total importance (%)", "feature": ""})
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
        show_plot(fig)
    st.markdown(
        """
**Reading this**
- When a train is already late, the delay at its last stop is the strongest hint for the next one.
- Before departure the model only has schedule and history, so predictions are less precise.
- Very long delays (6+ hours) are rare in the training data, so the model tends to under-predict them.
- Only trains and stations seen in training are supported.
"""
    )
    with st.expander("All features the model uses"):
        st.write(FEATURES)



# ==========================================================================
# PAGE 6 - About the project
# ==========================================================================
def page_about():
    hero("About this project", f"{APP_NAME} predicts how late Northern Railway trains will run, stop by stop.")

    st.markdown("### What it does")
    st.write(
        "Indian trains often run late, and a delay at one station usually carries on to the next. "
        "This project learns from millions of past station records of Northern Railway (NR) trains "
        "and predicts the delay in minutes at any stop on a train's route. It works in two ways: "
        "before the train starts, using schedule and history only, and while it is running, "
        "using the delays already seen at earlier stops."
    )

    st.markdown("### Scope")
    st.markdown(
        """
- **Zone:** Northern Railway (NR) only. Trains were kept when at least 70% of their stops lie in the NR zone.
- **Period:** daily runs from February 2025 to early February 2026.
- **Target:** delay at each station, in minutes (negative means early).
- **Not covered:** trains of other zones, and stations or trains the model never saw in training.
"""
    )

    st.markdown("### How it was built")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Collect**\n\nStation-level delay records, train schedules, train details and station names.")
    c2.markdown("**2. Clean**\n\nRemoved missing delays and extreme values (kept -60 to 1440 min), "
                "stored as Parquet for fast reading, and matched train numbers to a 5-digit format.")
    c3.markdown("**3. Engineer**\n\nBuilt features for route position, season, weekday, recent delays and "
                "delays at earlier stops of the same run.")
    c4.markdown("**4. Model**\n\nTrained LightGBM on past dates and tested it on later, unseen dates.")

    st.markdown("### Data cleaning and checks")
    st.markdown(
        """
- Rows with no delay value (about 4.9%) were removed because the model needs a real answer to learn from.
- Delays below -60 minutes or above 24 hours were treated as data errors and dropped.
- Duplicate station records were checked for and none were found.
- Train and station averages were rebuilt using only earlier dates, so the model never sees the future.
- The split is by time, not random: training up to 31 Dec 2025, validation 1 to 15 Jan 2026, and testing 16 Jan to 7 Feb 2026.
"""
    )

    st.markdown("### Features the model uses")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            """
**Live features (while the train is running)**
- Delay at the previous stop
- Delay two stops back and the change between them
- Delay at the starting station
- Highest and average delay so far on this run
"""
        )
    with f2:
        st.markdown(
            """
**Schedule and history features**
- Train number, name and type; station
- Distance from origin, stop position, stops remaining, route progress
- Scheduled hour, weekday, month, fog season (Dec-Jan)
- Train's average delay over its last 7 runs
"""
        )

    st.markdown("### Weather experiment")
    st.write(
        "Morning humidity, wind, temperature and rain from Open-Meteo were tested to capture fog. "
        "They did not give a reliable improvement on the validation period, so the final model does not use them."
    )

    st.markdown("### Model")
    st.write(
        "LightGBM gradient boosting regressor with early stopping on the validation set. "
        "Exact scores (MAE, RMSE, R-squared) are in the training notebook, and the Batch page "
        "can recompute them on any file that has actual delays."
    )

    st.markdown("### Pages in this app")
    st.markdown(
        """
- **Predict a delay:** pick a train and a stop and get the expected delay and arrival time.
- **Journey forecast:** enter the delays seen so far and forecast the rest of the route.
- **What if...:** change the last-stop delay, weekday or season and see the effect.
- **Batch predictions:** upload a CSV to predict many rows and check accuracy.
- **How the model thinks:** which features matter most.
"""
    )

    st.markdown("### Limitations")
    st.markdown(
        """
- Predictions are statistical estimates, not official railway information. Do not rely on them for safety or ticketing decisions.
- Before a train starts, predictions are rougher because no live delay is known.
- Very long delays (6+ hours) are rare in the data, so the model tends to under-predict them.
- The model reflects 2025 to early 2026 patterns and should be retrained as new data arrives.
"""
    )

    st.markdown("### Built with")
    st.write("Python, pandas, DuckDB, LightGBM, scikit-learn, SHAP, Plotly and Streamlit.")
    st.caption(f"Designed and built by {AUTHOR}.")


def render_footer():
    st.markdown(
        f'<div class="footer">{APP_NAME} &nbsp;|&nbsp; Northern Railway delay predictor &nbsp;|&nbsp; '
        f'Made by <b>{AUTHOR}</b> &nbsp;|&nbsp; Estimates only, not official railway data</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
PG_PREDICT = st.Page(page_predict, title="Predict a delay", icon="🎯", default=True)
PG_JOURNEY = st.Page(page_journey, title="Journey forecast", icon="🗺️")
PG_WHATIF = st.Page(page_whatif, title="What if...", icon="🔀")
PG_BATCH = st.Page(page_batch, title="Batch predictions", icon="📄")
PG_INSIGHTS = st.Page(page_insights, title="How the model thinks", icon="🧠")
PG_ABOUT = st.Page(page_about, title="About the project", icon="ℹ️")

st.markdown(
    f'<div class="brand"><span class="name">🚆 {APP_NAME}</span><span class="tag">{APP_TAGLINE}</span></div>',
    unsafe_allow_html=True,
)
st.navigation([PG_PREDICT, PG_JOURNEY, PG_WHATIF, PG_BATCH, PG_INSIGHTS, PG_ABOUT]).run()
render_footer()
