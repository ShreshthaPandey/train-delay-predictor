# 🚆 Uttar Rail Pulse: Northern Railway Train Delay Predictor

A machine-learning project that predicts how many minutes late a **Northern Railway (NR)** train will be at any stop on its route, plus a multi-page **Streamlit** app to explore the predictions.

It works in two ways:

- **Live mode:** the train is already running and you know the delay at the previous stop(s).
- **Pre-departure mode:** the train has not started, so only schedule, date and history are used (less accurate).

> ⚠️ Estimates only. This is a student/portfolio project and is **not** official railway information. Do not use it for safety or ticketing decisions.

---

## Results

Time-based split (the model never sees the future):

| Split | Dates | Rows | MAE (min) | RMSE (min) | R² |
|---|---|---|---|---|---|
| Train | 8 Feb 2025 – 31 Dec 2025 | 3,122,016 | – | – | – |
| Validation | 1 – 15 Jan 2026 | 130,488 | 5.90 | 18.28 | 0.901 |
| **Test** | 16 Jan – 7 Feb 2026 | 199,630 | **5.74** | **16.58** | **0.894** |

Share of test predictions within a given error:

| ≤ 5 min | ≤ 10 min | ≤ 15 min | ≤ 30 min |
|---|---|---|---|
| 70.3% | 85.8% | 91.8% | 97.5% |

Test MAE by actual delay size:

| Actual delay | Rows | MAE (min) |
|---|---|---|
| Early / on time (≤ 0) | 24,282 | 6.63 |
| 0 – 15 min | 58,499 | 3.59 |
| 15 – 60 min | 77,562 | 5.00 |
| 60 – 180 min | 34,818 | 8.48 |
| 180 – 360 min | 4,112 | 14.25 |
| 360+ min | 357 | 93.11 |

**How to read these numbers:** they are for *live* prediction, where the true delay at the previous stop is known. Delays carry over from stop to stop (correlation about 0.95), so a naive "same as last stop" guess is already strong (validation MAE 7.17). The model improves on it, but most of the R² comes from this carry-over effect. Pre-departure prediction is much harder: in a separate experiment a schedule-only model reached about 24.8 min MAE on validation, against 28.6 min for a per-train median baseline.

---

## Dataset and scope

- **Source:** station-level Indian Railways delay archive (`archive (19).zip`). *Add the dataset link and licence here.*
- **Raw size:** 38,428,703 station records, 8 Feb 2025 – 7 Feb 2026.
- **Zone:** only trains with **at least 70% of stops in the NR zone** (zone taken from `station_full_names.csv`).
- **Final modelling set:** about **3.45 million rows**, 684 trains, 1,091 stations.
- **Target:** `delay_minutes` at each station (negative means early).

### Cleaning steps

1. Train numbers converted to 5-digit text (`961` → `00961`) so files join correctly.
2. Rows with no delay value removed (1,874,271 rows, 4.9%).
3. Delays outside **-60 to 1440 minutes** treated as errors and removed.
4. Duplicate records checked (none found); data stored as Parquet for fast reading with DuckDB.
5. Schedule joined on train, station number and station name; a `has_schedule_match` flag marks rows with no schedule (about 12%).

---

## Method

**1. Features**

| Group | Features |
|---|---|
| Live (from earlier stops of the same run) | `previous_station_delay_min`, `delay_2_stops_back_min`, `delay_change_last_stop_min`, `origin_delay_min`, `max_delay_so_far_min`, `avg_delay_so_far_min` |
| History | `avg_delay_last_7_runs_min`, `train_avg_delay_min`, `station_avg_delay_min` |
| Route and schedule | `stop_sequence_no`, `stop_index`, `total_stops_in_route`, `stops_remaining`, `route_progress_ratio`, `distance_from_origin_km`, `scheduled_arrival_hour`, `journey_day_of_arrival`, `has_schedule_match` |
| Calendar | `day_of_week`, `month`, `is_fog_season` (Dec–Jan) |
| Categorical | `train_number`, `train_name`, `train_type`, `station_code` |

**2. No future leakage**

- Delay-history features use only earlier stops of the same run.
- `avg_delay_last_7_runs_min` uses only earlier runs of the train (`shift(1)` then a rolling window).
- `train_avg_delay_min` and `station_avg_delay_min` are built from earlier dates only and joined with `merge_asof(allow_exact_matches=False)`.
- Splits are chronological and checked with assertions that the date ranges do not overlap.

**3. Model:** LightGBM regressor (63 leaves, learning rate 0.05, early stopping on validation) with native categorical handling.

**4. Weather experiment:** morning humidity, wind, temperature and rain from Open-Meteo were tested to capture fog. They gave no reliable gain on validation, so the final model does not use them.

---

## Repository structure

```
train-delay-predictor/
├── app_v2.py                    # Streamlit app (6 pages)
├── train_delay_model.ipynb      # data prep, features, training, evaluation
├── requirements.txt
├── README.md
├── model/                       # created by the notebook
│   ├── train_delay_model.pkl
│   ├── feature_columns.pkl
│   └── categorical_columns.pkl
└── railway_data/                # NOT committed (large); see "Data"
    ├── combined_schedule.csv
    ├── train_details.csv
    ├── station_full_names.csv
    └── combined_delay.csv
```

---

## Setup and run

```bash
# 1. clone and enter the folder
git clone <your-repo-url>
cd train-delay-predictor

# 2. create an environment and install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. run the app
streamlit run app_v2.py
```

The app needs the three files in `model/`. For easy inputs, also place `railway_data/combined_schedule.csv` and `railway_data/train_details.csv` next to the app (or upload them from the sidebar). Without them you must type route details by hand.

Use the **same LightGBM version** that trained the model, because pickled models can fail to load across versions. The notebook was developed with LightGBM 4.7.0.

### Retrain from the notebook

Extra packages for the notebook: `duckdb`, `pyarrow`, `shap`, `matplotlib`, `requests`.

1. Download the dataset zip and extract it into `railway_data/`.
2. Run `train_delay_model.ipynb` from top to bottom. It cleans the data, builds features, trains LightGBM, evaluates, and saves the files in `model/`.

---

## App pages

| Page | What it does |
|---|---|
| **Predict a delay** | Pick a train, stop and date; say whether it is running; get the expected delay and arrival time |
| **Journey forecast** | Enter delays seen so far and forecast the rest of the route |
| **What if...** | Change the last-stop delay, weekday or season and see the effect |
| **Batch predictions** | Upload a CSV, get predictions for every row, and see accuracy if actual delays are included |
| **How the model thinks** | Feature importance in plain language |
| **About the project** | Scope, method and limitations |

---

## Limitations

- **Pre-departure predictions are rough.** Without a live delay the model has little to go on.
- **Very long delays (6+ hours) are rare** and often sudden (diversions, cancellations), so the model under-predicts them.
- **Only about one year of data.** Seasonal patterns are learned from a single year, and January was never seen in training (data starts 8 Feb 2025). Fog-season results come mostly from December.
- **Only trains and stations seen in training** are supported, and only NR-zone trains.
- **Manual input.** The app does not read live railway feeds; you enter the delays yourself.
- **Data ages.** Retrain as new data arrives.

### Known issues and roadmap

- [ ] The app does not yet supply `train_avg_delay_min` and `station_avg_delay_min`, which the model was trained with. Export these two lookups from the notebook and load them in the app.
- [ ] Validate pre-departure mode with the model the app actually uses (or ship the separate pre-departure model).
- [ ] Evaluate multi-stop *Journey forecast* accuracy, since it feeds predictions back in as inputs.
- [ ] Add a "previous stop delay" baseline on the final test split to the notebook.
- [ ] Collect more months or years of data and re-test seasonal effects.

---

## Built with

Python · pandas · DuckDB · PyArrow · LightGBM · scikit-learn · SHAP · Plotly · Streamlit · Open-Meteo API

## Author

Designed and built by **Shreshtha Pandey**.

## Disclaimer

Predictions are statistical estimates and are not official Indian Railways data.
