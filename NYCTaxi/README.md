# NYC Taxi Online Pricing Experiments

This repository reproduces the **simulator training** and **December 2014
out-of-sample testing** pipeline for online taxi fare pricing under a learned
passenger-acceptance environment.

## Experiment overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  TRAIN (Simulator/) — Kaggle NYC Taxi Fare Prediction, 2013-12 → 2014-11 │
├─────────────────────────────────────────────────────────────────────────┤
│  train.csv  →  load_data_with_map.py  →  processed feature CSV         │
│            →  beta_simulator.py        →  beta_sensitivity_outputs_2013_4/ │
│            →  simulator_response.py    (load artifacts + accept/reject)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  trained β parameters
┌─────────────────────────────────────────────────────────────────────────┐
│  TEST (2014_data/) — NYC TLC yellow+green, December 2014                │
├─────────────────────────────────────────────────────────────────────────┤
│  *.parquet → combine_yellow_green_taxi.py → sorted feature CSV         │
│           → filter_data.py (optional)                                   │
│           → [timestamp de-duplication — see 2014_data/README.md]        │
│           → data/2014_december_test_arrivals.csv                        │
│           → PGD_ILQX.py (online pricing simulation)                     │
│           → plot_cumulative_regret.py → regret/reward PDFs (sim_output/) │
└─────────────────────────────────────────────────────────────────────────┘
```

| Stage | Folder | Details |
| --- | --- | --- |
| Feature engineering + simulator training | [`Simulator/`](Simulator/) | [Simulator/README.md](Simulator/README.md) |
| December 2014 test + online algorithms | [`2014_data/`](2014_data/) | [2014_data/README.md](2014_data/README.md) |

**Raw data is not redistributed.** You must download:

- **Training:** [Kaggle — New York City Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction) (`train.csv`)
- **Weather (auxiliary):** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) → `Simulator/daily_weather_data_2009_2015.csv`
- **Taxi zones (auxiliary):** [NYC TLC Taxi Zone Maps](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) → `Simulator/taxi_zones.zip`
- **Testing:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — December 2014 yellow and green parquet files (direct links in `2014_data/README.md`)

Pre-trained simulator artifacts ship under `Simulator/beta_sensitivity_outputs_2013_4/` so you can run the December experiments without retraining.

## Quick start

```bash
# Install dependencies (adjust per stage; see each README)
pip install -r Simulator/requirements.txt
pip install -r 2014_data/requirements.txt

# Use pre-trained simulator + your locally prepared December test CSV
export TEST_DATA_PATH=2014_data/data/2014_december_test_arrivals.csv
python 2014_data/PGD_ILQX.py
```

See the per-folder READMEs for the full data-prep and retraining workflows.
