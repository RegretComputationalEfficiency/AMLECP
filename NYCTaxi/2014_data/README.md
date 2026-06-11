# December 2014 Online Pricing Experiments

This folder runs **out-of-sample** online pricing simulations on NYC yellow and
green taxi trips from **December 2014**, using the passenger-acceptance simulator
trained in [`Simulator/`](../Simulator/README.md).

The main entry point is `PGD_ILQX.py`, which loads trained `beta_simulator`
parameters from `Simulator/beta_sensitivity_outputs_2013_4/` and simulates
arrival streams under PGD and ILQX pricing policies.

---

## Data availability & citation

**Raw trip data is NOT included** in this repository. Download the official NYC
TLC parquet files and place them under `TestData/`:

| File | URL |
| --- | --- |
| Yellow taxis, 2014-12 | <https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2014-12.parquet> |
| Green taxis, 2014-12 | <https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2014-12.parquet> |

Source page: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Suggested acknowledgement:

```text
NYC Taxi & Limousine Commission. Trip Record Data.
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
```

Auxiliary lookups are reused from `Simulator/`:

- **Weather:** download `daily_weather_data_2009_2015.csv` from
  [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
  into `Simulator/` (not shipped in this repo — see
  [Simulator/README.md](../Simulator/README.md#weather-data-daily_weather_data_2009_2015csv))
- **Taxi zones:** download `taxi_zones.zip` from NYC TLC *Taxi Zone Maps*
  ([trip-record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page))
  into `Simulator/` (not in repo — see [Simulator/README.md](../Simulator/README.md))

---

## Pipeline

### Step 1 — Merge yellow + green, engineer features, sort by time

```bash
# After placing the two parquet files in TestData/
python 2014_data/TestData/combine_yellow_green_taxi.py
```

**Output:** `TestData/2014_combined_pickup_datetime_taxi_with_directionality.csv`

This script:

- Standardises yellow (`tpep_pickup_datetime`) and green (`lpep_pickup_datetime`)
  schemas
- Joins taxi-zone and weather features (same schema as simulator training)
- Appends directionality / congestion flags
- Merges both fleets and **sorts globally by `pickup_time` ascending**

### Step 2 — Filter invalid timestamps (optional but recommended)

```bash
python 2014_data/TestData/filter_data.py
```

**Input:** `2014_combined_pickup_datetime_taxi_with_directionality.csv`  
**Output:** `2014_filtered_combined_pickup_datetime_taxi_with_directionary.csv`

Drops rows whose `pickup_time` cannot be parsed as a datetime.

### Step 3 — Duplicate pickup-time handling (academic note)

Many TLC records share the **exact same second-level `pickup_time`** after the
merge. For the online arrival model used in the paper, we apply an additional
**timestamp de-duplication** step: contiguous rows with identical `pickup_time`
strings are assigned distinct sub-second offsets (when the gap to the next
distinct timestamp allows), so that each simulated arrival has a unique time
ordering. Rows whose timestamps are already unique are left unchanged.

This preprocessing is **described here for reproducibility** but its
implementation is **not distributed** in this repository. Researchers who wish
to replicate the exact arrival stream should implement an equivalent procedure
on the sorted CSV from Step 1 (or Step 2).

After this step, save the result as:

```text
2014_data/data/2014_december_test_arrivals.csv
```

This is the default path consumed by `PGD_ILQX.py`.

### Step 4 — Run online pricing simulation

```bash
# Default: reads 2014_data/data/2014_december_test_arrivals.csv
python 2014_data/PGD_ILQX.py

# Or point at your own processed file:
TEST_DATA_PATH=path/to/your_arrivals.csv python 2014_data/PGD_ILQX.py
```

**Simulator scenarios** (loaded from `Simulator/beta_sensitivity_outputs_2013_4/`):

- `pair_1p5__noise_05`
- `pair_1p5_2p0__noise_05`
- `pair_1p5_2p0__noise_10`
- `pair_1p5_2p0__noise_20`

Simulation outputs (CSVs, PDFs, cached runs) are written under
`TestData/sim_output/` and are **not uploaded** to the repository.

### Step 5 — Plot results (`plot_cumulative_regret.py`)

After `PGD_ILQX.py` finishes, generate paper-style figures from the streamed
`revenue_regret.csv` files. Inline plotting inside `PGD_ILQX.py` is disabled
(`GENERATE_INLINE_PLOTS = False`); use this script instead.

```bash
# Default: read TestData/sim_output/, write PDFs alongside the CSVs
python 2014_data/TestData/plot_cumulative_regret.py

# Custom input root and stop-eps filter (defaults match PGD_ILQX STOP_EPS_VALUES)
python 2014_data/TestData/plot_cumulative_regret.py \
    --input-root 2014_data/TestData/sim_output \
    --stop-eps 1.0 --stop-eps 1.5

# Preview interactively
python 2014_data/TestData/plot_cumulative_regret.py --show
```

**Expected input tree** (one folder per hyperparameter run):

```text
TestData/sim_output/
└── pair_1p5_2p0__noise_05/          # beta_simulator scenario
    └── repeat_0/
        └── eta0.05_lam0.05_stopeps1.0_etapow0.0/
            └── revenue_regret.csv
```

**Output PDFs** (written to `--input-root` by default):

| File | Content |
| --- | --- |
| `cumulative_regret_styled.pdf` | TSP, ILQX, and AMLE-PGD cumulative regret curves |
| `total_reward_bars.pdf` | Mean total reward bars (all hyperparameter configs) |
| `cumulative_regret_best.pdf` | Best ILQX + best AMLE-PGD per scenario, plus TSP |
| `total_reward_bars_best.pdf` | Bar chart for those best configs only |

**Strategies shown**

- **TSP** — baseline using the realised fare (`reward_original` / `regret_original`)
- **ILQX** — exact CVX-MLE (`reward_cvx` / `regret_cvx`)
- **AMLE-PGD** — penalised gradient pricing (`reward_pgd` / `regret_pgd`)

Scenarios, η / λ grids, and `stop_eps` filters are kept in sync with
`PGD_ILQX.py` via the constants at the top of `plot_cumulative_regret.py`.

---

## Layout

```
2014_data/
├── PGD_ILQX.py                 # main online-pricing experiment
├── requirements.txt
├── data/
│   └── 2014_december_test_arrivals.csv   # user-built; git-ignored
└── TestData/
    ├── combine_yellow_green_taxi.py      # Step 1
    ├── filter_data.py                    # Step 2
    ├── plot_cumulative_regret.py         # Step 5
    ├── green_tripdata_2014-12.parquet  # download; git-ignored
    ├── yellow_tripdata_2014-12.parquet # download; git-ignored
    └── sim_output/                     # PGD_ILQX CSVs + plot PDFs; git-ignored (except .gitkeep)
```

---

## Configuration highlights (`PGD_ILQX.py`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_PATH` | `data/2014_december_test_arrivals.csv` | Test arrival CSV (override via `TEST_DATA_PATH`) |
| `BETA_SENSITIVITY_OUTPUT_ROOT` | `Simulator/beta_sensitivity_outputs_2013_4` | Trained simulator artifacts |
| `DATA_DAY` | `"2014-12-06"` | Calendar slice for simulation (`None` = full file) |
| `DATA_HOUR_START` / `DATA_HOUR_END` | `7` / `9` | Hour window within `DATA_DAY` |
| `OUTPUT_ROOT` | `TestData/sim_output` | Per-run CSV / summary outputs |

---

## Installation

```bash
pip install -r 2014_data/requirements.txt
```

`cvxpy` (and a MOSEK licence, if available) enables the exact CVX-MLE baseline
inside `PGD_ILQX.py`. The PGD path runs without MOSEK.
