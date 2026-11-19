# NYC ride-hailing case study (Section 5.2)

This folder holds the code, the simulator and the results of Case Study II of the revised paper. The study prices
New York City ride-hailing requests online, times every update, and measures the revenue lost when an update is late.
The code of the first-round version is kept at the tag
[`round1`](https://github.com/RegretComputationalEfficiency/AMLECP/tree/round1).

## Design

- **Data.** We use the High Volume For-Hire Vehicle trip records (Uber and Lyft) of the NYC Taxi and Limousine
  Commission. The recorded upfront fare is the historical price. Feasible prices are $[5,150]$ USD.
- **Simulator.** Each trip gets a willingness to pay equal to its fare times a lognormal multiplier. The log
  standard deviation is 0.3, and the median depends on the trip features and is 1.6 at the baseline.
  - We draw two offers per trip, between 0.4 and 4 times the fare, for 2,000,000 trips from October 2025 to July
    2026, and label them by this rule.
  - The logit model of Section 5.2 is fitted to these offers. It has 20 contextual features and 41 parameters, and
    it is the ground truth (`data/sim/wtp16.npy`).
  - `run.py prepare` also fits the medians 1.4 and 1.8, which are not used in the paper.
- **Features.** Each request carries the 20 contextual features of Table EC.2 and 344 independent standard normal
  features. The standard normal features do not affect demand, in the base part or the price part. The policies
  therefore estimate $\kappa_\theta=729$ parameters, and $s=41$ of them are nonzero.
- **Schedules.**
  - AMLE-PGD and AMLE-MOSEK follow Algorithm 1 with $m_0=410$ and $m_k=28$, using the blocks $[k^2,k^2+27]$.
  - ILQX follows the schedule of Ban et al. (2025) with the true $s=41$. Its $L$th block runs two price
    experiments at request $\lceil L^2/s\rceil$ and the next one, and it re-solves after the block.
  - Exploration prices are drawn from $\{20,30,40,50,60\}$ USD.
- **Policies.**
  - AMLE-PGD uses $q=4$, $\eta_k=\mathring\eta\,\kappa_\theta^{1/q-1/2}\sqrt{|\mathcal I_k|}$ and
    $\varepsilon_k=\mathring\varepsilon\sqrt{\kappa_\theta|\mathcal I_k|}$.
  - AMLE-MOSEK solves the same problem with MOSEK and uses the returned solution as it is.
  - ILQX uses $\lambda_t=\mathring\lambda\,t^{1/4}\sqrt{\log\kappa_\theta+\log t}$.
  - HPF is the recorded fare.
- **Selection** is done on the first 200,000 requests of Monday, September 7, 2026, with three seeds, and uses the
  revenue with updates available at once.
  - For AMLE-PGD, $\mathring\eta$ minimizes the mean regret at the smallest tolerance $\mathring\varepsilon=0.0003$.
    The representative tolerance is the largest $\mathring\varepsilon$ whose mean regret is within 1% of the minimum
    at that $\mathring\eta$.
  - For ILQX, $\mathring\lambda$ minimizes the mean regret.
  - The selected values are $\mathring\eta=1$ and $\mathring\varepsilon=0.001$ (representative), $0.0003$
    (minimum regret), and $\mathring\lambda=0.3$. AMLE-MOSEK uses the same $\mathring\eta$.
- **Test.** The test uses the first 200,000 requests of Mondays, September 14, 21 and 28, 2026, with three seeds per
  day (9 runs per policy). The runs are made one after another on an idle machine with one thread, and every
  update is timed.
- **Deployment.** One worker computes the updates.
  - Under Algorithm 1, the estimate of an epoch is fixed at its first exploitation request. It is the epoch's own
    estimate if that is finished, and otherwise the newest finished estimate. It prices the whole epoch.
  - Before the first estimate, the price is 40 USD.
  - The request rate is varied by dividing the arrival times by a constant, while the solve times stay as measured.
    This is equivalent to multiplying every update time by that constant at the recorded rate of 5.1 requests per second.

## Where the paper's numbers come from

| Paper | File |
|---|---|
| Table 4 and the results of Section 5.2 | `formal_report.txt`, computed from `formal.csv` |
| Figure 5(a), (b) | `figures/fig_taxi_delay_a.pdf`, `figures/fig_taxi_delay_b.pdf` (`figure_paper.py`) |
| Selection on the validation Monday | `results/formal/wtp16/selection.csv`, `selected.json` |
| Simulator parameters | `data/sim/wtp16.npy`, `data/sim/scale.npy`, `data/sim/simulator.json` |

`formal.csv` has one row per policy, test day, seed, request rate (`level`) and window (`all`, `first20`, `first5`).
The `first5` window is the first 10,000 requests. The revenue columns are sums in USD over the window.

- `instant`: revenue with updates available at once.
- `freeze`: revenue under Algorithm 1, as used in the paper.
- `deployed`: revenue when each request uses the newest finished estimate, also within an epoch.
- `oracle` and `tsp`: the oracle revenue and the revenue at the recorded fare.

The loss to delay is `(instant - freeze) / oracle`. `seconds` is the total update time.

For each test run, `results/formal/wtp16/<day>/seed<k>/<policy>.npz` holds the estimates and the measured times of
all updates. `live/` holds AMLE-PGD re-solved with its true warm starts at the request rates where updates are skipped.

## Files

| File | Purpose |
|---|---|
| `build/build_data.py`, `build/config.py` | Cleaning and features from the TLC records |
| `run.py` | Data access, design matrix and the simulator fit (`python run.py prepare`) |
| `replay.py`, `solvers.py` | Exploration schedule, pricing, and the AMLE-PGD, AMLE-MOSEK and Lasso solvers |
| `single_update.py`, `online.py` | The update of each policy and the timeline of the single worker |
| `formal.py` | `select wtp16`, `test wtp16` (timed runs) and `evaluate` (writes `formal.csv`) |
| `mosek_raw.py` | Reruns the AMLE-MOSEK test runs with MOSEK's own solutions |
| `formal_report.py` | `formal_report.txt` from `formal.csv` |
| `figure_paper.py` | Figure 5 from `formal.csv` |
| `run_formal.sh` | The full pipeline |
| `environment.json` | Python and package versions of the reported runs |

## Reproducing

You need Python 3 with numpy, scipy, pandas, pyarrow, numba, matplotlib, geopandas and MOSEK (`environment.json`).

1. Download the raw data. These data are not redistributed here.
   - The HVFHV trip records `fhvhv_tripdata_<YYYY-MM>.parquet` for October 2025 to September 2026 from the
     [TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) into `data/raw/`.
   - The taxi zone shapefile into `data/zones/taxi_zones/`.
   - The daily New York City weather from the
     [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) into
     `data/weather/daily_weather_2025_2026.csv`, covering September 2025 to September 2026. It needs the columns
     `date`, `temperature_2m_max`, `temperature_2m_min` (degrees Fahrenheit), `rain_sum`, `snowfall_sum` and
     `daylight_duration` (seconds).
2. Build the request tables:

   ```bash
   cd build && python build_data.py && cd ..
   ```

3. Run the pipeline. `formal.py test` runs AMLE-MOSEK with repairs of non-optimal solves first, and `mosek_raw.py`
   replaces these runs with MOSEK's own solutions.

   ```bash
   ./run_formal.sh
   ```

   To recompute only the report and the figure from the saved results, run:

   ```bash
   python formal_report.py && python figure_paper.py figures
   ```

   Replaying the saved runs (`python formal.py evaluate`) also needs the request tables of step 2.
