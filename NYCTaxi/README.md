# NYC taxi case study (Section 5.2)

Code, simulator and results of Case Study II of the revised paper: online pricing of New York City taxi requests,
the computation time of each update, and the revenue lost when an update is late. The code of the first-round
version is kept at the tag [`round1`](https://github.com/RegretComputationalEfficiency/AMLECP/tree/round1).

## Design

- **Simulator.** Every trip gets a willingness to pay equal to its recorded fare times a lognormal multiplier
  (log standard deviation 0.3, median depending on the trip features). Two offers per trip, between 0.4 and 4
  times the fare, are drawn for 2,000,000 trips from December 2013 to November 2014 (excluding the week of
  November 17, 2014) and labeled by this rule. The logit model of Section 5.2 ($\kappa_\theta=41$) fitted to these
  offers is the ground truth. The median multiplier at the baseline is 1.6 (`wtp16`, main scenario), with 1.4
  (`wtp14`) and 1.8 (`wtp18`) as further scenarios. The fitted parameters are in `data/`.
- **Features.** 20 contextual features, defined in [`feature_summary.md`](feature_summary.md).
- **Schedule.** Algorithm 1 with $m_0=410$ and $m_k=28$. Exploration prices are drawn from
  $\{10,15,20,25,30\}$ USD, and exploitation maximizes the revenue over $[2.5,70]$ USD.
- **Policies.** AMLE-PGD with $q=4$, $\eta_k=\mathring\eta\,\kappa_\theta^{1/q-1/2}\sqrt{|\mathcal I_k|}$ and
  $\varepsilon_k=\mathring\varepsilon\sqrt{\kappa_\theta|\mathcal I_k|}$. AMLE-MOSEK solves the same problem
  exactly. ILQX uses $\lambda_t=\mathring\lambda t^{1/4}\sqrt{\log\kappa_\theta+\log t}$. TSP is the recorded fare.
- **Selection** on the first 200,000 requests of the week of November 17, 2014, by the revenue with updates
  available at once. For AMLE-PGD (10 seeds), $\mathring\eta$ minimizes the mean regret at the smallest tolerance
  $\mathring\varepsilon=0.0003$, and $\mathring\varepsilon$ is the largest tolerance whose mean regret is within
  $1\%$ of the minimum at that $\mathring\eta$. For ILQX (3 seeds), $\mathring\lambda$ minimizes the mean regret.
  Selected values are $\mathring\eta=0.005$, $\mathring\varepsilon=0.002$ (wtp16, wtp18) or $0.005$ (wtp14), and
  $\mathring\lambda=0.01$.
- **Test** on the first 200,000 requests of the weeks starting December 1, 8, 15 and 22, 2014, three seeds per
  week (12 runs per scenario). Every update is timed, one run after another on an idle machine with one thread.
- **Deployment.** One worker computes the updates and, when free, starts the newest completed exploration phase.
  Two pricing rules are replayed from the same timed runs. Under the newest-estimate rule, each request is priced
  with the newest estimate finished before it arrives (20 USD before the first one). Under the epoch rule of
  Algorithm 1, a request of epoch $k$ is priced with the estimate of epoch $k$ once it is finished and with the
  recorded fare before that. To vary the request rate, the arrival times are divided by a constant (10 to 100,000
  requests per second, besides the recorded rate of 3.9).

## Where the paper's numbers come from

| Paper | File |
|---|---|
| Table 4 and the results of Section 5.2 | `formal_report.txt`, computed from `formal.csv` |
| Figure 5(a), (b), (c) | `figures/formal_pressure_wtp16_1.pdf`, `_2.pdf`, `_3.pdf` |
| Scenarios with median multipliers 1.4 and 1.8 | `formal_report.txt` |
| Selection on the validation week | `results/formal/<scenario>/selection.csv`, `selected.json` |
| Feature definitions and statistics | `feature_summary.md` |

`formal.csv` has one row per scenario, week, seed, policy, request rate and window (`all`, `first20`, `first5`).
The revenue columns are sums in USD over the window: `instant` (updates available at once), `deployed`
(newest-estimate rule), `lock` (epoch rule), `oracle` and `tsp`. `timely` is the timeliness rate, `behind` the share
of exploration observations missing from the estimate in use, `seconds` the total update time and `machine_load`
the peak one-minute load during the run.

`results/formal/<scenario>/<week>/seed<k>/<policy>.npz` holds the estimates and measured times of all updates of a
test run. `live/` holds AMLE-PGD re-solved with its true warm starts at the request rates where updates are skipped.

## Files

| File | Purpose |
|---|---|
| `build/build_data.py`, `build/config.py`, `build/make_cache.py` | Cleaning and features from the TLC records |
| `run.py` | Data access, design matrix and the simulator fit (`python run.py prepare`) |
| `replay.py`, `solvers.py` | Exploration schedule, pricing, and the AMLE-PGD, AMLE-MOSEK and Lasso solvers |
| `single_update.py`, `online.py` | The update of each policy (`single_update.solve`) and the replay timeline used by `formal.py` |
| `formal.py` | `select <scenario>`, `test <scenario>` (timed runs), `evaluate` (writes `formal.csv`) |
| `formal_report.py` | `formal_report.txt` and `figures/` from `formal.csv` |
| `run_formal.sh` | The full pipeline for the three scenarios |
| `environment.json` | Python and package versions of the reported runs |

## Reproducing

Requirements: Python 3 with numpy, scipy, pandas, pyarrow, numba, matplotlib and MOSEK (`environment.json`).
`build/build_data.py` also needs geopandas.

1. Download the yellow taxi trip records of December 2013 to December 2014 and the green records of 2014 from the
   [TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) into `build/data/raw/`,
   and the weather and taxi zone files into `Simulator/` (see `Simulator/README.md`). These data are not
   redistributed here.
2. Build the trip tables:

   ```bash
   cd build && python build_data.py && python make_cache.py && cd ..
   ```

3. Fit the simulator (overwrites `data/`):

   ```bash
   python run.py prepare
   ```

4. Run selection, timed tests, replay and report:

   ```bash
   ./run_formal.sh
   ```

Runs whose output exists are skipped, so with the saved results step 4 only replays the saved updates and rebuilds
`formal.csv`, `formal_report.txt` and the figures. Timed runs wait until the one-minute machine load is below 5, and
runs with a peak load of 7 or more are left out of the delay measures. Update times depend on the machine.
