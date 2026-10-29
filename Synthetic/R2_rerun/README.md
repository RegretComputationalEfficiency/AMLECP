# R2 rerun of Section 5.1 (Case Study I: linear demand)

Rerun of the simulation study of Section 5.1 (Case Study I) and of the sensitivity study to q (Figure EC.1),
written for the second-round revision. The first-round code is in `../Main_Gaussian/` and `../q_sensitivity/`.

The results in the paper come from runs with the exploration schedule of Algorithm 1 (`env_r2.schedule`).
The summaries for all hyperparameters are in `figures/dense_alg1/` (Table 2 and Figure 4) and
`figures/qsens_alg1/` (Figure EC.1). The raw per-trial outputs (`results/<tag>/`, about 600 MB) are not
included; the commands under Running regenerate them in `results/dense_alg1`, `results/qsens_alg1`,
`results/qsens_tol_alg1` and `results/qsens_heavy100_alg1`.

Requirements (versions used): Python 3.12, numpy 1.26, scipy 1.11, cvxpy 1.5 with MOSEK 11 (license required)
and CLARABEL 0.10, PyTorch 2.13 (SPO+), matplotlib 3.9.

## Files

| File | Content |
|---|---|
| `env_r2.py` | Data-generating process (identical to R1), Algorithm 1 exploration schedule, revenue / oracle, eta_k, eps_k, lambda_t |
| `truncate_normal_sample_time_varying.py` | Feature sampler of the first-round code (the same file as `../Main_Gaussian/gaussian_pricing/utils/truncate_normal_sample_time_varying.py`), imported by `env_r2.py` |
| `solvers_r2.py` | AMLE-PGD (proximal gradient, Algorithm 2), AMLE-MOSEK, ILQX, AdaGrad, SPO+ |
| `run_r2.py` | Main experiment script: one job = (SNR, method, hyper-parameter, trial k) -> one `.npz` |
| `plot_r2.py` | Figures and tables |
| `config_r2.py` | Settings of Case Study I that the paper refers to the repository for: implementation details, hyperparameter grids, selection rules and the result files for all hyperparameters (`figures/dense_alg1/`); `python3 config_r2.py` prints them |
| `tests_r2.py` | Checks: environment = R1, prox / PGD vs MOSEK, SPO+ loss |
| `check_timing_r2.py` | Flags runs whose per-update times are implausible (e.g. the machine slept); `--delete` removes them for recomputation |
| `grid_dense.json` | Per-SNR grids of the dense design (`{"by_snr": {snr: {method: [params]}}}`; see below) |
| `grid_qsens.json`, `plot_qsens_r2.py` | Sensitivity to q (Figure EC.1): q in {2, 1 + log kappa} at mid SNR in the dense design (results/qsens_alg1; q = 4 is taken from results/dense_alg1); the parameter `q` of a configuration sets p = q/(q-1) and the dimension factors of eta_k and eps_k, while the exploration schedule of Section 5.1 is kept for every q |
| `grid_qsens_tol.json`, `grid_qsens_heavy100.json` | Sensitivity to q at a common tolerance (Figure EC.1): every q uses the tolerance of Section 5.1, eps_k = eps0 kappa^(1/4) sqrt(|I_k|) (parameter `eps_q` = 4), so that AMLE-PGD is compared at the same accuracy. Case I: AMLE-PGD for q = 2 and 1 + log kappa (results/qsens_tol_alg1; q = 4 and AMLE-MOSEK are reused from results/dense_alg1 and results/qsens_alg1). Case II: design `heavy100` (prices in units of 100 in the estimation problem, which lowers the condition number of A'A from about 1.8e7 to 7e4; eta grids centred by an offline pilot), all q (results/qsens_heavy100_alg1) |
| `grid_dense_adagrad.json` | AdaGrad lr in {0.316, 3.16, 31.6, 100} for the dense design (lr = 10 was best at high SNR, the edge of the decade grid) |
| `queue_qsens_alg1.sh` | Run queue of the q-sensitivity runs used in Figure EC.1 (results/qsens_alg1, results/qsens_tol_alg1, results/qsens_heavy100_alg1) |

## Designs

`env_r2.py` reads the design from the environment variable `R2_DESIGN`, set by `run_r2.py --design`
(default `base`); each `results/<tag>` folder holds one design, recorded in `results/<tag>/DESIGN`.
The paper uses `dense` (Section 5.1 and Case I of Figure EC.1) and `heavy100` (Case II of Figure EC.1).

| Design | theta0 | Everything else |
|---|---|---|
| `base` | R1 parameters: approximately sparse (96 of 729 entries in [1, 2] carry 99.98% of the squared l2 norm; the rest in [0.001, 0.015]) | as below |
| `heavy100` (`results/qsens_heavy100_alg1`) | as `heavy`, with the price column of the estimation problem divided by 100 | as `heavy` |
| `heavy` | Case II of the q-sensitivity study: sparse theta0 = (50, 20, 0.3, 0.2, 0.1, 0, ...), theta1 = 0.1; heavy-tailed features (R1 sampler `l4_heavy_tail_sample`: N(0, 0.1^2) coordinates shifted by 0.3, a collinear pair whose second coordinate jumps by +-U(5, 10) with probability 0.05); prices on [0, E[theta0'x]/theta1] = [0, 211.8] (501 / 11 points); T = 5,000 | sigma calibrated as in R1 (SNR = 1: sigma = 3.36) |
| `dense` (`results/dense_alg1`) | i.i.d. Gaussian entries with the mean and standard deviation of the R1 vector (seed 777): same sum, same l2 norm, hence the same mean intercept, spread of theta0^T x and sigma at each SNR; 245 negative entries | identical to `base` |

Grids of the dense design (`grid_dense.json`): AMLE-MOSEK and ILQX use five half-decade values (ratio sqrt(10))
centred on the optimum of an offline pilot (per SNR, same rule for both methods); AMLE-PGD uses the three central
eta0 values x eps0 in {3, 10, 30, 100, 300, 1000}.

`plot_r2.py` also reports the exploitation regret (sum over exploitation periods only): the exploration
periods use the same prices for every method and contribute about 658K to every method's regret in
the dense design.

## Environment (unchanged from R1)

- d = 729 features, parameter dimension kappa = 730 (theta0, gamma); theta0 from seed 4242 (85% weak, 15% strong
  entries), gamma0 = 5.
- Features: truncated normal on [0.5, 5] (mu = 0, sigma = 15), the same sampler and seeds as R1.
- Demand d_t = max(theta0^T x_t - 5 p_t + eps_t, 0), eps_t ~ N(0, sigma^2); sigma = 38.73 / 12.25 / 3.87 for
  SNR = 0.1 / 1 / 10, where SNR = ||theta0||^2 lambda_min(E[x x^T]) / sigma^2 (R1 calibration, 5000 draws).
- Exploitation price grid: 401 points on [30, 50]; exploration prices: 6 points on [30, 50].
- Oracle: max_p p E[max(theta0^T x - 5 p + eps, 0)] over the grid.
- `tests_r2.py` verifies that the calibrated sigma's and the oracle revenues coincide with the first-round
  outputs (`Linear_code/results/*_SNR0/`, not included in this repository) to machine precision.

## What differs from R1 (to match the paper)

| Item | R1 code | R2 rerun |
|---|---|---|
| Estimation history | initial 73 samples + one sample per epoch | every exploration period (Algorithm 1: I_k = union of M_s) |
| Exploration length | 73 initial, then 36 per epoch | m_0 = m_k = floor(kappa^(1-2/q)) = 27; blocks M_0 = [1, 27], M_k = [k^2, k^2 + 26] of Algorithm 1 (exploration = union of the blocks) |
| Refit time | end of each epoch | end of each block M_k (no refit at T); the estimate is used until the next refit |
| Trials | K = 5 | K = 10 |
| Noise / exploration draws | `np.random.seed(k + t)` (trials are shifted copies) | independent stream per trial, common to all methods |
| Objective | PGD: sample-averaged; MOSEK: not averaged | both minimize D_k = 0.5 ||y - A v||^2 + eta_k * 0.5 ||v||_{4/3}^2 |
| eta_k | eta0 * d^(1/2-1/q) * \|I_k\|^(1/4) | eta0 * kappa^(1/q-1/2) * sqrt(\|I_k\|) (corrected Proposition 1) |
| eps_k | eps0 * d^(1/2-1/q) * sqrt(\|I_k\|) on the averaged gradient | eps0 * kappa^(1/2-1/q) * sqrt(\|I_k\|) on the gradient of D_k |
| AMLE-PGD | momentum, adaptive step, 1000-iteration cap, gradient-norm stop | proximal gradient, step 1/L_k (L_k = 1.02 * lambda_max(A^T A)), prox of the l_{4/3} term (Lemma C.2), stop when the certificate \|\|G\|\| <= eps_k (verifies Condition 1) |
| AMLE-MOSEK / ILQX | MOSEK, fallback CLARABEL/SCS | MOSEK; if it stalls, MOSEK on the objective scaled by 1/N, then CLARABEL (recorded per fit) |
| ILQX penalty | lambda_t = lambda0 t^(1/4) sqrt(log kappa + log t) with t = number of stored samples | same formula with t = current period, as in Ban and Keskin (2021) |
| SPO+ | PyEPO + Gurobi MILP per sample | same loss (maximization form of SPO+), exact argmax over the one-hot price choice; no pyepo/Gurobi |
| AdaGrad | missing results | port of R1 `AdaGrad.py` (init 0.5, l1 projection), updated every period |
| Regret | realized revenue | pseudo-regret with the expected revenue at the posted price (paper definition); realized revenue is also stored |
| Best hyper-parameter | full-horizon final regret (Table 2) / holdout (code) | minimum mean final regret, stated in the paper |

## Running

```bash
# Checks that do not need the first-round outputs (python3 tests_r2.py also runs test_environment_matches_r1)
python3 -c "import tests_r2 as t; t.test_schedule(); t.test_prox_against_cvxpy(); t.test_pgd_against_mosek(); t.test_spoplus_loss()"

# Runs used in the paper (exploration schedule of Algorithm 1)
python3 run_r2.py --design dense --tag dense_alg1 --snr low mid high --methods all --grid main --grid-json grid_dense.json --workers 9
python3 run_r2.py --design dense --tag dense_alg1 --snr low mid high --methods AdaGrad --grid main --grid-json grid_dense_adagrad.json --workers 9
caffeinate -i bash queue_qsens_alg1.sh              # q-sensitivity runs of Figure EC.1
python3 check_timing_r2.py --tag dense_alg1          # timing sanity check
R2_DESIGN=dense python3 plot_r2.py --tag dense_alg1  # Table 2, Figure 4 and results for all hyperparameters (figures/dense_alg1)
python3 plot_qsens_r2.py                             # Figure EC.1 (figures/qsens_alg1)
```

Jobs whose output exists are skipped, so an interrupted run can be resumed with the same command.
Each `.npz` stores prices, demand, expected/realized revenue, oracle revenue, and per-update logs
(sample size, time, PGD iterations, stop reason, certificate norm, first-order residual).
