"""Case Study I (Section 5.1): implementation details, hyperparameter grids and result files.

The paper refers to the code repository for these settings.  Values that the code stores as named constants
are read from env_r2.py, run_r2.py and plot_r2.py, and the grids from the grid files of the dense design, so
this summary cannot drift from the code.  Print it with

    python3 config_r2.py

Data-generating process (env_r2.py, R2_DESIGN=dense)
  * d = 729 contextual features; the parameter (theta0, theta1) has dimension kappa = 730; q = 4, p = q/(q-1).
  * theta0 has i.i.d. normal entries with mean 0.208 and standard deviation 0.523, drawn once (seed 777);
    theta1 = 5.
  * x_t is i.i.d. over time, with coordinates i.i.d. N(0, 15^2) truncated to [0.5, 5].
  * D_t = theta0' x_t - theta1 p_t + eps_t with eps_t ~ N(0, sigma^2).  Realized demand is truncated at zero,
    which almost never binds in this design; the oracle and the regret use the expected revenue
    p E[max(D_t, 0)].
  * SNR = ||theta||^2 lambda_min(E[x x']) / sigma^2 in {0.1, 1, 10}, with lambda_min estimated from 5,000 draws.
  * Prices: M = 401 equally spaced prices in [30, 50]; exploration prices are drawn uniformly from 6 equally
    spaced prices in the same interval.
  * Exploration (Algorithm 1): m_0 = m_k = floor(kappa^(1 - 2/q)) = 27; M_0 = [1, 27], M_k = [k^2, k^2 + 26]; the
    exploration periods are the union of the blocks (periods 1-195 all explore), 2,518 in total, with a refit at
    the end of each block except at T: 99 updates for T = 10,000.
  * 10 trials; within a trial all methods face the same features, noise and exploration prices.

Estimators (run_r2.py, solvers_r2.py)
  * Every estimator is refit at the end of each exploration phase on all exploration samples collected so far.
  * AMLE-PGD (Algorithm 2): eta_k = eta0 kappa^(1/q - 1/2) sqrt(|I_k|) and eps_k = eps0 kappa^(1/2 - 1/q)
    sqrt(|I_k|).  Each update starts from the previous estimate (the first from 0.1 * 1) and uses the step
    size 1/L_k, where L_k is 1.02 times the largest eigenvalue of the Hessian of the negative log-likelihood,
    computed by warm-started power iterations (run_r2.py, ``1.02 * lam_max``).  The proximal step is the prox of
    the half-squared l_p norm, computed by bisection as in Lemma C.2 (solvers_r2.prox_half_sq_pnorm).  An update
    stops as soon as ||G^{j+1}||_2 <= eps_k, which certifies Condition 1; a safeguard caps an update at
    20,000 iterations.  All updates stopped by the certificate except 0.2% of the updates of one configuration
    (eta0 = 0.5, eps0 = 3 at mid SNR).
  * C_k = R^kappa.  AMLE-MOSEK, ILQX and AdaGrad keep the bound ||v||_1 <= 1e4 as a numerical safeguard; it is
    never active.
  * AMLE-MOSEK computes the exact minimizer of the same regularized problem with MOSEK through CVXPY.  ILQX
    solves the Lasso problem with lambda_t = lambda0 t^(1/4) sqrt(log kappa + log t), t = current period (as in
    Ban and Keskin 2021), exactly with MOSEK.  When MOSEK stalls (at most 2%, 17% and 44% of the updates of a
    configuration at high, mid and low SNR), the same problem is solved again with the objective divided by
    |I_k| and, if needed, with CLARABEL (solvers_r2._solve_with_fallback).  In the timing comparisons such an
    update is charged the median time of a regular MOSEK solve of the same update over the other trials
    (plot_r2.aggregate).
  * SPO+: two-layer MLP with 64 hidden units, trained at each update on all exploration samples for 60 epochs
    with Adam (mini-batches of 64, L2 penalty 1e-5 on the network parameters) and doubly robust labels; the
    decision is the grid price with the largest predicted revenue, found by enumerating the M prices.
  * AdaGrad updates after every observation, starts from 0.5 * 1 and projects onto this l1 bound.

Selection rules (plot_r2.py)
  * Best hyperparameter of a method: minimum mean final regret over the 10 trials.
  * Representative AMLE-PGD configuration: at the best eta0, the largest eps0 whose mean final regret is within
    1% of the minimum.

Results for all hyperparameters (figures/dense_alg1/, from ``R2_DESIGN=dense python3 plot_r2.py --tag dense_alg1``)
  fig_all_params_final.pdf          final regret of every method and hyperparameter (mean, 95% CI)
  fig_all_params_curves_pgd.pdf     regret curves of AMLE-PGD for every eps0 at three values of eta0
  fig_all_params_curves_bench.pdf   regret curves of the benchmarks for every hyperparameter
  fig_tradeoff_eps.pdf              effect of eps0 on regret, time and iterations at three values of eta0
  fig_pgd_iterations.pdf            AMLE-PGD iterations per update against |I_k|
  summary_all.csv, summary_best.csv, table_best.tex, table_regret_time.tex
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("R2_DESIGN", "dense")

import env_r2 as env   # noqa: E402
import run_r2          # noqa: E402  (PGD and SPO+ constants)

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_FILES = ["grid_dense.json", "grid_dense_adagrad.json"]   # the AdaGrad file extends the lr grid
FIGURE_DIR = os.path.join(HERE, "figures", "dense_alg1")
PGD_TOL = 0.01   # representative AMLE-PGD (same value as plot_r2.PGD_TOL)
STEP_FACTOR = 1.02   # L_k = STEP_FACTOR * largest eigenvalue (run_r2.run_one)
SPO_BATCH = 64
ADAGRAD_INIT = 0.5


def grids() -> dict:
    """{snr: {method: sorted list of hyperparameter dicts}} for the dense design (Section 5.1)."""
    out = {snr: {m: [] for m in run_r2.METHODS} for snr in env.TARGET_SNR}
    for name in GRID_FILES:
        with open(os.path.join(HERE, name)) as f:
            g = json.load(f)
        for snr in out:
            per = g["by_snr"][snr] if "by_snr" in g else g
            for m, params in per.items():
                out[snr][m] += [p for p in params if p not in out[snr][m]]
    for snr in out:
        for m in out[snr]:
            out[snr][m].sort(key=lambda p: tuple(sorted(p.items())))
    return out


def settings() -> dict:
    assert env.DESIGN == "dense", "config_r2 describes the dense design of Section 5.1"
    return {
        "d": env.D_FEAT, "kappa": env.KAPPA, "q": env.Q_NORM, "p": env.P_REG,
        "theta0 mean / sd": (float(env.THETA0.mean()), float(env.THETA0.std())), "theta1": env.GAMMA0,
        "T": env.T_HORIZON, "trials": env.K_TRIALS, "m_k": env.M_EXPLORE,
        "updates": int(env.schedule()[1].sum()),
        "price grid": (float(env.PRICE_GRID[0]), float(env.PRICE_GRID[-1]), env.PRICE_GRID.size),
        "exploration prices": [float(v) for v in env.EXPLORE_GRID],
        "sigma by SNR": {snr: env.noise_scale(snr) for snr in env.TARGET_SNR},
        "l1 safeguard bound (never active)": env.W_L1,
        "AMLE-PGD": {"first start": run_r2.PGD_INIT, "step factor": STEP_FACTOR,
                     "iteration safeguard": run_r2.PGD_MAX_ITER},
        "SPO+": {"hidden units": run_r2.SPO_HIDDEN, "epochs": run_r2.SPO_EPOCHS, "L2 penalty": run_r2.SPO_L2,
                 "batch": SPO_BATCH},
        "AdaGrad": {"start": ADAGRAD_INIT},
        "representative AMLE-PGD tolerance": PGD_TOL,
    }


def main() -> None:
    for k, v in settings().items():
        print("{:36s} {}".format(k, v))
    print("\nHyperparameter grids")
    g = grids()
    for m in run_r2.METHODS:
        for snr in env.TARGET_SNR:
            keys = sorted({k for p in g[snr][m] for k in p})
            vals = {k: sorted({p[k] for p in g[snr][m]}) for k in keys}
            print("  {:8s} {:5s} {}".format(m, snr, "; ".join("{} in {}".format(k, vals[k]) for k in keys)))
    print("\nResults for all hyperparameters:", FIGURE_DIR)
    for name in sorted(os.listdir(FIGURE_DIR)) if os.path.isdir(FIGURE_DIR) else []:
        print("  ", name)


if __name__ == "__main__":
    main()
