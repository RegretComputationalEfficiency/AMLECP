"""Time of one exact MLE solve as the sample size and the dimension grow (Figure 2, R2 rerun).

Figure 2 times one exact solve of the estimation problem
  (a) for n = 2^4, ..., 2^15 observations with kappa_x = 150 features, and
  (b) for kappa_x = 2^2, ..., 2^12 features with n = 5,000 observations.

Data: D_t = a^T x_t - b p_t + eps_t (linear, not truncated at 0) with b = 5 and eps_t ~ N(0, 5^2).  Features and
prices are drawn as in the exploration periods of Section 5.1 (env_r2): i.i.d. truncated normal entries on
[0.5, 5] and prices uniform on the six exploration prices in [30, 50].  The entries of a are i.i.d. uniform on
[0, 300 / kappa_x], so that E[a^T x_t] does not depend on kappa_x.  Every run draws a new data set.

Solver: the problem of AMLE-MOSEK in Section 5.1 (solvers_r2.amle_mosek),
    min_v  0.5 ||y - A v||_2^2 + eta * 0.5 ||v||_{4/3}^2   s.t.  ||v||_1 <= 1e4 (never active),
with rows a_t = (x_t, -p_t), kappa = kappa_x + 1 and eta = kappa^(1/q - 1/2) sqrt(n) (eta0 = 1, q = 4), solved
by MOSEK through CVXPY at its default tolerances, with the fallback of solvers_r2 if MOSEK stalls.  The time of
a run is the wall-clock time of building and solving the problem, as in run_r2.py.  MOSEK's optimizer time, its
iteration count and the relative first-order residual ||grad D(v)|| / ||A^T y|| are stored as well.
MOSEK runs on one thread (--threads 0: MOSEK's default number of threads).  The dashed lines in the figures are
least-squares fits of log time on log size over the ranges in FIT_RANGES.

    python3 mle_scaling_r2.py --threads 1              # results/mle_scaling/, figures/mle_scaling/
    python3 mle_scaling_r2.py --threads 0              # MOSEK's default number of threads
    python3 mle_scaling_r2.py --threads 1 --plot-only

Runs whose output exists are skipped, so an interrupted run can be resumed with the same command.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import math
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator   # noqa: E402
from scipy.stats import truncnorm                     # noqa: E402

import env_r2 as env                                  # noqa: E402
import solvers_r2 as S                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "results", "mle_scaling")
FIGURE_DIR = os.path.join(HERE, "figures", "mle_scaling")

SEED = 20261029
NOISE_SD = 5.0
A_MEAN_SUM = 150.0           # a_j ~ U[0, 2 * A_MEAN_SUM / kappa_x], so E[sum_j a_j] = A_MEAN_SUM
ETA0 = 1.0
N_REPS = 10

# panel -> (varied column, grid, value of the other column, x-axis label, file name of the R1 figure)
PANELS = {
    "size": ("n", [2 ** e for e in range(4, 16)], 150, "Number of Samples", "MLE_Time_Test"),
    "dim": ("kappa_x", [2 ** e for e in range(2, 13)], 5000, r"Feature Dimension $\kappa_x$", "MLE_Time_Dimension"),
}
FIELDS = ["panel", "n", "kappa_x", "rep", "threads", "wall", "solver_time", "iters", "used", "foc_rel"]
# fitted ranges: the upper half of each grid
FIT_RANGES = {"size": [(2 ** 10, 2 ** 15)], "dim": [(2 ** 7, 2 ** 12)]}


def draw_data(n: int, kx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lo = (env.FEATURE_A - env.FEATURE_MU) / env.FEATURE_SIGMA
    hi = (env.FEATURE_B - env.FEATURE_MU) / env.FEATURE_SIGMA
    X = truncnorm.rvs(lo, hi, loc=env.FEATURE_MU, scale=env.FEATURE_SIGMA, size=(n, kx), random_state=rng)
    a = rng.uniform(0.0, 2.0 * A_MEAN_SUM / kx, size=kx)
    p = rng.choice(env.EXPLORE_GRID, size=n)
    y = X @ a - env.GAMMA0 * p + NOISE_SD * rng.standard_normal(n)
    return np.column_stack([X, -p]), y


def exact_solve(A: np.ndarray, y: np.ndarray, eta: float, threads: int) -> tuple[np.ndarray, str, float, int]:
    """solvers_r2.amle_mosek with MOSEK's number of threads set; also returns MOSEK's time and iterations."""
    import cvxpy as cp

    mosek = {"mosek_params": {"MSK_IPAR_NUM_THREADS": int(threads)}}
    for label, scale, solver, opts in (("mosek", False, cp.MOSEK, mosek), ("mosek_scaled", True, cp.MOSEK, mosek),
                                       ("clarabel", True, cp.CLARABEL, {})):
        v = cp.Variable(A.shape[1])
        obj = 0.5 * cp.sum_squares(y - A @ v) + 0.5 * eta * cp.square(cp.pnorm(v, env.P_REG))
        if scale:
            obj = obj / A.shape[0]
        prob = cp.Problem(cp.Minimize(obj), [cp.norm1(v) <= env.W_L1])
        try:
            prob.solve(solver=solver, verbose=False, **opts)
        except cp.error.SolverError:
            continue
        if v.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            st = prob.solver_stats
            iters = -1 if st.num_iters is None else int(st.num_iters)
            return np.asarray(v.value, dtype=np.float64), label, float(st.solve_time), iters
    raise RuntimeError("all solvers failed")


def csv_path(threads: int) -> str:
    return os.path.join(RESULT_DIR, "timings_threads{}.csv".format(threads))


def load(threads: int) -> list[dict]:
    if not os.path.exists(csv_path(threads)):
        return []
    with open(csv_path(threads), newline="") as f:
        return list(csv.DictReader(f))


def run(threads: int, reps: int) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    done = {(r["panel"], int(r["n"]), int(r["kappa_x"]), int(r["rep"])) for r in load(threads)}
    warnings.filterwarnings("ignore", module="mosek")
    A, y = draw_data(64, 16, np.random.default_rng(0))       # warm-up (licence check, first CVXPY call)
    exact_solve(A, y, 1.0, threads)
    new = not os.path.exists(csv_path(threads))
    with open(csv_path(threads), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        # every replication sweeps all grid points, so slow drifts of the machine affect all sizes alike
        for rep in range(reps):
            for i, (panel, (var, grid, fixed, _, _)) in enumerate(PANELS.items()):
                for g in grid:
                    n, kx = (g, fixed) if var == "n" else (fixed, g)
                    if (panel, n, kx, rep) in done:
                        continue
                    A, y = draw_data(n, kx, np.random.default_rng([SEED, i, g, rep]))
                    eta = ETA0 * (kx + 1) ** (1.0 / env.Q_NORM - 0.5) * math.sqrt(n)
                    t0 = time.perf_counter()
                    v, used, solver_time, iters = exact_solve(A, y, eta, threads)
                    wall = time.perf_counter() - t0
                    foc_rel = S.foc_residual(A, y, v, eta, env.P_REG) / float(np.linalg.norm(A.T @ y))
                    w.writerow({"panel": panel, "n": n, "kappa_x": kx, "rep": rep, "threads": threads,
                                "wall": "{:.6f}".format(wall), "solver_time": "{:.6f}".format(solver_time),
                                "iters": iters, "used": used, "foc_rel": "{:.3e}".format(foc_rel)})
                    f.flush()
                    print("rep {} {:4s} n={:6d} kappa_x={:5d}  {:8.3f} s (MOSEK {:8.3f} s, {} it, {})".format(
                        rep, panel, n, kx, wall, solver_time, iters, used), flush=True)


def fit(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of log2 t = c + s log2 x; returns (s, c)."""
    s, c = np.polyfit(np.log2(x), np.log2(t), 1)
    return float(s), float(c)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for panel, (var, grid, fixed, _, _) in PANELS.items():
        rs = [r for r in rows if r["panel"] == panel]
        x = [g for g in grid if any(int(r[var]) == g for r in rs)]
        col = {k: [np.array([float(r[k]) for r in rs if int(r[var]) == g]) for g in x]
               for k in ("wall", "solver_time", "iters", "foc_rel")}
        xs = np.array(x, dtype=float)
        wall = np.array([np.median(c) for c in col["wall"]])
        solver = np.array([np.median(c) for c in col["solver_time"]])
        half = len(x) // 2
        s_wall, c_wall = fit(xs, wall)
        fits = []
        for lo, hi in FIT_RANGES[panel]:
            m = (xs >= lo) & (xs <= hi)
            if m.sum() < 2:
                continue
            s_fit, c_fit = fit(xs[m], wall[m])
            fits.append({"lo": lo, "hi": hi, "slope": s_fit, "intercept": c_fit,
                         "slope_solver": fit(xs[m], solver[m])[0]})
        out[panel] = {
            "varied": var, "fixed": fixed, "x": x, "reps": int(min(len(c) for c in col["wall"])),
            "wall_median": wall.tolist(),
            "wall_q25": [float(np.percentile(c, 25)) for c in col["wall"]],
            "wall_q75": [float(np.percentile(c, 75)) for c in col["wall"]],
            "solver_median": solver.tolist(),
            "iters_median": [float(np.median(c)) for c in col["iters"]],
            "fallbacks": int(sum(r["used"] != "mosek" for r in rs)),
            "foc_rel_max": float(max(c.max() for c in col["foc_rel"])),
            "slope_wall": s_wall, "intercept_wall": c_wall,
            "slope_solver": fit(xs, solver)[0],
            "upper_from": x[half],
            "slope_wall_upper": fit(xs[half:], wall[half:])[0],
            "slope_solver_upper": fit(xs[half:], solver[half:])[0],
            "local_slope_wall": (np.diff(np.log2(wall)) / np.diff(np.log2(xs))).tolist(),
            "fits": fits,
        }
    return out


def _time_label(y: float, _pos) -> str:
    return "{:.0f}".format(y) if y >= 1 else "{:.2g}".format(y)


def plot(threads: int) -> dict:
    summary = summarize(load(threads))
    os.makedirs(FIGURE_DIR, exist_ok=True)
    suffix = "" if threads == 1 else "_threads{}".format(threads)
    plt.style.use("seaborn-v0_8-whitegrid")
    for panel, (var, _, _, xlabel, name) in PANELS.items():
        s = summary[panel]
        x = np.array(s["x"], dtype=float)
        fig, ax = plt.subplots(figsize=(5.0, 3.2))      # printed at 0.45\textwidth, about 0.6 of this size
        ax.fill_between(x, s["wall_q25"], s["wall_q75"], color="royalblue", alpha=0.18, lw=0,
                        label="Interquartile range")
        ax.plot(x, s["wall_median"], "o-", color="royalblue", lw=1.2, ms=4,
                label="Median time ({} runs)".format(s["reps"]))
        for f, color in zip(s["fits"], ("red", "darkorange")):
            xf = x[(x >= f["lo"]) & (x <= f["hi"])]
            ax.plot(xf, 2.0 ** (f["intercept"] + f["slope"] * np.log2(xf)), "--", color=color, lw=1.3,
                    label="Log-Log Fit (slope={:.2f})".format(f["slope"]))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(x[1::2] if var == "n" else x[::2]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: "{:.0f}".format(v)))
        lo = math.floor(math.log(min(s["wall_q25"]), 4))
        hi = math.ceil(math.log(max(s["wall_q75"]), 4))
        ax.yaxis.set_major_locator(FixedLocator([4.0 ** e for e in range(lo, hi + 1)]))
        ax.yaxis.set_major_formatter(FuncFormatter(_time_label))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_locator(NullLocator())
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Computation Time (seconds)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11.5, loc="upper left", frameon=False)
        fig.savefig(os.path.join(FIGURE_DIR, name + suffix + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    with open(os.path.join(FIGURE_DIR, "summary_threads{}.json".format(threads)), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def report(summary: dict) -> None:
    for panel, s in summary.items():
        print("\n{} ({} varied, other = {}, {} runs per point, fallbacks = {}, max relative FOC residual = {:.1e})"
              .format(panel, s["varied"], s["fixed"], s["reps"], s["fallbacks"], s["foc_rel_max"]))
        print("{:>7s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s}".format(
            s["varied"], "median s", "q25 s", "q75 s", "MOSEK s", "iters"))
        for j, g in enumerate(s["x"]):
            print("{:7d} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:6.0f}".format(
                g, s["wall_median"][j], s["wall_q25"][j], s["wall_q75"][j], s["solver_median"][j],
                s["iters_median"][j]))
        print("slope (wall) {:.2f} over the full range, {:.2f} from {} on; slope (MOSEK) {:.2f}, {:.2f}".format(
            s["slope_wall"], s["slope_wall_upper"], s["upper_from"], s["slope_solver"], s["slope_solver_upper"]))
        print("plotted fits: " + "; ".join("{} to {}: {:.2f} (MOSEK {:.2f})".format(
            f["lo"], f["hi"], f["slope"], f["slope_solver"]) for f in s["fits"]))
        print("local slopes (wall): " + " ".join("{:.2f}".format(v) for v in s["local_slope_wall"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Timing of the exact MLE solver for Figure 2.")
    ap.add_argument("--threads", type=int, default=1, help="MOSEK threads (0 = MOSEK's default)")
    ap.add_argument("--reps", type=int, default=N_REPS, help="data sets per grid point")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        run(args.threads, args.reps)
    report(plot(args.threads))


if __name__ == "__main__":
    main()
