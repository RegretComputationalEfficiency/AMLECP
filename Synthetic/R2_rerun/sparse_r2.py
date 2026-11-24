"""Sparse-parameter experiment for the R2 revision (Appendix A.5, Proposition A.10).

Data: the heavy-tailed Case II design of the q-sensitivity study (design ``heavy100`` of env_r2.py), with a
configurable number of features d and horizon T:
  * theta0 = (50, 20, 0.3, 0.2, 0.1, 0, ..., 0) in R^d, price coefficient gamma0 = 0.1 (prices in units of
    100 in the estimation problem, so the fitted price coefficient is 10), parameter dimension kappa = d + 1;
  * heavy-tailed features (l4_heavy_tail_sample with mean shift 0.3), prices on [0, 211.8]
    (501 decision prices, 11 exploration prices), SNR = 1 calibrated as in env_r2 (per d);
  * Algorithm 1 schedule with m_k = m for k >= 1 and m_0 = max(m, s): m_0 = s = 6 and m_k = 3 when s is known,
    m_0 = m_k = 27 otherwise. Noise and exploration draws common to all methods.

Methods (one refit at the end of each exploration block, as in run_r2.py):
  * dense : AMLE-PGD with a fixed q (parameter "q", default 4), p = q/(q - 1), eta_k = eta0 kappa^(1/q - 1/2)
            sqrt(N), eps_k = eps0 kappa^(1/4) sqrt(N) (the same eps_k for every q);
  * sparse: the configuration of Proposition A.10. q_k = 1 + log(kappa N), p_k = q_k/(q_k - 1), two passes of
            AMLE-PGD with eta_k(W) = c_eta sqrt(N rho_k)/W, first W = W0, then W = (2/3)||v^(0)||_1 started at
            v^(0) (W is never taken below W0), the same eps_k as dense;
  * lasso : ILQX of Section 5.1 (Lasso solved exactly by MOSEK) with lambda_t = lambda0 t^(1/4)
            sqrt(log kappa + log t), t = current period. With m fixed, the tuned lambda0 absorbs the factor
            sqrt(m), so this is the Lasso estimator of Ban et al. (2026) under the exploration schedule above.

Each job (method, m, hyper-parameter, trial k) writes one .npz with per-period pseudo-regret inputs and the
per-refit logs.  Existing outputs are skipped.

    python3 sparse_r2.py --d 729 --T 50000 --tag sparse_d729 --grid-json grid_sparse_d729_m27.json --m 27 --k 1 2
    bash queue_sparse_ec1.sh            # all runs of Figure EC.1(b)-(c) and Appendix B (kappa = 730)
"""

from __future__ import annotations

import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import math
import multiprocessing as mp
import time
import traceback
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LINEAR_DIR = os.path.dirname(HERE)
for _p in (HERE, LINEAR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import solvers_r2 as S                                                # noqa: E402
from truncate_normal_sample_time_varying import l4_heavy_tail_sample   # noqa: E402

NOISE_SEED = 20261017          # as in env_r2
N_CALIB = 5000
L4_MEAN_SHIFT = 0.3
GAMMA0 = 0.1
PRICE_UNIT = 100.0
W_L1 = 1.0e4                   # numerical safeguard of MOSEK problems (never active)
W0 = 10.0                      # known lower bound on ||v||_1: the price coefficient gamma0 * PRICE_UNIT
PGD_INIT = 0.1
PGD_MAX_ITER = 20_000
METHODS = ("dense", "sparse", "lasso")


# ---------------------------------------------------------------------------
# Environment (Case II data with d features)
# ---------------------------------------------------------------------------
class Env:
    def __init__(self, d: int):
        self.d = int(d)
        self.kappa = self.d + 1
        self.theta0 = np.zeros(self.d)
        self.theta0[:5] = (50.0, 20.0, 0.3, 0.2, 0.1)
        p_hi = L4_MEAN_SHIFT * float(self.theta0.sum()) / GAMMA0
        self.price_grid = np.linspace(0.0, p_hi, 501)
        self.explore_grid = np.linspace(0.0, p_hi, 11)
        self.v_true = np.concatenate([self.theta0, [GAMMA0 * PRICE_UNIT]])

    def feature(self, k: int, t: int) -> np.ndarray:
        return l4_heavy_tail_sample(self.d, int(k), int(t), 0, mean_shift=L4_MEAN_SHIFT).astype(np.float64)

    @lru_cache(maxsize=1)
    def sigma(self) -> float:
        """SNR = 1: sigma^2 = ||theta0||^2 lambda_min(E[x x^T]), estimated as in env_r2 (k = 0, N_CALIB draws)."""
        Xc = np.empty((N_CALIB, self.d))
        for i in range(N_CALIB):
            Xc[i] = self.feature(0, i)
        lam_min = max(float(np.linalg.eigvalsh(Xc.T @ Xc / N_CALIB)[0]), 0.0)
        return float(math.sqrt(float(self.theta0 @ self.theta0) * lam_min))

    def exploit_price(self, v_hat: np.ndarray, x: np.ndarray) -> float:
        d_hat = float(v_hat[:self.d] @ x) - float(v_hat[self.d]) / PRICE_UNIT * self.price_grid
        return float(self.price_grid[int(np.argmax(self.price_grid * np.maximum(d_hat, 0.0)))])


def trial_draws(k: int, T: int, n_explore_prices: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([NOISE_SEED, int(k)])
    return rng.standard_normal(int(T)), rng.integers(0, n_explore_prices, size=int(T))


def schedule(T: int, m0: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 with M_0 = [1, m0] and M_k = [k^2, k^2 + m - 1] for k >= 1 (same rule as env_r2.schedule).

    The estimate is refit at the last period of each block, from all exploration data so far. With m0 = 6 and
    m = 3, M_1 = [1, 3] and M_2 = [4, 6] cover M_0, so the exploration periods and refit times are those of
    m0 = m = 3. The refit at period 3 is used only as the starting point of the refit at period 6, because
    periods 1-6 are all exploration periods.
    """
    is_explore = np.zeros(T, dtype=bool)
    refit_at = np.zeros(T, dtype=bool)
    is_explore[:m0] = True
    ends = {m0}
    k = 1
    while k * k <= T:
        start, end = k * k, min(k * k + m - 1, T)
        is_explore[start - 1:end] = True
        ends.add(end)
        k += 1
    for end in ends:
        if end < T:
            refit_at[end - 1] = True
    return is_explore, refit_at


def expected_positive_part(mu, sigma: float):
    from scipy.special import erf
    mu = np.asarray(mu, dtype=np.float64)
    z = mu / sigma
    return mu * 0.5 * (1.0 + erf(z / math.sqrt(2.0))) + sigma * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# One online run
# ---------------------------------------------------------------------------
def run_one(d: int, method: str, m: int, param: dict, k: int, T: int) -> dict:
    env = Env(d)
    kappa = env.kappa
    sigma = env.sigma()
    z, eidx = trial_draws(k, T, env.explore_grid.size)
    m0 = initial_length(m)
    is_explore, refit_at = schedule(T, m0, m)
    n_exp = int(is_explore.sum())
    A = np.empty((n_exp, kappa))
    yv = np.empty(n_exp)
    N = 0
    price = np.empty(T)
    mu_all = np.empty(T)
    log = {key: [] for key in ("t", "N", "time", "iters", "stop", "G", "eps", "p", "eta", "W")}
    v_hat = None
    u_pow = None
    L_prev = None                       # final backtracking constant of the previous fit
    eps_coef = float(param.get("eps", 3.0))

    for t in range(T):
        x = env.feature(k, t)
        mu = float(env.theta0 @ x)
        mu_all[t] = mu
        p = float(env.explore_grid[eidx[t]]) if is_explore[t] else env.exploit_price(v_hat, x)
        price[t] = p
        dem = max(mu - GAMMA0 * p + sigma * float(z[t]), 0.0)
        if is_explore[t]:
            A[N, :d] = x
            A[N, d] = -p / PRICE_UNIT
            yv[N] = dem
            N += 1
        if not refit_at[t]:
            continue
        An, yn = A[:N], yv[:N]
        t0 = time.perf_counter()
        info = {"iters": 0, "stop": "", "G": np.nan, "eps": np.nan, "p": np.nan, "eta": np.nan, "W": np.nan}
        if method in ("dense", "sparse"):
            # tolerance: param "epsd" = 1 gives eps0 sqrt(s N) (Theorem 2), 2 gives eps0 sqrt(kappa N) (eq. (7));
            # without it, the earlier eps0 kappa^(1/4) sqrt(N)
            epsd = int(param.get("epsd", 0))
            if epsd == 1:
                eps = eps_coef * math.sqrt(S_NONZERO * N)
            elif epsd == 2:
                eps = eps_coef * math.sqrt(kappa * N)
            else:
                eps = eps_coef * kappa ** 0.25 * math.sqrt(N)
            lam_max, u_pow = S.power_iteration(An, u_pow)
            L = 1.02 * lam_max
            v0 = np.full(kappa, PGD_INIT) if v_hat is None else v_hat
            if method == "dense":
                q_reg = float(param.get("q", 4.0))          # default q = 4 (Section 5.1)
                p_reg = q_reg / (q_reg - 1.0)
                eta = float(param["eta"]) * kappa ** (1.0 / q_reg - 0.5) * math.sqrt(N)
                res = S.solve_pgd(An, yn, eta, eps, v0, L, p_reg, PGD_MAX_ITER, L_prev)
                v_hat, L_prev = res["v"], res["L"]
                info.update(iters=res["iters"], stop=res["stop"], G=res["G"], eps=eps, p=p_reg, eta=eta)
            else:
                rho = math.log(kappa * N)
                p_reg = 1.0 + 1.0 / rho
                c_eta = float(param["ceta"])
                eta1 = c_eta * math.sqrt(N * rho) / W0
                res1 = S.solve_pgd(An, yn, eta1, eps, v0, L, p_reg, PGD_MAX_ITER, L_prev)
                W = max(2.0 / 3.0 * float(np.abs(res1["v"]).sum()), W0)
                eta2 = c_eta * math.sqrt(N * rho) / W
                res2 = S.solve_pgd(An, yn, eta2, eps, res1["v"], L, p_reg, PGD_MAX_ITER, res1["L"])
                v_hat, L_prev = res2["v"], res2["L"]
                stop = res1["stop"] if res1["stop"] != "cert" else res2["stop"]
                info.update(iters=res1["iters"] + res2["iters"], stop=stop, G=res2["G"], eps=eps, p=p_reg,
                            eta=eta2, W=W)
        elif method == "lasso":
            lam = float(param["lam"]) * (t + 1) ** 0.25 * math.sqrt(math.log(kappa) + math.log(t + 2))
            v_hat, used = S.ilqx_lasso(An, yn, lam, W_L1)
            info.update(stop=used, eta=lam)
        else:
            raise ValueError(method)
        elapsed = time.perf_counter() - t0
        log["t"].append(t)
        log["N"].append(N)
        log["time"].append(elapsed)
        for key in ("iters", "stop", "G", "eps", "p", "eta", "W"):
            log[key].append(info[key])

    rev_exp = price * expected_positive_part(mu_all - GAMMA0 * price, sigma)
    oracle = np.empty(T)
    for s in range(0, T, 5000):          # chunked to bound memory
        mu_c = mu_all[s:s + 5000]
        grid_rev = env.price_grid[None, :] * expected_positive_part(mu_c[:, None] - GAMMA0 * env.price_grid[None, :],
                                                                    sigma)
        oracle[s:s + 5000] = grid_rev.max(axis=1)
    return {
        "price": price, "rev_exp": rev_exp, "oracle": oracle, "is_explore": is_explore,
        "fit_t": np.array(log["t"]), "fit_N": np.array(log["N"]), "fit_time": np.array(log["time"]),
        "fit_iters": np.array(log["iters"]), "fit_stop": np.array(log["stop"]),
        "fit_G": np.array(log["G"], dtype=float), "fit_eps": np.array(log["eps"], dtype=float),
        "fit_p": np.array(log["p"], dtype=float), "fit_eta": np.array(log["eta"], dtype=float),
        "fit_W": np.array(log["W"], dtype=float), "v_final": v_hat,
        "err_final": float(np.linalg.norm(v_hat - env.v_true)) if v_hat is not None else np.nan,
        "total_time": float(np.sum(log["time"])),
        "meta": json.dumps({"d": d, "kappa": kappa, "method": method, "m0": m0, "m": m, "param": param, "k": k, "T": T,
                            "sigma": sigma, "W0": W0, "pgd_solver": S.PGD_SOLVER}),
    }


S_NONZERO = 6   # nonzero entries of (theta0, price coefficient)


def initial_length(m: int) -> int:
    """m_0 = s = 6 for m_k = 3 (s known) and m_0 = m_k for m_k = 27 (s unknown)."""
    return max(int(m), S_NONZERO)


def param_str(param: dict) -> str:
    return "_".join("{}={:g}".format(k, v) for k, v in sorted(param.items()))


def out_path(root: str, tag: str, method: str, m: int, param: dict, k: int) -> str:
    return os.path.join(root, tag, "{}_m{}".format(method, m), param_str(param), "k{}.npz".format(k))


def _job(args: tuple) -> tuple[str, str, float]:
    root, tag, d, method, m, param, k, T = args
    path = out_path(root, tag, method, m, param, k)
    if os.path.exists(path):
        return path, "skip", 0.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.perf_counter()
    try:
        res = run_one(d, method, m, param, k, T)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **res)
        os.replace(tmp, path)
        return path, "ok", time.perf_counter() - t0
    except Exception:
        with open(path[:-4] + ".err.txt", "w") as f:
            f.write(traceback.format_exc())
        return path, "error", time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--T", type=int, default=50_000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid-json", required=True, help='{"dense": [params], "sparse": [...], "lasso": [...]}')
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--m", nargs="+", type=int, default=None, help="m_k for k >= 1, default: floor(sqrt(kappa)) and 3; m_0 = max(m_k, s)")
    ap.add_argument("--k", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--root", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    kappa = args.d + 1
    ms = args.m if args.m else [int(math.floor(math.sqrt(kappa))), 3]
    tag_dir = os.path.join(args.root, args.tag)
    os.makedirs(tag_dir, exist_ok=True)
    marker = os.path.join(tag_dir, "DESIGN")
    design = "sparse_heavy100_d{}_T{}".format(args.d, args.T)
    if os.path.exists(marker):
        with open(marker) as f:
            if f.read().strip() != design:
                raise SystemExit("{} holds another design".format(tag_dir))
    else:
        with open(marker, "w") as f:
            f.write(design + "\n")
    with open(args.grid_json) as f:
        grid = json.load(f)
    order = {"lasso": 0, "sparse": 1, "dense": 2}
    jobs = []
    for method in sorted(args.methods, key=lambda s: order[s]):
        for m, param, k in itertools.product(ms, grid.get(method, []), args.k):
            jobs.append((args.root, args.tag, args.d, method, m, param, k, args.T))
    print("{} jobs, {} workers, (m_0, m_k) in {}".format(len(jobs), args.workers, [(initial_length(m), m) for m in ms]),
          flush=True)
    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
        for done, (path, status, el) in enumerate(pool.imap_unordered(_job, jobs), 1):
            print("[{}/{}] {:5s} {:8.1f}s  {}  (elapsed {:.1f} min)".format(
                done, len(jobs), status, el, os.path.relpath(path, args.root), (time.time() - t_start) / 60),
                flush=True)


if __name__ == "__main__":
    main()
