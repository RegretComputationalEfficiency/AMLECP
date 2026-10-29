"""Main experiment script for the R2 rerun of Section 5.1 (Case Study I).

Each job = (SNR tier, method, hyper-parameter, trial k) and writes one compressed
``.npz`` file, so runs can be interrupted and resumed (existing outputs are skipped).

Examples
--------
    python run_r2.py --tag pilot --snr mid --methods MOSEK --grid pilot --k 1 2 --workers 8
    python run_r2.py --tag main  --snr low mid high --methods all --grid main --workers 8
    python run_r2.py --tag smoke --snr mid --methods all --grid smoke --k 1 --T 600 --workers 5
    python run_r2.py --design dense --tag dense --snr low --methods MOSEK --grid main --grid-json grid.json

Each results/<tag> folder holds one design (recorded in results/<tag>/DESIGN; mixing is refused).
"""

from __future__ import annotations

import os

import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _design_from_argv() -> str:
    """--design must be known before env_r2 is imported (it fixes theta0); spawned workers inherit it."""
    for i, a in enumerate(sys.argv):
        if a == "--design" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--design="):
            return a.split("=", 1)[1]
    return os.environ.get("R2_DESIGN", "base")


os.environ["R2_DESIGN"] = _design_from_argv()

import argparse
import itertools
import json
import multiprocessing as mp
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import env_r2 as env          # noqa: E402
import solvers_r2 as S        # noqa: E402

METHODS = ["PGD", "MOSEK", "ILQX", "AdaGrad", "SPO+"]

# Hyper-parameter grids.  "main" is fixed after the pilot (see README.md).
GRIDS = {
    "smoke": {
        "PGD": [{"eta": 5.0, "eps": e} for e in (10.0, 1000.0)],
        "MOSEK": [{"eta": 5.0}],
        "ILQX": [{"lam": 5.0}],
        "AdaGrad": [{"lr": 1.0}],
        "SPO+": [{"lr": 5e-4}],
    },
    "pilot": {
        "MOSEK": [{"eta": e} for e in (0.05, 0.5, 5.0, 50.0, 500.0)],
        "ILQX": [{"lam": l} for l in (0.05, 5.0, 500.0, 5e4)],
        "PGD": [],   # filled after the MOSEK pilot (best eta) -- see --pgd-eta / --pgd-eps
        "AdaGrad": [{"lr": l} for l in (0.01, 0.1, 1.0, 10.0)],
        "SPO+": [{"lr": 5e-4}],
    },
    "main": {   # placeholders; overwritten by --grid-json
        "PGD": [], "MOSEK": [], "ILQX": [], "AdaGrad": [], "SPO+": [],
    },
}
SPO_EPOCHS, SPO_L2, SPO_HIDDEN = 60, 1e-5, 64
PGD_INIT = 0.1          # initial point of the first fit (as in R1)
PGD_MAX_ITER = 20_000   # per-fit iteration cap (hitting it is recorded)


def param_str(param: dict) -> str:
    return "_".join("{}={:g}".format(k, v) for k, v in param.items())


def out_path(root: str, tag: str, snr: str, method: str, param: dict, k: int) -> str:
    return os.path.join(root, tag, snr, method, param_str(param), "k{}.npz".format(k))


# ---------------------------------------------------------------------------
# One online run
# ---------------------------------------------------------------------------
def run_one(snr: str, method: str, param: dict, k_inst: int, T: int) -> dict:
    sigma = env.noise_scale(snr)
    # q (feature norm of Assumption 2) sets the regularizer exponent p = q/(q-1) and the dimension factors of
    # eta_k and eps_k; default q = 4 (Section 5.1). The exploration schedule is the same for every q.
    # eps_q, if given, fixes the dimension factor of eps_k (kappa^(1/2 - 1/eps_q)) independently of q, so that
    # different q are compared at the same tolerance.
    q_norm = float(param.get("q", env.Q_NORM))
    eps_q = float(param.get("eps_q", q_norm))
    p_reg = env.p_of_q(q_norm)
    X = env.feature_trajectory(k_inst, T)
    mu_x = X @ env.THETA0
    z, eidx = env.trial_draws(k_inst, T)
    is_explore, refit_at = env.schedule(T)
    n_exp = int(is_explore.sum())

    A = np.empty((n_exp, env.KAPPA), dtype=np.float64)
    yv = np.empty(n_exp, dtype=np.float64)
    N = 0
    price = np.empty(T, dtype=np.float64)
    dem = np.empty(T, dtype=np.float64)
    log = {"t": [], "N": [], "time": [], "iters": [], "stop": [], "G": [], "foc": [], "eps": []}
    upd_time = 0.0                      # AdaGrad: total recursive update time

    v_hat: np.ndarray | None = None
    net = None
    u_pow = None
    ada = S.AdaGradOnline(env.KAPPA, env.W_L1, lr=param["lr"]) if method == "AdaGrad" else None
    n_fit = 0

    for t in range(T):
        x = X[t]
        if is_explore[t]:
            p = float(env.EXPLORE_GRID[eidx[t]])
        elif method == "SPO+":
            import torch
            with torch.no_grad():
                ch = net(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
            p = float(env.PRICE_GRID[int(torch.argmax(ch, dim=1).item())])
        else:
            p = env.exploit_price(v_hat, x)
        d = max(float(mu_x[t]) - env.GAMMA0 * p + sigma * float(z[t]), 0.0)
        price[t], dem[t] = p, d
        a = np.empty(env.KAPPA, dtype=np.float64)
        a[:env.D_FEAT] = x
        a[env.D_FEAT] = -p / env.PRICE_UNIT

        if method == "AdaGrad":
            upd_time += ada.update(a, d)
            v_hat = ada.v
            continue

        if is_explore[t]:
            A[N] = a
            yv[N] = d
            N += 1
        if not refit_at[t]:
            continue

        An, yn = A[:N], yv[:N]
        info = {"iters": 0, "stop": "", "G": np.nan, "foc": np.nan, "eps": np.nan}
        t0 = time.perf_counter()
        if method == "PGD":
            eta = env.eta_k(param["eta"], N, q_norm)
            eps = env.eps_k(param["eps"], N, eps_q)
            lam_max, u_pow = S.power_iteration(An, u_pow)
            v0 = np.full(env.KAPPA, PGD_INIT) if v_hat is None else v_hat
            res = S.amle_pgd(An, yn, eta, eps, v0, 1.02 * lam_max, p_reg, PGD_MAX_ITER)
            v_hat = res["v"]
            info.update(iters=res["iters"], stop=res["stop"], G=res["G"], eps=eps)
        elif method == "MOSEK":
            eta = env.eta_k(param["eta"], N, q_norm)
            v_hat, used = S.amle_mosek(An, yn, eta, p_reg, env.W_L1)
            info["stop"] = used
        elif method == "ILQX":
            v_hat, used = S.ilqx_lasso(An, yn, env.lasso_lambda(param["lam"], t + 1), env.W_L1)  # period t + 1
            info["stop"] = used
        elif method == "SPO+":
            Xh, ph = An[:, :env.D_FEAT], -An[:, env.D_FEAT] * env.PRICE_UNIT
            c = S.dr_labels(Xh, ph, yn, env.PRICE_GRID, env.EXPLORE_GRID)
            net = S.train_spoplus(Xh, c, param["lr"], SPO_EPOCHS, SPO_L2, SPO_HIDDEN,
                                  seed=1_000_003 * k_inst + n_fit)
        elapsed = time.perf_counter() - t0
        if method in ("PGD", "MOSEK"):
            info["foc"] = S.foc_residual(An, yn, v_hat, env.eta_k(param["eta"], N, q_norm), p_reg)
        n_fit += 1
        log["t"].append(t)
        log["N"].append(N)
        log["time"].append(elapsed)
        for key in ("iters", "stop", "G", "foc", "eps"):
            log[key].append(info[key])

    rev_exp = price * env.expected_positive_part(mu_x - env.GAMMA0 * price, sigma)
    grid_rev = env.PRICE_GRID[None, :] * env.expected_positive_part(
        mu_x[:, None] - env.GAMMA0 * env.PRICE_GRID[None, :], sigma)
    oracle = grid_rev.max(axis=1)
    total_time = float(np.sum(log["time"])) if method != "AdaGrad" else upd_time
    return {
        "price": price, "demand": dem, "rev_exp": rev_exp, "rev_real": price * dem, "oracle": oracle,
        "fit_t": np.array(log["t"]), "fit_N": np.array(log["N"]), "fit_time": np.array(log["time"]),
        "fit_iters": np.array(log["iters"]), "fit_stop": np.array(log["stop"]),
        "fit_G": np.array(log["G"], dtype=float), "fit_foc": np.array(log["foc"], dtype=float),
        "fit_eps": np.array(log["eps"], dtype=float),
        "total_time": total_time,
        "meta": json.dumps({"design": env.DESIGN, "snr": snr, "method": method, "param": param, "k": k_inst, "T": T,
                            "sigma": sigma, "m": env.M_EXPLORE, "kappa": env.KAPPA,
                            "q": q_norm, "p_reg": p_reg, "eps_q": eps_q, "price_unit": env.PRICE_UNIT}),
    }


def _job(args: tuple) -> tuple[str, str, float]:
    root, tag, snr, method, param, k, T = args
    path = out_path(root, tag, snr, method, param, k)
    if os.path.exists(path):
        return path, "skip", 0.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.perf_counter()
    try:
        res = run_one(snr, method, param, k, T)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **res)
        os.replace(tmp, path)
        return path, "ok", time.perf_counter() - t0
    except Exception:  # keep the pool alive; record the failure next to the output
        with open(path[:-4] + ".err.txt", "w") as f:
            f.write(traceback.format_exc())
        return path, "error", time.perf_counter() - t0


def check_design_marker(tag_dir: str) -> None:
    """One design per results folder.  Folders written before designs existed hold the base design."""
    marker = os.path.join(tag_dir, "DESIGN")
    os.makedirs(tag_dir, exist_ok=True)
    if not os.path.exists(marker):
        existing = [d for d in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, d))]
        with open(marker, "w") as f:
            f.write(("base" if existing else env.DESIGN) + "\n")
    with open(marker) as f:
        found = f.read().strip()
    if found != env.DESIGN:
        raise SystemExit("{} holds design {!r}; refusing to add design {!r} (use another --tag)".format(
            tag_dir, found, env.DESIGN))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--snr", nargs="+", default=["low", "mid", "high"])
    ap.add_argument("--methods", nargs="+", default=["all"])
    ap.add_argument("--grid", default="main")
    ap.add_argument("--grid-json", default=None, help="JSON file overriding the grid {method: [params]}")
    ap.add_argument("--pgd-eta", nargs="*", type=float, default=None)
    ap.add_argument("--pgd-eps", nargs="*", type=float, default=None)
    ap.add_argument("--k", nargs="+", type=int, default=list(range(1, env.K_TRIALS + 1)))
    ap.add_argument("--T", type=int, default=env.T_HORIZON)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--root", default=os.path.join(HERE, "results"))
    ap.add_argument("--design", default=env.DESIGN, choices=env.DESIGNS)
    args = ap.parse_args()
    assert args.design == env.DESIGN
    check_design_marker(os.path.join(args.root, args.tag))

    grid = {m: list(v) for m, v in GRIDS[args.grid].items()}
    by_snr = None      # optional per-SNR grids: {"by_snr": {snr: {method: [params]}}}
    if args.grid_json:
        with open(args.grid_json) as f:
            spec = json.load(f)
        if "by_snr" in spec:
            by_snr = spec["by_snr"]
        else:
            grid.update(spec)
    if args.pgd_eta is not None and args.pgd_eps is not None:
        grid["PGD"] = [{"eta": a, "eps": b} for a in args.pgd_eta for b in args.pgd_eps]
    methods = METHODS if args.methods == ["all"] else args.methods

    jobs = []
    # Put the slow methods first so that the pool stays busy.
    order = {"SPO+": 0, "MOSEK": 1, "ILQX": 2, "PGD": 3, "AdaGrad": 4}
    for method in sorted(methods, key=lambda m: order[m]):
        for snr in args.snr:
            params = by_snr[snr].get(method, []) if by_snr is not None else grid.get(method, [])
            for param, k in itertools.product(params, args.k):
                jobs.append((args.root, args.tag, snr, method, param, k, args.T))
    print("{} jobs, {} workers".format(len(jobs), args.workers), flush=True)
    t_start = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=8) as pool:
        for path, status, el in pool.imap_unordered(_job, jobs):
            done += 1
            print("[{}/{}] {:5s} {:7.1f}s  {}  (elapsed {:.1f} min)".format(
                done, len(jobs), status, el, os.path.relpath(path, args.root),
                (time.time() - t_start) / 60), flush=True)


if __name__ == "__main__":
    main()
