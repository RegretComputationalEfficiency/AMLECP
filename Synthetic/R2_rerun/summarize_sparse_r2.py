"""Summary of the sparse-parameter runs of sparse_r2.py.

    python3 summarize_sparse_r2.py --tag sparse_d200

For every (method, m, hyper-parameter): number of trials, mean (and standard error) of the final regret, of the
exploitation regret and of the final estimation error ||v_hat - v||_2, mean optimization time, mean PGD
iterations per refit and the number of refits that hit the iteration cap.  Writes results/<tag>/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    base = os.path.join(args.root, args.tag)
    rows = []
    for cfg_dir in sorted(glob.glob(os.path.join(base, "*_m*", "*"))):
        files = sorted(glob.glob(os.path.join(cfg_dir, "k*.npz")))
        files = [f for f in files if not f.endswith(".tmp.npz")]
        if not files:
            continue
        reg, expl, err, tim, its, capped, nfit, Tval = [], [], [], [], [], [], [], None
        for f in files:
            r = np.load(f, allow_pickle=False)
            gap = r["oracle"] - r["rev_exp"]
            reg.append(float(gap.sum()))
            expl.append(float(gap[~r["is_explore"]].sum()))
            err.append(float(r["err_final"]))
            tim.append(float(r["total_time"]))
            it = r["fit_iters"].astype(float)
            its.append(float(it.mean()) if it.size else np.nan)
            capped.append(int(np.sum(r["fit_stop"] == "maxiter")))
            nfit.append(int(r["fit_N"].size))
            Tval = json.loads(str(r["meta"]))["T"]
        n = len(files)
        se = lambda v: float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({
            "config": os.path.basename(os.path.dirname(cfg_dir)), "param": os.path.basename(cfg_dir), "K": n,
            "T": Tval, "regret": float(np.mean(reg)), "regret_se": se(reg),
            "exploit_regret": float(np.mean(expl)), "exploit_se": se(expl),
            "err_final": float(np.mean(err)), "time_s": float(np.mean(tim)),
            "iters_per_fit": float(np.nanmean(its)), "capped_fits": int(np.sum(capped)), "fits": int(np.mean(nfit)),
        })
    if not rows:
        raise SystemExit("no results under " + base)
    out = os.path.join(base, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fmt = "{:<12s} {:<22s} {:>3s} {:>12s} {:>10s} {:>12s} {:>9s} {:>9s} {:>9s} {:>6s}"
    print(fmt.format("config", "param", "K", "regret", "se", "exploit", "err", "time_s", "iters", "capped"))
    for r in rows:
        print("{:<12s} {:<22s} {:>3d} {:>12.0f} {:>10.0f} {:>12.0f} {:>9.3f} {:>9.1f} {:>9.1f} {:>6d}".format(
            r["config"], r["param"], r["K"], r["regret"], r["regret_se"], r["exploit_regret"], r["err_final"],
            r["time_s"], r["iters_per_fit"], r["capped_fits"]))


if __name__ == "__main__":
    main()
