"""Timing sanity check for the R2 rerun.

For every (SNR, method, hyper-parameter), each update's wall-clock time is compared with the median over
the other trials k at the same update index (same data size |I_k|):
  * AMLE-MOSEK / ILQX: only updates solved by MOSEK at the first attempt are compared (fallback solves --
    MOSEK on the scaled objective, then CLARABEL -- are genuinely slower and are reported separately);
  * AMLE-PGD: the time per iteration is compared (iteration counts legitimately differ across trials);
  * SPO+: the time per update; AdaGrad: the total recursive-update time.
An update is flagged when it exceeds FACTOR times the median and the median by MIN_EXCESS seconds, e.g.
a job that kept running while the machine slept.  With --delete, flagged runs are removed so that
run_r2.py recomputes them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

HERE = os.path.dirname(os.path.abspath(__file__))
FACTOR, MIN_EXCESS = 5.0, 5.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(HERE, "results", args.tag, "*", "*", "*", "k*.npz")):
        if path.endswith(".tmp.npz"):
            continue
        r = np.load(path)
        meta = json.loads(str(r["meta"]))
        groups[(meta["snr"], meta["method"], json.dumps(meta["param"], sort_keys=True))].append(
            (path, r["fit_time"].astype(float), r["fit_iters"].astype(float), r["fit_stop"].astype(str),
             float(r["total_time"])))
    flagged, n_runs = [], 0
    fallback_share = defaultdict(lambda: [0.0, 0.0])       # (snr, method) -> [fallback time, total time]
    for key, runs in sorted(groups.items()):
        snr, method = key[0], key[1]
        n_runs += len(runs)
        if method == "AdaGrad":
            tot = np.array([r[4] for r in runs])
            med = np.median(tot)
            for path, *_, t in runs:
                if t > FACTOR * med and t - med > MIN_EXCESS:
                    flagged.append((path, "total {:.1f}s vs median {:.1f}s".format(t, med)))
            continue
        if len({len(r[1]) for r in runs}) != 1:
            flagged += [(r[0], "unexpected number of updates") for r in runs]
            continue
        T = np.vstack([r[1] for r in runs])
        S = np.vstack([r[3] for r in runs])
        if method in ("MOSEK", "ILQX"):
            fallback_share[(snr, method)][0] += float(T[S != "mosek"].sum())
            fallback_share[(snr, method)][1] += float(T.sum())
            V = np.where(S == "mosek", T, np.nan)
        elif method == "PGD":
            V = T / np.maximum(np.vstack([r[2] for r in runs]), 1.0)
        else:
            V = T
        with np.errstate(all="ignore"):
            med = np.nanmedian(V, axis=0)
        for r, row in zip(runs, V):
            excess = row - med if method != "PGD" else (row - med) * np.maximum(r[2], 1.0)
            bad = np.nonzero((row > FACTOR * med) & (excess > MIN_EXCESS))[0]
            if bad.size:
                i = int(bad[np.argmax(excess[bad])])
                flagged.append((r[0], "update {}: {:.3g} vs median {:.3g} ({})".format(
                    i, row[i], med[i], "s per iteration" if method == "PGD" else "s")))
    print("checked {} runs in {} configurations; flagged {}".format(n_runs, len(groups), len(flagged)))
    for path, why in flagged:
        print("  ", os.path.relpath(path, HERE), "--", why)
    print("share of solver time spent in fallback solves (MOSEK stalled):")
    for (snr, method), (fb, tot) in sorted(fallback_share.items()):
        print("   {:5s} {:6s} {:5.1f}%".format(snr, method, 100.0 * fb / tot if tot else 0.0))
    if args.delete and flagged:
        for path, _ in flagged:
            os.remove(path)
        print("deleted {} flagged runs; rerun run_r2.py with the same arguments to recompute them".format(len(flagged)))


if __name__ == "__main__":
    main()
