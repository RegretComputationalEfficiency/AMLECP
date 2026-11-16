"""Figure EC.1 of the revised paper, with the sparse-parameter Case II of sparse_r2.py.

    python3 plot_qsens_sparse_r2.py      # needs results/dense_alg1, qsens_alg1, qsens_tol_alg1 and sparse_d729

(a) Case I : dense design of Section 5.1, T = 10,000, configurations of plot_qsens_r2.select. The regrets of all
             configurations agree within about 1%, so the panel shows the paired relative regret
             R_k(t) / R_k^ref(t) - 1 (mean over trials k and 95% band), with AMLE-MOSEK (q = 2) as reference.
(b) Case II: heavy100 design with d = 729 (kappa = 730), T = 50,000, m_0 = m_k = 27 = floor(sqrt(kappa)).
             AMLE-PGD with fixed q in {2, 4, 1 + log kappa} (dense eta scaling), AMLE-PGD with the exponent
             p_k = 1 + 1/log(kappa |I_k|) and the two-pass eta_k(W) of Appendix A.5, and ILQX (MOSEK).
(c) Case II: the same with m_0 = m_k = 3 = ceil(sqrt(s)), s = 6.
Tolerances: m_k = 27 (s unknown) uses eps_k = eps0 sqrt(kappa |I_k|), m_k = 3 (s known) uses eps_k = eps0 sqrt(s |I_k|)
(runs with param epsd = 2 and 1 of sparse_r2.py). Selection as in Section 5.1: the (eta0 or c_eta, eps0) with the
minimum mean final regret over 10 trials, then on that eta0 (c_eta) the largest eps0 whose mean final regret is
within 1% of the minimum. ILQX: lambda0 with the minimum mean final regret.
Output: figures/qsens_sparse/fig_q_sensitivity_sparse.pdf and summary_sparse_case2.csv.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

import plot_r2 as P
import plot_qsens_r2 as Q
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.ticker import FuncFormatter   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "results", "sparse_d729")
Q_LOG = Q.Q_LOG
# validated categorical palette, fixed order: slots 1-3 as in plot_qsens_r2, slot 4 ILQX, slot 5 p_k
COLOR = {"q4": "#2a78d6", "q2": "#eb6834", "qlog": "#1baf7a", "lasso": "#eda100", "pk": "#e87ba4"}
ORDER = ["q2", "q4", "qlog", "pk", "lasso"]


def series_of(method: str, param: dict) -> str:
    if method == "sparse":
        return "pk"
    if method == "lasso":
        return "lasso"
    q = float(param.get("q", 4.0))
    return {2.0: "q2", 4.0: "q4"}.get(round(q, 6), "qlog" if abs(q - Q_LOG) < 1e-6 else None)


EPSD = {27: 2, 3: 1}   # tolerance form per m (sparse_r2.py param "epsd")


def load_case2(m: int) -> dict:
    """series -> dict(param, K, curve_mean, curve_se, final list, time_mean) at the selected parameter."""
    rows = {}
    for cfg in glob.glob(os.path.join(BASE, "*_m{}".format(m), "*")):
        files = sorted(f for f in glob.glob(os.path.join(cfg, "k*.npz")) if not f.endswith(".tmp.npz"))
        if len(files) < 10:
            continue
        curves, times, param, method = [], [], None, None
        for f in files:
            r = np.load(f, allow_pickle=False)
            meta = json.loads(str(r["meta"]))
            param, method = meta["param"], meta["method"]
            curves.append(np.cumsum(r["oracle"] - r["rev_exp"]))
            times.append(float(r["total_time"]))
        s = series_of(method, param)
        if s is None or (s != "lasso" and int(param.get("epsd", 0)) != EPSD[m]):
            continue
        C = np.vstack(curves)
        row = {"param": param, "method": method, "K": C.shape[0], "curve_mean": C.mean(0),
               "curve_se": C.std(0, ddof=1) / math.sqrt(C.shape[0]), "final": C[:, -1],
               "time_mean": float(np.mean(times))}
        rows.setdefault(s, []).append(row)
    best = {}
    for s, R in rows.items():
        b = min(R, key=lambda r: r["curve_mean"][-1])
        if s == "lasso":
            best[s] = b
            continue
        key = "ceta" if s == "pk" else "eta"
        cand = [r for r in R if r["param"][key] == b["param"][key] and r["curve_mean"][-1] <= 1.01 * b["curve_mean"][-1]]
        best[s] = max(cand, key=lambda r: r["param"]["eps"])
    return best


MARK_SPAN = (0.35, 0.95)   # one marker per curve, staggered over this fraction of the horizon


def mark_idx(n: int, i: int, n_series: int) -> list[int]:
    """Index of the single marker of series i of n_series on a curve of n points."""
    lo, hi = MARK_SPAN
    return [min(n - 1, int(n * (lo + (hi - lo) * (i + 0.5) / n_series)))]


REL_T0 = 1000   # first period shown in panel (a); earlier ratios are dominated by the first exploration block


def panel1_relative(ax, sel: dict, tags: list[str], title: str) -> None:
    """Paired relative regret of the Case I configurations with respect to AMLE-MOSEK (q = 2)."""
    runs = {}
    for tag in tags:
        root = os.path.join(HERE, "results", tag)
        if os.path.isdir(root):
            runs.update(P.load(root))

    def curves(r):
        lst = runs[(r["snr"], r["method"], json.dumps(r["param"], sort_keys=True))]
        return {d["k"]: d["regret"] for d in lst}

    ref = curves(sel[("MOSEK", 2.0)])
    ax.axhline(0.0, color=Q.Q_COLOR[2.0], ls=Q.M_STYLE["MOSEK"]["ls"], lw=1.2)
    handles = [plt.Line2D([], [], color=Q.Q_COLOR[2.0], ls=Q.M_STYLE["MOSEK"]["ls"], lw=1.2,
                          label=Q.label("MOSEK", sel[("MOSEK", 2.0)]) + ", reference")]
    keys = [(m, q) for m in ("MOSEK", "PGD") for q in Q.Q_ORDER if (m, q) != ("MOSEK", 2.0) and (m, q) in sel]
    for i, (method, q) in enumerate(keys):
            r = sel[(method, q)]
            cur = curves(r)
            ks = sorted(set(cur) & set(ref))
            R = np.vstack([cur[k] / ref[k] - 1.0 for k in ks]) * 100.0
            T = R.shape[1]
            idx = np.unique(np.linspace(REL_T0 - 1, T - 1, 300).astype(int))
            t, mu = idx + 1, R[:, idx].mean(0)
            se = R[:, idx].std(0, ddof=1) / math.sqrt(len(ks))
            ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=Q.Q_COLOR[q], alpha=0.10, lw=0)
            ax.plot(t, mu, color=Q.Q_COLOR[q], ls=Q.M_STYLE[method]["ls"], lw=1.4)
            mi = mark_idx(t.size, i, len(keys))
            ax.plot(t[mi], mu[mi], marker=Q.M_STYLE[method]["marker"], ms=4.5, color=Q.Q_COLOR[q], mec="white",
                    mew=0.8, ls="none")
            handles.append(plt.Line2D([], [], color=Q.Q_COLOR[q], ls=Q.M_STYLE[method]["ls"], lw=1.4,
                                      marker=Q.M_STYLE[method]["marker"], ms=4, mec="white", mew=0.8,
                                      label=Q.label(method, r)))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Relative regret (%)")
    ax.set_xlim(0, None)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.9 * (hi - lo))      # room for the legend above the curves
    ax.xaxis.set_major_formatter(FuncFormatter(P.compact))
    P._style_axes(ax)
    ax.legend(handles=handles, loc="upper right", fontsize=4.6, ncol=2, handlelength=1.8, labelspacing=0.2,
              columnspacing=0.6, borderaxespad=0.2)


def label(s: str, row: dict) -> str:
    p = row["param"]
    if s == "pk":
        return r"AMLE-PGD ($p_k$, $c_\eta$={:g}, {}={:g})".format(p["ceta"], P.PARAM_SYMBOL["eps"], p["eps"])
    if s == "lasso":
        return r"ILQX ({}={:g})".format(P.PARAM_SYMBOL["lam"], p["lam"])
    qtex = {"q2": "$q=2$", "q4": "$q=4$", "qlog": r"$q=1+\log\kappa_\theta$"}[s]
    return r"AMLE-PGD ({}, {}={:g}, {}={:g})".format(qtex, P.PARAM_SYMBOL["eta"], p["eta"], P.PARAM_SYMBOL["eps"],
                                                     p["eps"])


def panel2(ax, best: dict, title: str, ymax: float) -> None:
    handles = []
    present = [s for s in ORDER if s in best]
    for i, s in enumerate(present):
        row = best[s]
        T = row["curve_mean"].size
        idx = np.unique(np.linspace(0, T - 1, 400).astype(int))
        t, mu, se = idx + 1, row["curve_mean"][idx], row["curve_se"][idx]
        ls = "--" if s == "lasso" else "-"
        mk = "s" if s == "lasso" else "o"
        ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=COLOR[s], alpha=0.08, lw=0)
        ax.plot(t, mu, color=COLOR[s], ls=ls, lw=1.6)
        mi = mark_idx(t.size, i, len(present))
        ax.plot(t[mi], mu[mi], marker=mk, ms=4.5, color=COLOR[s], mec="white", mew=0.8, ls="none")
        handles.append(plt.Line2D([], [], color=COLOR[s], ls=ls, lw=1.6, marker=mk, ms=5, mec="white", mew=0.8,
                                  label=label(s, row)))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Regret")
    ax.set_xlim(0, None)
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_formatter(FuncFormatter(P.compact))
    ax.yaxis.set_major_formatter(FuncFormatter(P.compact))
    P._style_axes(ax)
    ax.legend(handles=handles, loc="upper left", fontsize=5.6, ncol=1, handlelength=2.4, labelspacing=0.25)


def main() -> None:
    outdir = os.path.join(HERE, "figures", "qsens_sparse")
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.0))
    _, title1, tags1 = Q.CASES[0]
    panel1_relative(axes[0], Q.select(Q.collect(tags1)), tags1, title1)
    b27, b3 = load_case2(27), load_case2(3)
    ymax = 1.08 * max(max(r["curve_mean"][-1] + 1.96 * r["curve_se"][-1] for r in b.values()) for b in (b27, b3))
    panel2(axes[1], b27, r"(b) Case II: sparse parameters, $m_k=\lfloor\sqrt{\kappa_\theta}\rfloor=27$", ymax)
    panel2(axes[2], b3, r"(c) Case II: sparse parameters, $m_k=\lceil\sqrt{s}\rceil=3$", ymax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_q_sensitivity_sparse.pdf"))
    fig.savefig(os.path.join(outdir, "fig_q_sensitivity_sparse.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(outdir, "summary_sparse_case2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "series", "param", "K", "final_mean", "final_se", "time_mean_s"])
        for m, b in ((27, b27), (3, b3)):
            for s in ORDER:
                if s in b:
                    r = b[s]
                    w.writerow([m, s, r["param"], r["K"], r["final"].mean(), r["final"].std(ddof=1) / math.sqrt(r["K"]),
                                r["time_mean"]])
                    print("m={:<3d} {:6s} {:40s} K={} regret {:.3f}M  time {:.1f}s".format(
                        m, s, str(r["param"]), r["K"], r["final"].mean() / 1e6, r["time_mean"]))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
