"""Figure EC.1 of the revised paper after the sparse Case II moved out of the e-companion (2026-11-19).

    python3 plot_qsens_caseI_r2.py      # needs results/dense_alg1, qsens_alg1 and qsens_tol_alg1

Case I of plot_qsens_sparse_r2's figure (dense design of Section 5.1): paired relative regret
R_k(t) / R_k^ref(t) - 1 (mean over trials k and 95% band), with AMLE-MOSEK (q = 2) as reference.
Drawn for a narrow e-companion panel: colour by q, line style by method, a distinct marker per curve repeated
along the curve, and a legend below the axes whose entries give the name on one line and the parameters on the next.
Output: figures/qsens_caseI/fig_q_sensitivity_caseI.pdf (and .png).
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import plot_r2 as P
import plot_qsens_r2 as Q
import plot_qsens_sparse_r2 as S

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = {("MOSEK", 2.0): None, ("MOSEK", 4.0): "s", ("MOSEK", Q.Q_LOG): "D",
          ("PGD", 2.0): "o", ("PGD", 4.0): "^", ("PGD", Q.Q_LOG): "v"}


def two_lines(method: str, r: dict, ref: bool = False) -> str:
    name = "AMLE-MOSEK" if method == "MOSEK" else "AMLE-PGD"
    hp = "{}={:g}".format(P.PARAM_SYMBOL["eta"], r["param"]["eta"])
    if method == "PGD":
        hp += ", {}={:g}".format(P.PARAM_SYMBOL["eps"], r["param"]["eps"])
    return "{}, {}{}\n{}".format(name, Q.Q_TEX[r["q"]], " (reference)" if ref else "", hp)


def main() -> None:
    outdir = os.path.join(HERE, "figures", "qsens_caseI")
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 7.0})
    _, _, tags = Q.CASES[0]
    sel = Q.select(Q.collect(tags))
    runs = {}
    for tag in tags:
        root = os.path.join(HERE, "results", tag)
        if os.path.isdir(root):
            runs.update(P.load(root))

    def curves(r):
        lst = runs[(r["snr"], r["method"], json.dumps(r["param"], sort_keys=True))]
        return {d["k"]: d["regret"] for d in lst}

    fig, ax = plt.subplots(figsize=(2.95, 2.1))
    ref = curves(sel[("MOSEK", 2.0)])
    ax.axhline(0.0, color=Q.Q_COLOR[2.0], ls="--", lw=1.4)
    handles = {("MOSEK", 2.0): plt.Line2D([], [], color=Q.Q_COLOR[2.0], ls="--", lw=1.4,
                                          label=two_lines("MOSEK", sel[("MOSEK", 2.0)], ref=True))}
    keys = [(m, q) for m in ("MOSEK", "PGD") for q in Q.Q_ORDER if (m, q) != ("MOSEK", 2.0) and (m, q) in sel]
    for method, q in keys:
        r = sel[(method, q)]
        cur = curves(r)
        ks = sorted(set(cur) & set(ref))
        R = np.vstack([cur[k] / ref[k] - 1.0 for k in ks]) * 100.0
        idx = np.unique(np.linspace(S.REL_T0 - 1, R.shape[1] - 1, 300).astype(int))
        t, mu = idx + 1, R[:, idx].mean(0)
        se = R[:, idx].std(0, ddof=1) / math.sqrt(len(ks))
        ls = Q.M_STYLE[method]["ls"]
        mfc = "white" if method == "MOSEK" else Q.Q_COLOR[q]
        ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=Q.Q_COLOR[q], alpha=0.06, lw=0)
        off = {2.0: 0, 4.0: 20, Q.Q_LOG: 40}[q] + (10 if method == "PGD" else 0)
        ax.plot(t, mu, color=Q.Q_COLOR[q], ls=ls, lw=1.3, marker=MARKER[(method, q)], ms=4.2,
                markevery=(off, 60), mfc=mfc, mec=Q.Q_COLOR[q], mew=1.0)
        handles[(method, q)] = plt.Line2D([], [], color=Q.Q_COLOR[q], ls=ls, lw=1.3, marker=MARKER[(method, q)],
                                          ms=4.2, mfc=mfc, mec=Q.Q_COLOR[q], mew=1.0, label=two_lines(method, r))
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Relative regret (%)")
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_xlim(0, None)
    ax.xaxis.set_major_formatter(FuncFormatter(P.compact))
    P._style_axes(ax)
    order = [("MOSEK", q) for q in Q.Q_ORDER] + [("PGD", q) for q in Q.Q_ORDER]   # column-major, two columns
    hs = [handles[k] for k in order if k in handles]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=5.7, handlelength=2.0, labelspacing=0.35,
               columnspacing=0.8, frameon=False)
    fig.tight_layout(rect=(0, 0.34, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "fig_q_sensitivity_caseI." + ext), dpi=150)
    plt.close(fig)
    print("wrote", outdir)


if __name__ == "__main__":
    sys.exit(main())
