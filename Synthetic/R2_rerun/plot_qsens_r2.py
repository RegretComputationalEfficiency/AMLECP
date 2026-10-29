"""Figure EC.1 (sensitivity to q) for the R2 rerun.

Both cases: mid SNR (SNR = 1), 10 trials, Algorithm 2 for AMLE-PGD. q in {2, 4, 1 + log kappa_theta} sets the
regularizer exponent p = q/(q-1) and the dimension factor of eta_k; the exploration schedule (m_k = 27) is the
same for every q, so that only the regularizer changes. All q use the same tolerance as Section 5.1,
eps_k = eps0 kappa^(1/4) sqrt(|I_k|) (parameter eps_q = 4), so that they are compared at equal accuracy.
  Case I : the dense design of Section 5.1 (T = 10,000). q = 4 is taken from results/dense_alg1; AMLE-MOSEK for
           q = 2 and q = 1 + log kappa_theta from results/qsens_alg1 (exact solutions, independent of eps); AMLE-PGD
           for these q from results/qsens_tol_alg1 (grid_qsens_tol.json). All runs use the Algorithm 1 schedule of
           env_r2.schedule (queue_qsens_alg1.sh).
  Case II: heavy-tailed features and a sparse parameter vector with prices in units of 100 in the estimation
           problem (R2_DESIGN=heavy100, T = 5,000), all q from results/qsens_heavy100_alg1 (grid_qsens_heavy100.json).

For each case and q: AMLE-MOSEK at its best eta0 (minimum mean final regret) and AMLE-PGD at its representative
configuration (at its best eta0, the largest eps0 whose mean final regret is within 1% of its minimum), as in
Table 2.

Outputs (figures/qsens_alg1/): fig_q_sensitivity.pdf, table_q_sensitivity.tex, summary_q_sensitivity.csv,
summary_q_by_eps.csv (for each tolerance eps0: the best AMLE-PGD regret over eta0 for every q)
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

import plot_r2 as P   # style, loading and aggregation
import matplotlib.pyplot as plt   # noqa: E402  (backend set by plot_r2)
from matplotlib.ticker import FuncFormatter   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SNR = "mid"
Q_LOG = 1.0 + math.log(730.0)
Q_ORDER = [2.0, 4.0, Q_LOG]
Q_TEX = {2.0: "$q=2$", 4.0: "$q=4$", Q_LOG: r"$q=1+\log\kappa_\theta$"}
Q_COLOR = {4.0: "#2a78d6", 2.0: "#eb6834", Q_LOG: "#1baf7a"}        # validated categorical palette, fixed order
M_STYLE = {"MOSEK": dict(ls="--", marker="s"), "PGD": dict(ls="-", marker="o")}
CASES = [
    ("I", "(a) Case I: Section 5.1 design (dense parameters)", ["dense_alg1", "qsens_alg1", "qsens_tol_alg1"]),
    ("II", "(b) Case II: heavy-tailed features and sparse parameters", ["qsens_heavy100_alg1"]),
]
COMMON_EPS_Q = 4.0   # dimension factor of eps_k shared by all q (Section 5.1)
EPS_GRID = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def _q_of(r) -> float:
    q = float(r["param"].get("q", 4.0))
    return min(Q_ORDER, key=lambda z: abs(z - q))


def _eps_q_of(r) -> float:
    return float(r["param"].get("eps_q", r["param"].get("q", 4.0)))


def collect(tags: list[str]) -> list[dict]:
    """AMLE-MOSEK and AMLE-PGD rows at mid SNR; AMLE-PGD only at the common tolerance (eps_q = 4)."""
    rows = []
    for tag in tags:
        root = os.path.join(HERE, "results", tag)
        if not os.path.isdir(root):
            continue
        rows += [r for r in P.aggregate(P.load(root)) if r["snr"] == SNR and r["method"] in ("MOSEK", "PGD")]
    rows = [r for r in rows if r["method"] == "MOSEK" or abs(_eps_q_of(r) - COMMON_EPS_Q) < 1e-9]
    for r in rows:
        r["q"] = _q_of(r)
    return rows


def by_eps(rows: list[dict]) -> list[tuple]:
    """(q, eps0, best AMLE-PGD row over eta0) for every tolerance."""
    out = []
    for q in Q_ORDER:
        for eps in EPS_GRID:
            pg = [r for r in rows if r["method"] == "PGD" and r["q"] == q and r["param"]["eps"] == eps]
            if pg:
                out.append((q, eps, min(pg, key=lambda r: r["final_mean"])))
    return out


def select(rows: list[dict]) -> dict:
    """(method, q) -> row: best AMLE-MOSEK and representative AMLE-PGD (plus the minimum-regret AMLE-PGD)."""
    out = {}
    for q in Q_ORDER:
        ms = [r for r in rows if r["method"] == "MOSEK" and r["q"] == q]
        pg = [r for r in rows if r["method"] == "PGD" and r["q"] == q]
        if ms:
            out[("MOSEK", q)] = min(ms, key=lambda r: r["final_mean"])
        if pg:
            r_min = min(pg, key=lambda r: r["final_mean"])
            eta = r_min["param"]["eta"]
            cand = [r for r in pg if r["param"]["eta"] == eta and r["final_mean"] <= (1 + P.PGD_TOL) * r_min["final_mean"]]
            out[("PGD", q)] = max(cand, key=lambda r: r["param"]["eps"])
            out[("PGD_min", q)] = r_min
    return out


def label(method: str, r: dict) -> str:
    name = "AMLE-MOSEK" if method == "MOSEK" else "AMLE-PGD"
    hp = "{}={:g}".format(P.PARAM_SYMBOL["eta"], r["param"]["eta"])
    if method == "PGD":
        hp += ", {}={:g}".format(P.PARAM_SYMBOL["eps"], r["param"]["eps"])
    return "{} ({}, {})".format(name, Q_TEX[r["q"]], hp)


def panel(ax, sel: dict, title: str) -> None:
    handles = []
    for method in ("MOSEK", "PGD"):
        for q in Q_ORDER:
            r = sel.get((method, q))
            if r is None:
                continue
            T = r["curve_mean"].size
            idx = np.unique(np.linspace(0, T - 1, 400).astype(int))
            t, mu, se = idx + 1, r["curve_mean"][idx], r["curve_se"][idx]
            ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=Q_COLOR[q], alpha=0.08, lw=0)
            ax.plot(t, mu, color=Q_COLOR[q], ls=M_STYLE[method]["ls"], lw=1.6)
            ax.plot(t[-1], mu[-1], marker=M_STYLE[method]["marker"], ms=6, color=Q_COLOR[q], mec="white", mew=1.0,
                    ls="none")
            handles.append(plt.Line2D([], [], color=Q_COLOR[q], ls=M_STYLE[method]["ls"], lw=1.6,
                                      marker=M_STYLE[method]["marker"], ms=5, mec="white", mew=0.8,
                                      label=label(method, r)))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Regret")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(FuncFormatter(P.compact))
    ax.yaxis.set_major_formatter(FuncFormatter(P.compact))
    P._style_axes(ax)
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=6.4, ncol=2, handlelength=2.4, labelspacing=0.3,
                  columnspacing=1.0)


def tables(selections: dict, outdir: str, eps_rows: dict | None = None) -> None:
    if eps_rows:
        with open(os.path.join(outdir, "summary_q_by_eps.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "q", "eps0", "eta0", "final_mean", "final_se", "exploit_mean", "time_adj_mean",
                        "iters_mean", "cert_rate"])
            for case, lst in eps_rows.items():
                for q, eps, r in lst:
                    w.writerow([case, q, eps, r["param"]["eta"], r["final_mean"], r["final_se"], r["exploit_mean"],
                                r["time_adj_mean"], r.get("iters_mean", ""), r.get("cert_rate", "")])
    with open(os.path.join(outdir, "summary_q_sensitivity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method", "q", "param", "final_mean", "final_se", "exploit_mean", "exploit_se",
                    "time_adj_mean", "iters_mean", "cert_rate", "K"])
        for case, sel in selections.items():
            for (m, q), r in sorted(sel.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                w.writerow([case, m, q, r["param"], r["final_mean"], r["final_se"], r["exploit_mean"], r["exploit_se"],
                            r["time_adj_mean"], r.get("iters_mean", ""), r.get("cert_rate", ""), r["K"]])
    names = {"MOSEK": "AMLE-MOSEK", "PGD": "AMLE-PGD (representative)", "PGD_min": "AMLE-PGD (minimum regret)"}
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Case & Method & Hyperparameters & Regret ($\times10^3$) & Exploitation regret ($\times10^3$)"
             r" & Total time (s) \\", r"\midrule"]
    for case, sel in selections.items():
        for m in ("MOSEK", "PGD", "PGD_min"):
            for q in Q_ORDER:
                r = sel.get((m, q))
                if r is None:
                    continue
                hp = r"{}, $\mathring\eta={:g}$".format(Q_TEX[q], r["param"]["eta"])
                if m != "MOSEK":
                    hp += r", $\mathring\varepsilon={:g}$".format(r["param"]["eps"])
                lines.append(r"{} & {} & {} & {:.1f} ({:.1f}) & {:.1f} ({:.1f}) & {:.2f} \\".format(
                    case, names[m], hp, r["final_mean"] / 1e3, r["final_se"] / 1e3, r["exploit_mean"] / 1e3,
                    r["exploit_se"] / 1e3, r["time_adj_mean"]))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(os.path.join(outdir, "table_q_sensitivity.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    outdir = os.path.join(HERE, "figures", "qsens_alg1")
    os.makedirs(outdir, exist_ok=True)
    selections, eps_rows = {}, {}
    fig, axes = plt.subplots(1, len(CASES), figsize=(5.6 * len(CASES), 2.8), squeeze=False)
    for j, (case, title, tags) in enumerate(CASES):
        rows = collect(tags)
        sel = select(rows)
        selections[case] = sel
        eps_rows[case] = by_eps(rows)
        panel(axes[0, j], sel, title)
        for (m, q), r in sorted(sel.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            print("Case {:3s} {:8s} q={:<6.3f} K={:<3d} {}  regret {:.1f}K  exploit {:.1f}K  time {:.1f}s".format(
                case, m, q, r["K"], r["param"], r["final_mean"] / 1e3, r["exploit_mean"] / 1e3, r["time_adj_mean"]))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_q_sensitivity.pdf"))
    plt.close(fig)
    tables(selections, outdir, eps_rows)
    for case, lst in eps_rows.items():
        print("Case {} -- best AMLE-PGD regret over eta0 at each eps0 (common tolerance):".format(case))
        for q, eps, r in lst:
            print("  q={:<6.3f} eps0={:<6g} eta0={:<6g} regret {:.1f}K  cert {}".format(
                q, eps, r["param"]["eta"], r["final_mean"] / 1e3, r.get("cert_rate", "")))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
