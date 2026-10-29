"""Figures and tables for the R2 rerun of Section 5.1.

Reads ``results/<tag>/<snr>/<method>/<param>/k*.npz`` written by ``run_r2.py`` and produces

  main text
    fig_best_regret.pdf     regret of every method at its best hyper-parameter (one panel per SNR, broken y-axis)
    fig_tradeoff_eps.pdf    effect of the tolerance eps0 on regret, computation time and PGD iterations
    fig_regret_time.pdf     regret against total computation time: AMLE-PGD path over eps0, one point per method
  appendix
    fig_all_params_final.pdf   final regret of every (method, hyper-parameter), one panel per SNR
    fig_all_params_curves_pgd.pdf    AMLE-PGD regret curves, one panel per (SNR, eta0), one curve per eps0
    fig_all_params_curves_bench.pdf  benchmark regret curves, one panel per (SNR, method), one curve per value
    fig_pgd_iterations.pdf     PGD iterations per update against |I_k| (empirical check of Proposition 2)
  tables
    table_regret_time.tex   main text: best hyper-parameter, regret and time per method (layout of R1 Table 2)
    summary_all.csv, summary_best.csv, table_best.tex (detailed, incl. exploitation regret and ratios)

Regret is the pseudo-regret sum_t [R*(X_t) - R(p_t, X_t)], with R(p, X) the expected revenue at the
posted price (the paper's definition); the best hyper-parameter minimizes the mean final regret.
The exploitation regret sums only over exploitation periods: the exploration periods (prices drawn by
Algorithm 1, identical for every method) contribute the same amount to every method's regret.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import env_r2 as env                                  # noqa: E402  (exploration schedule only)

# ---------------------------------------------------------------------------
# Style (validated categorical palette, recessive chrome, 2px lines, ringed markers)
# ---------------------------------------------------------------------------
SNRS = ["low", "mid", "high"]
SNR_TITLE = {"low": "Low SNR (0.1)", "mid": "Mid SNR (1.0)", "high": "High SNR (10.0)"}
METHODS = ["PGD", "MOSEK", "ILQX", "SPO+", "AdaGrad"]
LABEL = {"PGD": "AMLE-PGD", "MOSEK": "AMLE-MOSEK", "ILQX": "ILQX", "SPO+": "SPO+", "AdaGrad": "AdaGrad"}
COLOR = {"PGD": "#2a78d6", "MOSEK": "#eb6834", "ILQX": "#1baf7a", "SPO+": "#eda100", "AdaGrad": "#e87ba4"}
MARKER = {"PGD": "o", "MOSEK": "s", "ILQX": "D", "SPO+": "^", "AdaGrad": "v"}
LSTYLE = {"PGD": "-", "MOSEK": "--", "ILQX": "-.", "SPO+": (0, (5, 1.5, 1, 1.5)), "AdaGrad": ":"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # ordinal blue ramp, steps 250-700
def ramp_levels(n: int) -> list[str]:
    """n ordered colors from the blue ramp, spread over steps 350-700 (light = small value)."""
    if n <= 1:
        return [COLOR_PGD_DEFAULT]
    return [RAMP[int(round(x))] for x in np.linspace(1, len(RAMP) - 1, n)]
COLOR_PGD_DEFAULT = "#2a78d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PARAM_SYMBOL = {"eta": r"$\mathring{\eta}$", "eps": r"$\mathring{\varepsilon}$", "lam": r"$\mathring{\lambda}$",
                "lr": "lr", "q": "q", "eps_q": r"$q_\varepsilon$"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def compact(x: float, _pos=None) -> str:
    ax = abs(x)
    if ax >= 1e6:
        return "{:g}M".format(round(x / 1e6, 2))
    if ax >= 1e3:
        return "{:g}K".format(round(x / 1e3, 1))
    return "{:.3g}".format(x)


def param_label(method: str, param: dict) -> str:
    return ", ".join("{}={:g}".format(PARAM_SYMBOL.get(k, k), param[k]) for k in sorted(param, key=_ORDER.index))


_ORDER = ["q", "eta", "eps", "eps_q", "lam", "lr"]


def _pkey(param: dict) -> tuple:
    return tuple(param.get(k, 0.0) for k in _ORDER)


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------
def load(root: str) -> dict:
    """runs[(snr, method, param_json)] -> list of per-k dicts."""
    runs: dict = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "k*.npz"))):
        if path.endswith(".tmp.npz"):
            continue
        r = np.load(path, allow_pickle=False)
        meta = json.loads(str(r["meta"]))
        inc = r["oracle"] - r["rev_exp"]
        is_exp = env.schedule(int(meta["T"]))[0]
        runs[(meta["snr"], meta["method"], json.dumps(meta["param"], sort_keys=True))].append({
            "k": meta["k"], "regret": np.cumsum(inc), "exploit": float(inc[~is_exp].sum()),
            "time": float(r["total_time"]),
            "fit_N": r["fit_N"], "fit_iters": r["fit_iters"], "fit_stop": r["fit_stop"],
            "fit_time": r["fit_time"],
        })
    return runs


def aggregate(runs: dict) -> list[dict]:
    rows = []
    for (snr, method, pj), lst in runs.items():
        lst = sorted(lst, key=lambda d: d["k"])
        R = np.vstack([d["regret"] for d in lst])
        fin = R[:, -1]
        times = np.array([d["time"] for d in lst])
        exploits = np.array([d["exploit"] for d in lst])
        K = R.shape[0]
        se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")  # noqa: E731
        row = {"snr": snr, "method": method, "param": json.loads(pj), "K": K,
               "ks": np.array([d["k"] for d in lst]), "finals": fin, "times": times, "exploits": exploits,
               "final_mean": float(fin.mean()), "final_se": se(fin),
               "exploit_mean": float(exploits.mean()), "exploit_se": se(exploits),
               "time_mean": float(times.mean()), "time_se": se(times),
               "curve_mean": R.mean(axis=0), "curve_se": R.std(axis=0, ddof=1) / np.sqrt(K) if K > 1 else 0 * R[0],
               "n_updates": float(np.mean([d["fit_N"].size for d in lst]))}
        if method == "PGD":
            its = np.concatenate([d["fit_iters"] for d in lst])
            stops = np.concatenate([d["fit_stop"] for d in lst])
            row.update(iters_mean=float(its.mean()), iters_max=int(its.max()),
                       cert_rate=float(np.mean(stops == "cert")),
                       iters_by_fit=np.mean(np.vstack([d["fit_iters"] for d in lst]), axis=0),
                       N_by_fit=lst[0]["fit_N"])
        row["times_adj"], row["time_adj_mean"] = times, row["time_mean"]
        if method in ("MOSEK", "ILQX"):
            stops = np.concatenate([d["fit_stop"] for d in lst])
            row.update(fallback_rate=float(np.mean(stops != "mosek")))
            if len({d["fit_time"].size for d in lst}) == 1:
                # Conservative solver time: an update that needed a fallback solve (MOSEK stalled) is charged
                # the median first-attempt MOSEK time of the same update index over the other trials.
                Tm = np.vstack([d["fit_time"] for d in lst]).astype(float)
                ok = np.vstack([d["fit_stop"] for d in lst]).astype(str) == "mosek"
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(np.where(ok, Tm, np.nan), axis=0)
                good = ~np.isnan(med)
                if good.any():
                    idx = np.arange(med.size)
                    med[~good] = np.interp(idx[~good], idx[good], med[good])
                    adj = np.where(ok, Tm, med[None, :]).sum(axis=1)
                    row["times_adj"], row["time_adj_mean"] = adj, float(adj.mean())
        rows.append(row)
    return rows


PGD_TOL = 0.01   # representative AMLE-PGD: largest eps0 whose regret is within 1% of the best PGD regret
RATIO_CAP = 1.5  # y-limit of the regret-ratio panel; larger ratios (estimation breakdown) are marked off scale


def best_rows(rows: list[dict]) -> dict:
    """Best hyper-parameter = minimum mean final regret.  For AMLE-PGD, best[(snr, 'PGD')] is the
    representative configuration: at the best eta0, the largest tolerance eps0 whose mean final regret is
    within PGD_TOL of the minimum (the regret-minimizing configuration is kept as best[(snr, 'PGD_min')])."""
    best = {}
    for r in rows:
        key = (r["snr"], r["method"])
        if key not in best or r["final_mean"] < best[key]["final_mean"]:
            best[key] = r
    for snr in {r["snr"] for r in rows}:
        if (snr, "PGD") not in best:
            continue
        r_min = best[(snr, "PGD")]
        best[(snr, "PGD_min")] = r_min
        eta = r_min["param"]["eta"]
        cand = [r for r in rows if r["snr"] == snr and r["method"] == "PGD" and r["param"]["eta"] == eta
                and r["final_mean"] <= (1.0 + PGD_TOL) * r_min["final_mean"]]
        best[(snr, "PGD")] = max(cand, key=lambda r: r["param"]["eps"])
    return best


def paired_ratio(num: dict, den: dict, field: str, den_field: str | None = None) -> tuple[float, float]:
    """Mean and standard error over common trials k of num[field]_k / den[den_field or field]_k."""
    common = np.intersect1d(num["ks"], den["ks"])
    if common.size == 0:
        return float("nan"), float("nan")
    a = num[field][np.searchsorted(num["ks"], common)]
    b = den[den_field or field][np.searchsorted(den["ks"], common)]
    q = a / b
    se = float(np.std(q, ddof=1) / np.sqrt(q.size)) if q.size > 1 else float("nan")
    return float(q.mean()), se


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style_axes(ax):
    ax.tick_params(length=2.5, width=0.6)
    ax.set_axisbelow(True)


def _nice_ceil(x: float) -> float:
    """Round up to a multiple of half a power of ten (e.g. 929K -> 950K, 1.91M -> 2M)."""
    step = 10 ** np.floor(np.log10(x)) / 2
    return float(np.ceil(x / step) * step)


def fig_best_regret(rows, best, snrs, out):
    """Regret of every method at its best hyper-parameter (AMLE-PGD: representative configuration), one column
    per SNR.  The y-axis is broken: the lower segment (3/4 of the height) covers every method except AdaGrad,
    whose much larger regret is shown in the upper segment; the two segments are contiguous."""
    fig = plt.figure(figsize=(2.55 * len(snrs) + 0.6, 3.35))
    gs = fig.add_gridspec(2, len(snrs), height_ratios=[1, 3], hspace=0.06, wspace=0.34)
    for j, snr in enumerate(snrs):
        ax_top = fig.add_subplot(gs[0, j])
        ax_bot = fig.add_subplot(gs[1, j], sharex=ax_top)
        lo_max, hi_max = 0.0, 0.0
        for m in METHODS:
            r = best.get((snr, m))
            if r is None:
                continue
            T = r["curve_mean"].size
            idx = np.unique(np.linspace(0, T - 1, 400).astype(int))
            t = idx + 1
            mu, se = r["curve_mean"][idx], r["curve_se"][idx]
            end = float(mu[-1] + 1.96 * se[-1])
            if m == "AdaGrad":
                hi_max = max(hi_max, end)
            else:
                lo_max = max(lo_max, end)
            for ax in (ax_top, ax_bot):
                ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=COLOR[m], alpha=0.10, lw=0)
                ax.plot(t, mu, color=COLOR[m], ls=LSTYLE[m], label=LABEL[m])
                ax.plot(t[-1], mu[-1], marker=MARKER[m], ms=5.5, color=COLOR[m], mec="white", mew=1.0, ls="none")
        y_b = _nice_ceil(1.06 * lo_max)
        y_t = _nice_ceil(1.04 * hi_max) if hi_max > y_b else 1.25 * y_b
        ax_bot.set_ylim(0, y_b)
        ax_top.set_ylim(y_b, y_t)
        ax_top.set_xlim(0, None)
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10]))
        ax_top.spines["bottom"].set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.set_title(SNR_TITLE[snr])
        ax_bot.set_xlabel("Period $t$")
        for ax in (ax_top, ax_bot):
            ax.xaxis.set_major_formatter(FuncFormatter(compact))
            ax.yaxis.set_major_formatter(FuncFormatter(compact))
            _style_axes(ax)
        d, kw = 0.03, dict(color=INK2, lw=0.9, clip_on=False)       # break marks on the y-axis
        ax_top.plot([-d, d], [-3 * d, 3 * d], transform=ax_top.transAxes, **kw)
        ax_bot.plot([-d, d], [1 - d, 1 + d], transform=ax_bot.transAxes, **kw)
        if j == 0:
            ax_bot.set_ylabel("Regret")
            ax_bot.yaxis.set_label_coords(-0.27, 0.69)
    handles = [plt.Line2D([], [], color=COLOR[m], ls=LSTYLE[m], marker=MARKER[m], ms=5, mec="white", mew=0.8,
                          label=LABEL[m]) for m in METHODS if any((s, m) in best for s in snrs)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.84)
    fig.savefig(out)
    plt.close(fig)


def fig_tradeoff(rows, best, snrs, out):
    """For each SNR, AMLE-PGD at its best regularization level eta0* (minimum regret) and at eta0*/sqrt(10) and
    sqrt(10) eta0*, each compared with AMLE-MOSEK at the same eta0, i.e. with the exact solution of the same
    regularized problem; ratios are paired over common trials (common random numbers).  Open markers:
    configurations in which some updates stopped at the iteration cap, i.e. Condition 1 was not certified."""
    pgd = [r for r in rows if r["method"] == "PGD"]
    if not pgd:
        return
    levels = [(10 ** -0.5, RAMP[1], "o", r"AMLE-PGD, next smaller $\mathring{\eta}$ on the grid"),
              (1.0, RAMP[3], "s", r"AMLE-PGD, best $\mathring{\eta}$ (bold in panel title)"),
              (10 ** 0.5, RAMP[5], "D", r"AMLE-PGD, next larger $\mathring{\eta}$ on the grid")]
    fig, axes = plt.subplots(3, len(snrs), figsize=(2.75 * len(snrs) + 0.6, 7.2), sharex=True, squeeze=False)
    lowest, any_unc = 1.0, False
    for j, snr in enumerate(snrs):
        rs = [r for r in pgd if r["snr"] == snr]
        if not rs:
            continue
        eta_star = best[(snr, "PGD_min")]["param"]["eta"]
        etas = sorted({r["param"]["eta"] for r in rs})
        ms_rows = {r["param"]["eta"]: r for r in rows if r["snr"] == snr and r["method"] == "MOSEK"}
        off_scale: dict = defaultdict(list)
        shown = []
        for fac, col, mk, lab in levels:
            match = [e for e in etas if abs(e / (eta_star * fac) - 1.0) < 0.02]
            if not match:
                continue
            eta = match[0]
            shown.append(r"$\mathbf{%g}$" % eta if fac == 1.0 else "%g" % eta)
            sub = sorted([r for r in rs if r["param"]["eta"] == eta], key=lambda r: r["param"]["eps"])
            eps = np.array([r["param"]["eps"] for r in sub])
            kw = dict(color=col, marker=mk, ms=5, mec="white", mew=0.9, label=lab, zorder=3)
            unc = np.array([r["cert_rate"] < 1.0 for r in sub])
            any_unc = any_unc or bool(unc.any())
            open_kw = dict(ls="none", marker=mk, ms=5, mfc="white", mec=col, mew=1.2, zorder=5)
            ref = next((m for e, m in ms_rows.items() if abs(e / eta - 1.0) < 0.02), None)
            if ref is not None:
                q = np.array([paired_ratio(r, ref, "finals") for r in sub])
                ok = q[:, 0] <= RATIO_CAP
                lowest = min(lowest, float(np.min(q[ok, 0] - 1.96 * q[ok, 1]))) if ok.any() else lowest
                axes[0, j].errorbar(eps[ok], q[ok, 0], yerr=1.96 * q[ok, 1], capsize=0, elinewidth=1.0, **kw)
                if (unc & ok).any():
                    axes[0, j].plot(eps[unc & ok], q[unc & ok, 0], **open_kw)
                if (~ok).any():   # estimation breaks down: mark off-scale points at the top edge
                    axes[0, j].plot(eps[~ok], np.full((~ok).sum(), RATIO_CAP), ls="none", marker="^", ms=6,
                                    color=col, mec="white", mew=0.9, clip_on=False, zorder=4)
                    for e_, v_ in zip(eps[~ok], q[~ok, 0]):
                        off_scale[float(e_)].append(float(v_))
                qt = np.array([paired_ratio(r, ref, "times", "times_adj") for r in sub])
                axes[1, j].errorbar(eps, qt[:, 0], yerr=1.96 * qt[:, 1], capsize=0, elinewidth=1.0, **kw)
                if unc.any():
                    axes[1, j].plot(eps[unc], qt[unc, 0], **open_kw)
            iters = np.array([max(r["iters_mean"], 1.0) for r in sub])
            axes[2, j].plot(eps, iters, **kw)
            if unc.any():
                axes[2, j].plot(eps[unc], iters[unc], **open_kw)
        for n_off, (e_, vals) in enumerate(sorted(off_scale.items())):
            txt = ("×{:.1f}".format(vals[0]) if len(vals) == 1
                   else "×{:.1f}–{:.1f}".format(min(vals), max(vals)))
            axes[0, j].annotate(txt, (e_, RATIO_CAP), xytext=(0, -12 - 10 * (n_off % 2)), textcoords="offset points",
                                ha="center", fontsize=6.3, color=INK2, zorder=6,
                                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        for i in (0, 1):
            axes[i, j].axhline(1.0, color=COLOR["MOSEK"], ls="--", lw=1.4, zorder=2,
                               label="AMLE-MOSEK (exact solution, same {})".format(PARAM_SYMBOL["eta"]))
        axes[0, j].set_title("{}\n{} = {}".format(SNR_TITLE[snr], PARAM_SYMBOL["eta"], ", ".join(shown)))
        axes[2, j].set_xlabel(r"Tolerance $\mathring{\varepsilon}$")
        for i in range(3):
            ax = axes[i, j]
            ax.set_xscale("log")
            if i > 0:
                ax.set_yscale("log")
            _style_axes(ax)
    lo = min(0.98, np.floor((lowest - 0.01) * 20) / 20)
    all_eps = [r["param"]["eps"] for r in pgd]
    for j in range(len(snrs)):
        axes[0, j].set_ylim(lo, RATIO_CAP)
        axes[0, j].set_xlim(min(all_eps) / 1.8, max(all_eps) * 1.8)
    axes[0, 0].set_ylabel("Final regret / AMLE-MOSEK")
    axes[1, 0].set_ylabel("Optimization time / AMLE-MOSEK\n(MOSEK without fallback solves)")
    axes[2, 0].set_ylabel("PGD iterations per update")
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    order = sorted(range(len(l)), key=lambda i: (l[i].startswith("AMLE-MOSEK"),
                                                 [lv[3] for lv in levels].index(l[i]) if l[i] in [lv[3] for lv in levels] else 0))
    h, l = [h[i] for i in order], [l[i] for i in order]
    h.append(plt.Line2D([], [], color=MUTED, marker="^", ls="none", ms=6, mec="white"))
    l.append("regret ratio > {:g} (actual value printed)".format(RATIO_CAP))
    if any_unc:
        h.append(plt.Line2D([], [], color=MUTED, marker="o", ls="none", ms=5, mfc="white", mec=MUTED, mew=1.2))
        l.append("PGD hit the 20,000-iteration cap in some updates")
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(out)
    plt.close(fig)


def fig_regret_time(rows, best, snrs, out):
    """Regret against total computation time over the horizon (both axes log scale), one panel per SNR.
    AMLE-PGD: its path over the tolerance eps0 at the best regularization level (large dot = representative
    configuration, named in the panel title); every other method: one point at its best hyper-parameter.  The y-axis stops just above the
    largest regret among the other methods; path points beyond it are drawn as open circles at the top edge with their value.
    Times as in the tables (AMLE-MOSEK and ILQX without the extra cost of fallback solves)."""
    others = ("MOSEK", "ILQX", "SPO+", "AdaGrad")
    fig, axes = plt.subplots(1, len(snrs), figsize=(2.85 * len(snrs) + 0.6, 3.4), squeeze=False)
    lab_kw = dict(textcoords="offset points", fontsize=7, color=INK2, zorder=7)
    for j, snr in enumerate(snrs):
        ax = axes[0, j]
        rep = best.get((snr, "PGD"))
        if rep is None:
            continue
        eta = rep["param"]["eta"]
        path = sorted([r for r in rows if r["snr"] == snr and r["method"] == "PGD" and r["param"]["eta"] == eta],
                      key=lambda r: r["param"]["eps"])
        px = np.array([r["time_adj_mean"] for r in path])
        py = np.array([r["final_mean"] for r in path])
        pts = {m: best[(snr, m)] for m in others if (snr, m) in best}
        lo = 0.9 * min(py.min(), min(r["final_mean"] for r in pts.values()))
        hi = 1.15 * max(r["final_mean"] for r in pts.values())
        inside = py <= hi
        # path, clipped at the top; points above the cap are drawn on the top edge and labelled with their value
        ax.plot(px, np.minimum(py, hi), color=COLOR["PGD"], lw=1.6, zorder=3, clip_on=False)
        ax.plot(px[inside], py[inside], ls="none", marker="o", ms=5, color=COLOR["PGD"], mec="white", mew=1.2, zorder=4)
        off_pts = [(x_, y_, r) for x_, y_, r, ok in zip(px, py, path, inside) if not ok]
        for i, (x_, y_, r) in enumerate(sorted(off_pts, key=lambda z: -z[2]["param"]["eps"])):
            ax.plot(x_, hi, ls="none", marker="o", ms=5.5, mfc="white", mec=COLOR["PGD"], mew=1.3, clip_on=False, zorder=4)
            ax.annotate(r"$\mathring{\varepsilon}$=" + "{:g}: {}".format(r["param"]["eps"], compact(y_)),
                        (max(z[0] for z in off_pts), hi), xytext=(6, -3 - 10 * i), ha="left", va="top", **lab_kw)
        ax.plot(rep["time_adj_mean"], rep["final_mean"], marker="o", ms=9, color=COLOR["PGD"], mec="white", mew=1.6,
                ls="none", zorder=5)
        ax.annotate(r"$\mathring{\varepsilon}$=" + "{:g}".format(path[0]["param"]["eps"]), (px[0], py[0]),
                    xytext=(3, 6), ha="left", va="bottom", **lab_kw)
        for m, r in pts.items():
            ax.plot(r["time_adj_mean"], r["final_mean"], marker=MARKER[m], ms=7.5, color=COLOR[m], mec="white", mew=1.5,
                    ls="none", zorder=6)
        for m, off in (("SPO+", (0, 7, "center", "bottom")), ("AdaGrad", (0, 7, "center", "bottom"))):
            if m in pts:
                ax.annotate(LABEL[m], (pts[m]["time_adj_mean"], pts[m]["final_mean"]), xytext=off[:2], ha=off[2],
                            va=off[3], **lab_kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ticks = [t for t in (5e5, 6e5, 7e5, 8e5, 1e6, 1.2e6, 1.5e6, 2e6, 3e6) if lo <= t <= hi]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_formatter(FuncFormatter(compact))
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.set_xlim(0.02, 1500)
        ax.set_title("{}\nAMLE-PGD: {}={:g}, representative {}={:g}".format(
            SNR_TITLE[snr], PARAM_SYMBOL["eta"], eta, PARAM_SYMBOL["eps"], rep["param"]["eps"]), fontsize=8.5)
        ax.set_xlabel("Total time over the horizon (s)")
        _style_axes(ax)
    axes[0, 0].set_ylabel("Regret")
    handles = [plt.Line2D([], [], color=COLOR["PGD"], marker="o", ms=5, mec="white", mew=1.0,
                          label=r"AMLE-PGD (path over $\mathring{\varepsilon}$)"),
               plt.Line2D([], [], color=COLOR["PGD"], marker="o", ls="none", ms=9, mec="white", mew=1.4,
                          label="AMLE-PGD representative"),
               plt.Line2D([], [], marker="o", ls="none", ms=5.5, mfc="white", mec=COLOR["PGD"], mew=1.3,
                          label="AMLE-PGD beyond plotted range")]
    handles += [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="none", ms=7, mec="white", mew=1.0,
                           label=LABEL[m]) for m in others]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)


def fig_all_params_final(rows, best, snrs, out):
    """Final regret of every (method, hyper-parameter); each panel lists the configurations run at that SNR
    (grids may differ across SNRs). Sized for a full page of the e-companion (6.5 in wide)."""
    per = {snr: sorted([r for r in rows if r["snr"] == snr],
                       key=lambda r: (METHODS.index(r["method"]), _pkey(r["param"]))) for snr in snrs}
    n_max = max(len(v) for v in per.values())
    short = {"PGD": "PGD", "MOSEK": "MOSEK", "ILQX": "ILQX", "SPO+": "SPO+", "AdaGrad": "AdaGrad"}
    height = 0.185 * n_max + 0.9
    fig, axes = plt.subplots(1, len(snrs), figsize=(6.5, height), squeeze=False)
    for j, snr in enumerate(snrs):
        ax = axes[0, j]
        rs = per[snr]
        for i, r in enumerate(rs):
            m = r["method"]
            is_best = best[(snr, m)] is r
            ax.errorbar(r["final_mean"], -i, xerr=1.96 * r["final_se"], fmt=MARKER[m], color=COLOR[m],
                        ms=5.6 if is_best else 3.6, mec="white" if is_best else COLOR[m],
                        mfc=COLOR[m] if is_best else "white", mew=0.8, elinewidth=0.8, capsize=0)
        for i in range(1, len(rs)):
            if rs[i]["method"] != rs[i - 1]["method"]:
                ax.axhline(-i + 0.5, color=AXIS, lw=0.7)
        ax.set_yticks([-i for i in range(len(rs))])
        ax.set_yticklabels(["{}  {}".format(short[r["method"]], param_label(r["method"], r["param"])) for r in rs],
                           fontsize=5.6)
        ax.set_ylim(-n_max + 0.4, 0.6)
        ax.set_title(SNR_TITLE[snr], fontsize=7.5)
        ax.set_xlabel("Final regret (log scale)", fontsize=6.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
        ax.tick_params(axis="x", labelsize=5.8)
        _style_axes(ax)
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="none", ms=4.5, mec="white", label=LABEL[m])
               for m in METHODS if any(r["method"] == m for r in rows)]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", ls="none", ms=4.5, mfc="white", label="other hyperparameters"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6.3, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0 + 0.18 / height))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out)
    plt.close(fig)


def fig_all_params_curves(rows, snrs, out, which):
    """Regret curves of every hyper-parameter.  which='PGD': one column per AMLE-PGD regularization level
    (curves = tolerances); which='bench': one column per other method (curves = its hyper-parameters)."""
    if which == "PGD":
        # column j = the j-th smallest eta0 run at that SNR (the eta0 grids may differ across SNRs)
        etas = {snr: sorted({r["param"]["eta"] for r in rows if r["method"] == "PGD" and r["snr"] == snr})
                for snr in snrs}
        n_col = max((len(v) for v in etas.values()), default=0)
        columns = [("PGD", j) for j in range(n_col)]
    else:
        columns = [(m, None) for m in METHODS if m != "PGD" and any(r["method"] == m for r in rows)]
    if not columns:
        return
    fig, axes = plt.subplots(len(snrs), len(columns), figsize=(1.95 * len(columns) + 0.5, 1.95 * len(snrs) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for i, snr in enumerate(snrs):
        for j, (m, slot) in enumerate(columns):
            ax = axes[i, j]
            eta = None
            if m == "PGD":
                if slot >= len(etas[snr]):
                    ax.set_visible(False)
                    continue
                eta = etas[snr][slot]
            rs = sorted([r for r in rows if r["snr"] == snr and r["method"] == m
                         and (eta is None or r["param"]["eta"] == eta)], key=lambda r: _pkey(r["param"]))
            cols = ramp_levels(max(len(rs), 2))
            for n, r in enumerate(rs):
                T = r["curve_mean"].size
                idx = np.unique(np.linspace(0, T - 1, 300).astype(int))
                lab = ("{}={:g}".format(PARAM_SYMBOL["eps"], r["param"]["eps"]) if m == "PGD"
                       else param_label(m, r["param"]))
                ax.plot(idx + 1, r["curve_mean"][idx], color=cols[n], lw=1.3, label=lab)
            if eta is not None:
                ax.set_title("{}, {}={:g}".format(LABEL[m], PARAM_SYMBOL["eta"], eta), fontsize=8)
                if i == 0 and j == 0:
                    ax.legend(fontsize=6.0, handlelength=1.6, loc="upper left", labelspacing=0.2, borderaxespad=0.2)
            elif i == 0:
                ax.set_title(LABEL[m], fontsize=8.5)
                ax.legend(fontsize=6.0, handlelength=1.6, loc="upper left", labelspacing=0.2, borderaxespad=0.2)
            if j == 0:
                ax.set_ylabel("{}\ncumulative regret".format(SNR_TITLE[snr]))
            if i == len(snrs) - 1:
                ax.set_xlabel("Period $t$")
            ax.xaxis.set_major_formatter(FuncFormatter(compact))
            ax.yaxis.set_major_formatter(FuncFormatter(compact))
            ax.tick_params(axis="x", labelsize=7)
            _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_pgd_iterations(rows, best, snrs, out):
    pgd = [r for r in rows if r["method"] == "PGD"]
    if not pgd:
        return
    fig, axes = plt.subplots(1, len(snrs), figsize=(2.55 * len(snrs) + 0.6, 2.7), sharey=True, squeeze=False)
    for j, snr in enumerate(snrs):
        ax = axes[0, j]
        eta_best = best[(snr, "PGD")]["param"]["eta"]
        sub = sorted([r for r in pgd if r["snr"] == snr and r["param"]["eta"] == eta_best],
                     key=lambda r: r["param"]["eps"])
        cols = ramp_levels(max(len(sub), 2))
        for n, r in enumerate(sub):
            ax.plot(r["N_by_fit"], np.maximum(r["iters_by_fit"], 1), color=cols[n], lw=1.3,
                    label="{}={:g}".format(PARAM_SYMBOL["eps"], r["param"]["eps"]))
        ax.set_yscale("log")
        ax.set_title("{}, {}={:g}".format(SNR_TITLE[snr], PARAM_SYMBOL["eta"], eta_best))
        ax.set_xlabel(r"Exploration sample size $|\mathcal{I}_k|$")
        _style_axes(ax)
    axes[0, 0].set_ylabel("PGD iterations per update")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5, title="Tolerance", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_tables(rows, best, snrs, outdir):
    import csv
    cols = ["snr", "method", "param", "K", "final_mean", "final_se", "exploit_mean", "exploit_se", "time_mean",
            "time_se", "time_adj_mean", "n_updates", "iters_mean", "iters_max", "cert_rate", "fallback_rate"]
    with open(os.path.join(outdir, "summary_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (SNRS.index(r["snr"]), METHODS.index(r["method"]), str(r["param"]))):
            w.writerow([r.get(c, "") if c != "param" else json.dumps(r["param"]) for c in cols])
    def reference_mosek(snr, m, r):
        """AMLE-PGD is compared with AMLE-MOSEK at the same eta0; other methods with the best AMLE-MOSEK."""
        if m in ("PGD", "PGD_min"):
            for x in rows:
                if x["snr"] == snr and x["method"] == "MOSEK" and x["param"]["eta"] == r["param"]["eta"]:
                    return x
            return None
        return best.get((snr, "MOSEK"))

    entries = []   # (snr, key, label, row)
    for snr in snrs:
        for m in METHODS:
            if (snr, m) in best:
                lab = LABEL[m] + (" (representative)" if m == "PGD" else "")
                entries.append((snr, m, lab, best[(snr, m)]))
            if m == "PGD" and (snr, "PGD_min") in best and best[(snr, "PGD_min")] is not best[(snr, "PGD")]:
                entries.append((snr, "PGD_min", "AMLE-PGD (minimum regret)", best[(snr, "PGD_min")]))

    with open(os.path.join(outdir, "summary_best.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + cols + ["regret_ratio_vs_mosek", "regret_ratio_se", "exploit_ratio_vs_mosek",
                                       "exploit_ratio_se", "time_ratio_vs_mosek", "time_ratio_se"])
        for snr, key, lab, r in entries:
            ref = reference_mosek(snr, key, r)
            rr = paired_ratio(r, ref, "finals") if ref is not None else (float("nan"),) * 2
            er = paired_ratio(r, ref, "exploits") if ref is not None else (float("nan"),) * 2
            tr = paired_ratio(r, ref, "times_adj", "times_adj") if ref is not None else (float("nan"),) * 2
            w.writerow([lab] + [r.get(c, "") if c != "param" else json.dumps(r["param"]) for c in cols]
                       + [rr[0], rr[1], er[0], er[1], tr[0], tr[1]])

    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"SNR & Method (hyperparameters) & Final regret & Regret ratio & Exploitation regret"
             r" & Exploit.\ ratio & Time w/o fallback (s) & Time ratio & Iter./update \\", r"\midrule"]
    for snr in snrs:
        for _snr, key, lab, r in [e for e in entries if e[0] == snr]:
            ref = reference_mosek(snr, key, r)
            rr = paired_ratio(r, ref, "finals")[0] if ref is not None else float("nan")
            er = paired_ratio(r, ref, "exploits")[0] if ref is not None else float("nan")
            tr = paired_ratio(r, ref, "times_adj", "times_adj")[0] if ref is not None else float("nan")
            it = "{:.0f}".format(r["iters_mean"]) if key in ("PGD", "PGD_min") else "--"
            lines.append("{} & {} ({}) & {} $\\pm$ {} & {:.3f} & {} $\\pm$ {} & {:.3f} & {:.2f} & {:.3f} & {} \\\\".format(
                snr.capitalize(), lab.replace("%", r"\%"), param_label(key, r["param"]), compact(r["final_mean"]),
                compact(1.96 * r["final_se"]), rr, compact(r["exploit_mean"]), compact(1.96 * r["exploit_se"]), er,
                r["time_adj_mean"], tr, it))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(os.path.join(outdir, "table_best.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_table_regret_time(best, snrs, outdir):
    """Main-text table in the layout of the R1 Table 2 (SNR levels as column groups), with a regret column added:
    best hyper-parameter, final regret (thousands; mean over trials, standard error in parentheses) and
    optimization time per method."""
    order = ["ILQX", "SPO+", "AdaGrad", "MOSEK", "PGD", "PGD_min"]
    row_label = dict(LABEL, PGD="AMLE-PGD (representative)", PGD_min="AMLE-PGD (minimum regret)")
    hp_name = {"lam": r"\mathring{\lambda}", "eta": r"\mathring{\eta}", "eps": r"\mathring{\varepsilon}"}

    def hp(m, param):
        m = "PGD" if m == "PGD_min" else m
        if m in ("SPO+", "AdaGrad"):
            return r"$\mathrm{{lr}}_{{\mathrm{{{}}}}} = {:g}$".format(m, param["lr"])
        return "$" + ", ".join("{} = {:g}".format(hp_name[k], param[k])
                               for k in sorted(param, key=_ORDER.index)) + "$"

    lowest = {snr: min(best[(snr, m)]["final_mean"] for m in order if (snr, m) in best) for snr in snrs}
    explore = {snr: best[(snr, "MOSEK")]["final_mean"] - best[(snr, "MOSEK")]["exploit_mean"]
               for snr in snrs if (snr, "MOSEK") in best}
    lines = [
        "% Suggested caption: Regret and computational time ($T = 10{,}000$) at the best hyperparameters across",
        "% different SNR levels.  Regret in thousands: mean over {} trials (standard error); lowest regret per".format(
            best[(snrs[0], "MOSEK")]["K"]),
        "% SNR in bold.  Total time: time spent updating the estimator over the whole horizon (s), mean over trials:",
        "% {:.0f} refits (end of each exploration phase) for ILQX, SPO+, AMLE-MOSEK and AMLE-PGD; {:,} recursive updates".format(
            best[(snrs[0], "MOSEK")]["n_updates"], best[(snrs[0], "MOSEK")]["curve_mean"].size),
        "% (one per period) for AdaGrad; pricing decisions excluded.  For AMLE-MOSEK and ILQX, refits in which MOSEK",
        "% stalled are charged the median time of a regular solve.  AMLE-PGD (representative): largest tolerance",
        "% within 1% of its minimum regret at the best eta0.  The exploration periods, identical",
        "% for all methods, contribute " + ", ".join("{}K ({})".format(round(v / 1e3, 1), snr) for snr, v in explore.items())
        + " to every method's regret.",
        r"\begin{tabular}{l" + "ccc" * len(snrs) + "}",
        r"\toprule",
        " & " + " & ".join(r"\multicolumn{3}{c}{" + SNR_TITLE[snr].split(" (")[0] + "}" for snr in snrs) + r" \\",
        " ".join(r"\cmidrule(lr){{{}-{}}}".format(2 + 3 * i, 4 + 3 * i) for i in range(len(snrs))),
        "Method & " + " & ".join([r"Best Hyperparameter & Regret ($\times 10^3$) & Total time (s)"] * len(snrs)) + r" \\",
        r"\midrule",
    ]
    for m in order:
        if m == "PGD_min" and all(best.get((snr, "PGD_min")) is best.get((snr, "PGD")) for snr in snrs):
            continue
        cells = [row_label[m]]
        for snr in snrs:
            r = best.get((snr, m))
            if r is None:
                cells += ["--"] * 3
                continue
            reg = "{:.1f} ({:.1f})".format(r["final_mean"] / 1e3, r["final_se"] / 1e3)
            if r["final_mean"] == lowest[snr]:
                reg = r"\textbf{" + reg + "}"
            cells += [hp(m, r["param"]), reg, "{:.2f}".format(r["time_adj_mean"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(outdir, "table_regret_time.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = os.path.join(args.root, args.tag)
    outdir = args.out or os.path.join(HERE, "figures", args.tag)
    os.makedirs(outdir, exist_ok=True)
    rows = aggregate(load(root))
    snrs = [s for s in SNRS if any(r["snr"] == s for r in rows)]
    best = best_rows(rows)
    fig_best_regret(rows, best, snrs, os.path.join(outdir, "fig_best_regret.pdf"))
    fig_tradeoff(rows, best, snrs, os.path.join(outdir, "fig_tradeoff_eps.pdf"))
    fig_regret_time(rows, best, snrs, os.path.join(outdir, "fig_regret_time.pdf"))
    fig_all_params_final(rows, best, snrs, os.path.join(outdir, "fig_all_params_final.pdf"))
    fig_all_params_curves(rows, snrs, os.path.join(outdir, "fig_all_params_curves_pgd.pdf"), "PGD")
    fig_all_params_curves(rows, snrs, os.path.join(outdir, "fig_all_params_curves_bench.pdf"), "bench")
    fig_pgd_iterations(rows, best, snrs, os.path.join(outdir, "fig_pgd_iterations.pdf"))
    write_tables(rows, best, snrs, outdir)
    write_table_regret_time(best, snrs, outdir)
    print("wrote figures and tables to", outdir)


if __name__ == "__main__":
    main()
