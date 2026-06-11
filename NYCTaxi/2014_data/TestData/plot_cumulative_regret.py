"""
Plot cumulative regret and total-reward bar charts from PGD_ILQX.py outputs.

Input layout (written by PGD_ILQX.py under TestData/sim_output/):

    sim_output/
    └── <scenario>/                    # e.g. pair_1p5_2p0__noise_05
        └── repeat_<k>/
            └── eta<η>_lam<λ>_stopeps<ε>_etapow<a>/
                └── revenue_regret.csv

Each revenue_regret.csv must contain columns used by load_simulation_results():
arrival_idx, regret_pgd, regret_cvx, regret_original, reward_pgd, reward_cvx,
reward_original.

Outputs (default paths under sim_output/):
  - cumulative_regret_styled.pdf   — all runs, cumulative regret curves
  - total_reward_bars.pdf          — grouped bar chart of mean total reward
  - cumulative_regret_best.pdf     — best ILQX + best AMLE-PGD per scenario + TSP
  - total_reward_bars_best.pdf     — bar chart for best configs only

Hyperparameter grids default to PGD_ILQX.py:
  HYPERPARAMETER_PAIRS  → PGD_BAR_ETA_VALUES, ILQX_BAR_LAM_VALUES
  ETA_K_POWER_VALUES    → PGD_BAR_A_VALUES
  STOP_EPS_VALUES       → DEFAULT_STOP_EPS_VALUES

Usage (from repo root):
    python 2014_data/TestData/plot_cumulative_regret.py
    python 2014_data/TestData/plot_cumulative_regret.py --input-root 2014_data/TestData/sim_output
    python 2014_data/TestData/plot_cumulative_regret.py --stop-eps 1.0 --stop-eps 1.5 --show
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "sim_output"
DEFAULT_OUT = OUTPUT_ROOT / "cumulative_regret_styled.pdf"
DEFAULT_REWARD_OUT = OUTPUT_ROOT / "total_reward_bars.pdf"
DEFAULT_BEST_OUT = OUTPUT_ROOT / "cumulative_regret_best.pdf"
DEFAULT_BEST_REWARD_OUT = OUTPUT_ROOT / "total_reward_bars_best.pdf"

# Shared figure size (inches) for regret curve and reward bar plots
FIG_W, FIG_H = 14, 9

SCENARIO_ORDER = [
    "pair_1p5__noise_05",
    "pair_1p5_2p0__noise_05",
    "pair_1p5_2p0__noise_10",
    "pair_1p5_2p0__noise_20",
]

SCENARIO_LABEL = {
    "pair_1p5__noise_05": "1.5×, noise 5%",
    "pair_1p5_2p0__noise_05": "1.5×/2.0×, noise 5%",
    "pair_1p5_2p0__noise_10": "1.5×/2.0×, noise 10%",
    "pair_1p5_2p0__noise_20": "1.5×/2.0×, noise 20%",
}

# Short beta-pair headers for reward bar chart x-axis
SCENARIO_BETA_PAIR = {
    "pair_1p5__noise_05": r"1.5$\times$, 5% noise",
    "pair_1p5_2p0__noise_05": r"1.5$\times$/2.0$\times$, 5% noise",
    "pair_1p5_2p0__noise_10": r"1.5$\times$/2.0$\times$, 10% noise",
    "pair_1p5_2p0__noise_20": r"1.5$\times$/2.0$\times$, 20% noise",
}

# Match PGD_ILQX.py HYPERPARAMETER_PAIRS: (η, λ) ∈ {(5e-2,5e-2), (1e-1,1e-1), (5e-1,5e-1)}.
PGD_BAR_ETA_VALUES = [5e-2, 1e-1, 5e-1]
PGD_BAR_A_VALUES = [0.0, 0.25, 0.5]
# Ascending λ for ILQX bars (left → right within each scenario group).
ILQX_BAR_LAM_VALUES = [5e-2, 1e-1, 5e-1]

# Default stop-eps filter. Keep this aligned with PGD_ILQX.py STOP_EPS_VALUES.
DEFAULT_STOP_EPS_VALUES = [1.5e0, 1e0]

# Custom dashed line: (offset, (dash_len, gap_len)) — larger gap than default '--'
DASH_LINESTYLE = (0, (6, 10))

# PGD line colors by scenario
PGD_SCENARIO_COLOR = {
    "pair_1p5__noise_05": "#C62828",  # dark red 
    "pair_1p5_2p0__noise_05": "#E57373",  # medium red (between light and dark)
    "pair_1p5_2p0__noise_10": "#FFAAAA",  # light red
    "pair_1p5_2p0__noise_20": "#FFD6D6",  # very light red
}

# TSP: no markers; strong green contrast
TSP_SCENARIO_COLOR = {
    "pair_1p5__noise_05": "#1B5E20",
    "pair_1p5_2p0__noise_05": "#43A047",
    "pair_1p5_2p0__noise_10": "#A5D6A7",
    "pair_1p5_2p0__noise_20": "#DCEDC8",
}

# ILQX: blue family, strong contrast across scenarios
ILQX_SCENARIO_COLOR = {
    "pair_1p5__noise_05": "#0D47A1",
    "pair_1p5_2p0__noise_05": "#1976D2",
    "pair_1p5_2p0__noise_10": "#64B5F6",
    "pair_1p5_2p0__noise_20": "#90CAF9",
}

# PGD markers: (eta_coeff, eta_k_power) -> marker
PGD_MARKER = {
    (5e-2, 0.0): "*",
    (5e-2, 0.25): "o",
    (5e-2, 0.5): "^",
    (1e-1, 0.0): "D",
    (1e-1, 0.25): "v",
    (1e-1, 0.5): "p",
    (5e-1, 0.0): "x",
    (5e-1, 0.25): "+",
    (5e-1, 0.5): "s",
}

# ILQX markers: lambda_factor -> marker
ILQX_MARKER = {
    5e-2: "*",
    1e-1: "D",
    5e-1: "+",
}

# Readability-focused style mapping for dense PGD bars.
PGD_ETA_BAR_COLOR = {
    5e-2: "#8E24AA",  # purple
    1e-1: "#FB8C00",  # orange
    5e-1: "#D32F2F",  # red
}
STOP_EPS_HATCH = {
    1.5e0: "",
    1e0: "///",
}
# Lighten PGD bar fill by a (same η hue, three distinct shades).
PGD_A_WHITE_BLEND = {0.0: 0.45, 0.25: 0.18, 0.5: 0.0}
# Background bands per (a, eps-pair): soft cream → amber → light orange.
PGD_A_PAIR_BG = ["#FFF8E1", "#FFE082", "#FFCC80"]
# TSP / ILQX background bands (strategy blocks encoded visually, not via x-axis text).
TSP_BAND_BG = "#C8E6C9"
# ILQX bar colors: same blue family across all beta simulators; λ → light / medium / dark.
ILQX_LAM_BAR_COLOR = {
    5e-2: "#BBDEFB",
    1e-1: "#42A5F5",
    5e-1: "#0D47A1",
}
ILQX_LAM_BAND_BG = ["#E3F2FD", "#BBDEFB", "#90CAF9"]

RUN_DIR_RE = re.compile(
    r"eta(?P<eta>[^_]+)_lam(?P<lam>[^_]+)_stopeps(?P<seps>[^_]+)_etapow(?P<pow>[^/]+)"
)


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _ilqx_bar_color(lam: float) -> str:
    return next(
        (c for lam_key, c in ILQX_LAM_BAR_COLOR.items() if _approx_equal(lam, lam_key)),
        ILQX_LAM_BAR_COLOR[5e-2],
    )


def discover_runs(
    root: Path,
    eta_values: list[float],
    lam_values: list[float],
    eta_pow_values: list[float],
    stop_eps_values: list[float] | None,
) -> list[dict]:
    runs: list[dict] = []
    for scenario in SCENARIO_ORDER:
        scenario_root = root / scenario
        if not scenario_root.is_dir():
            continue
        for csv_path in sorted(scenario_root.glob("**/revenue_regret.csv")):
            m = RUN_DIR_RE.search(str(csv_path))
            if m is None:
                continue

            eta = float(m.group("eta"))
            lam = float(m.group("lam"))
            seps = float(m.group("seps"))
            eta_pow = float(m.group("pow"))

            if stop_eps_values is not None and not any(
                _approx_equal(seps, v) for v in stop_eps_values
            ):
                continue
            if not any(_approx_equal(eta, v) for v in eta_values):
                continue
            if not any(_approx_equal(lam, v) for v in lam_values):
                continue
            if not any(_approx_equal(eta_pow, v) for v in eta_pow_values):
                continue

            rep_m = re.search(r"repeat_(\d+)", str(csv_path))
            repeat_idx = int(rep_m.group(1)) if rep_m else 0

            runs.append(
                {
                    "scenario": scenario,
                    "eta": eta,
                    "lam": lam,
                    "stop_eps": seps,
                    "eta_pow": eta_pow,
                    "repeat_idx": repeat_idx,
                    "csv_path": csv_path,
                }
            )
    return runs


def _blend_hex(hex_color: str, white_frac: float) -> str:
    """Blend hex color toward white; white_frac=1 → white."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    w = max(0.0, min(1.0, float(white_frac)))
    return f"#{int(r + (255 - r) * w):02x}{int(g + (255 - g) * w):02x}{int(b + (255 - b) * w):02x}"


def _fmt_coeff_math(x: float) -> str:
    """Format coefficient for mathtext labels, e.g. 5e-4."""
    s = f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant}\\mathrm{{e}}{{{exp}}}"
    return s


def _pgd_label(scenario: str, eta: float, eta_pow: float) -> str:
    return (
        f"AMLE-PGD [{SCENARIO_LABEL[scenario]}] "
        rf"$\mathring{{\eta}}={_fmt_coeff_math(eta)}$, $a={eta_pow:g}$"
    )


def _pgd_label_with_stop_eps(scenario: str, eta: float, eta_pow: float, stop_eps: float) -> str:
    return (
        f"AMLE-PGD [{SCENARIO_LABEL[scenario]}] "
        rf"$\mathring{{\eta}}={_fmt_coeff_math(eta)}$, $a={eta_pow:g}$, "
        rf"$\mathring{{\epsilon}}={_fmt_coeff_math(stop_eps)}$"
    )


def _select_representative_pgd_keys(
    mean_results: dict[tuple, pd.DataFrame],
) -> list[tuple]:
    """
    Deduplicate PGD curves by (scenario, eta, stop_eps, eta_pow), ignoring lambda.
    If multiple keys remain (different lambda), keep the one with smallest final cumulative PGD regret.
    """
    grouped: dict[tuple, list[tuple]] = {}
    for k in mean_results.keys():
        scenario, eta, _lam, stop_eps, eta_pow = k
        rep_key = (scenario, eta, stop_eps, eta_pow)
        grouped.setdefault(rep_key, []).append(k)

    selected: list[tuple] = []
    for candidates in grouped.values():
        best_k = min(
            candidates,
            key=lambda x: float(mean_results[x]["cumulative_pgd"].iloc[-1])
            if "cumulative_pgd" in mean_results[x].columns and len(mean_results[x]) > 0
            else float("inf"),
        )
        selected.append(best_k)

    selected.sort(
        key=lambda k: (
            SCENARIO_ORDER.index(k[0]) if k[0] in SCENARIO_ORDER else 99,
            k[1],
            k[4],
            k[3],
        )
    )
    return selected


def _ilqx_label(scenario: str, lam: float) -> str:
    return (
        f"ILQX [{SCENARIO_LABEL[scenario]}] "
        rf"$\mathring{{\lambda}}={_fmt_coeff_math(lam)}$"
    )


def load_simulation_results(runs: list[dict]) -> tuple[dict[tuple, pd.DataFrame], pd.DataFrame]:
    """Load mean cumulative regret curves and total-reward summary for bar chart."""
    grouped: dict[tuple, list[tuple[int, pd.DataFrame]]] = {}
    for run in runs:
        key = (run["scenario"], run["eta"], run["lam"], run["stop_eps"], run["eta_pow"])
        usecols = [
            "arrival_idx",
            "regret_pgd",
            "regret_cvx",
            "regret_original",
            "reward_pgd",
            "reward_cvx",
            "reward_original",
        ]
        df = pd.read_csv(run["csv_path"], usecols=usecols)
        for col in ("regret_pgd", "regret_cvx", "regret_original"):
            df[f"cumulative_{col.replace('regret_', '')}"] = df[col].cumsum()
        grouped.setdefault(key, []).append((int(run.get("repeat_idx", 0)), df))

    mean_results: dict[tuple, pd.DataFrame] = {}
    reward_rows: list[dict] = []
    tsp_done: set[str] = set()
    ilqx_done: set[tuple[str, float]] = set()

    for key, repeat_dfs in grouped.items():
        scenario, eta, lam, stop_eps, eta_pow = key
        aligned = []
        for repeat_idx, df in repeat_dfs:
            tmp = df.copy()
            tmp["_repeat"] = int(repeat_idx)
            aligned.append(tmp)
        stacked = pd.concat(aligned, ignore_index=True)
        mean_df = (
            stacked.groupby("arrival_idx", as_index=False)
            .agg(
                {
                    **{c: "mean" for c in stacked.columns if c.startswith("cumulative_")},
                }
            )
            .sort_values("arrival_idx")
        )
        mean_results[key] = mean_df

        totals = {
            "pgd": stacked.groupby("_repeat")["reward_pgd"].sum().mean(),
            "cvx": stacked.groupby("_repeat")["reward_cvx"].sum().mean(),
            "original": stacked.groupby("_repeat")["reward_original"].sum().mean(),
        }

        if scenario not in tsp_done:
            tsp_done.add(scenario)
            reward_rows.append(
                {
                    "scenario": scenario,
                    "strategy": "TSP",
                    "eta": np.nan,
                    "lam": np.nan,
                    "eta_pow": np.nan,
                    "sort_key": (0, 0, 0),
                    "bar_label": f"TSP [{SCENARIO_LABEL[scenario]}]",
                    "color": TSP_SCENARIO_COLOR[scenario],
                    "total_reward": totals["original"],
                }
            )

        ilqx_key = (scenario, lam)
        if ilqx_key not in ilqx_done and lam in ILQX_MARKER:
            ilqx_done.add(ilqx_key)
            reward_rows.append(
                {
                    "scenario": scenario,
                    "strategy": "ILQX",
                    "eta": np.nan,
                    "lam": lam,
                    "stop_eps": stop_eps,
                    "eta_pow": np.nan,
                    "sort_key": (1, lam, 0),
                    "bar_label": _ilqx_label(scenario, lam),
                    "color": _ilqx_bar_color(lam),
                    "total_reward": totals["cvx"],
                }
            )

        if (eta, eta_pow) in PGD_MARKER:
            reward_rows.append(
                {
                    "scenario": scenario,
                    "strategy": "PGD",
                    "eta": eta,
                    "lam": lam,
                    "stop_eps": stop_eps,
                    "eta_pow": eta_pow,
                    "sort_key": (2, eta, eta_pow, stop_eps),
                    "bar_label": _pgd_label_with_stop_eps(scenario, eta, eta_pow, stop_eps),
                    "color": PGD_SCENARIO_COLOR[scenario],
                    "total_reward": totals["pgd"],
                }
            )

    reward_df = pd.DataFrame(reward_rows)
    if not reward_df.empty:
        scenario_rank = {s: i for i, s in enumerate(SCENARIO_ORDER)}
        reward_df = (
            reward_df.assign(_scn_rank=reward_df["scenario"].map(scenario_rank))
            .sort_values(["_scn_rank", "sort_key"])
            .drop(columns="_scn_rank")
            .reset_index(drop=True)
        )

    return mean_results, reward_df


def load_mean_cumulative_regret(runs: list[dict]) -> dict[tuple, pd.DataFrame]:
    mean_results, _ = load_simulation_results(runs)
    return mean_results


def _markevery(n_points: int, series_idx: int, n_series: int) -> tuple[int, int]:
    """Stagger marker positions so overlapping curves remain distinguishable."""
    stride = max(1, n_points // 28)
    if n_series <= 1:
        return (0, stride)
    # Larger spread: step ~ stride/3 per series (not stride/n_series), plus prime shift
    stagger_step = max(1, (2 * stride) // max(3, n_series))
    offset = (series_idx * stagger_step + series_idx * 37) % stride
    return (offset, stride)


def _draw_regret_curves(
    ax,
    mean_results: dict[tuple, pd.DataFrame],
) -> int:
    """Plot TSP / ILQX / PGD cumulative regret on a single axes. Returns n_points."""
    pgd_keys = _select_representative_pgd_keys(mean_results)
    n_marker_series = (
        sum(1 for k in pgd_keys if (k[1], k[4]) in PGD_MARKER)
        + len({(k[0], k[2]) for k in pgd_keys})
    )

    tsp_done: set[str] = set()
    ilqx_done: set[tuple[str, float]] = set()
    marker_idx = 0
    n_points = max(len(df) for df in mean_results.values()) if mean_results else 1

    for scenario in SCENARIO_ORDER:
        sample_key = next((k for k in pgd_keys if k[0] == scenario), None)
        if sample_key is None or scenario in tsp_done:
            continue
        df = mean_results[sample_key]
        tsp_done.add(scenario)
        ax.plot(
            df["arrival_idx"],
            df["cumulative_original"],
            color=TSP_SCENARIO_COLOR[scenario],
            linestyle=DASH_LINESTYLE,
            linewidth=2.0,
            label=f"TSP [{SCENARIO_LABEL[scenario]}]",
            zorder=2,
        )

    for key in pgd_keys:
        scenario, _, lam, _, _ = key
        ilqx_key = (scenario, lam)
        if ilqx_key in ilqx_done or lam not in ILQX_MARKER:
            continue
        ilqx_done.add(ilqx_key)
        df = mean_results[key]
        me = _markevery(n_points, marker_idx, n_marker_series)
        marker_idx += 1
        ax.plot(
            df["arrival_idx"],
            df["cumulative_cvx"],
            color=ILQX_SCENARIO_COLOR[scenario],
            marker=ILQX_MARKER[lam],
            markevery=me,
            markersize=7,
            linewidth=1.4,
            linestyle=DASH_LINESTYLE,
            label=_ilqx_label(scenario, lam),
            zorder=4,
        )

    for key in pgd_keys:
        scenario, eta, _, stop_eps, eta_pow = key
        marker_key = (eta, eta_pow)
        if marker_key not in PGD_MARKER:
            continue
        df = mean_results[key]
        me = _markevery(n_points, marker_idx, n_marker_series)
        marker_idx += 1
        ax.plot(
            df["arrival_idx"],
            df["cumulative_pgd"],
            color=PGD_SCENARIO_COLOR[scenario],
            marker=PGD_MARKER[marker_key],
            markevery=me,
            markersize=6,
            linewidth=1.2,
            linestyle="-" if _approx_equal(stop_eps, DEFAULT_STOP_EPS_VALUES[0]) else (0, (3, 4)),
            label=_pgd_label_with_stop_eps(scenario, eta, eta_pow, stop_eps),
            zorder=3,
        )

    return n_points


def plot_cumulative_regret(
    mean_results: dict[tuple, pd.DataFrame],
    out_path: Path,
    show: bool = False,
) -> None:
    # Use a larger canvas for dense bar annotations and legends.
    fig, ax = plt.subplots(figsize=(20, 10.5))
    _draw_regret_curves(ax, mean_results)

    ax.set_xlabel("Arrival Index (t)", fontsize=16)
    ax.set_ylabel("Regret", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        fontsize=10,
        loc="upper left",
        ncol=2,
        framealpha=0.92,
        handlelength=1.4,
        columnspacing=0.9,
        labelspacing=0.3,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cumulative_regret_best(
    mean_results: dict[tuple, pd.DataFrame],
    best_df: pd.DataFrame,
    out_path: Path,
    show: bool = False,
) -> None:
    """Best-only plot: exactly 3 curves per scenario (TSP + best ILQX + best PGD)."""
    if best_df.empty:
        print("No best-only rows to plot.")
        return

    keys = list(mean_results.keys())
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    def _key_for_ilqx(scenario: str, lam: float):
        return next((k for k in keys if k[0] == scenario and _approx_equal(k[2], lam)), None)

    def _key_for_pgd(scenario: str, eta: float, eta_pow: float, stop_eps: float):
        # Choose any key with this (scenario, eta, eta_pow, stop_eps); if multiple lambdas exist,
        # prefer the one with smallest final cumulative PGD regret.
        cand = [
            k
            for k in keys
            if k[0] == scenario
            and _approx_equal(k[1], eta)
            and _approx_equal(k[4], eta_pow)
            and _approx_equal(k[3], stop_eps)
        ]
        if not cand:
            return None
        return min(cand, key=lambda k: float(mean_results[k]["cumulative_pgd"].iloc[-1]))

    for scenario in SCENARIO_ORDER:
        sub = best_df.loc[best_df["scenario"] == scenario]
        if sub.empty:
            continue

        ilqx_row = sub.loc[sub["strategy"] == "ILQX"]
        pgd_row = sub.loc[sub["strategy"] == "PGD"]
        if ilqx_row.empty or pgd_row.empty:
            continue

        lam = float(ilqx_row.iloc[0]["lam"])
        eta = float(pgd_row.iloc[0]["eta"])
        eta_pow = float(pgd_row.iloc[0]["eta_pow"])
        stop_eps = float(pgd_row.iloc[0]["stop_eps"])

        ilqx_key = _key_for_ilqx(scenario, lam)
        pgd_key = _key_for_pgd(scenario, eta, eta_pow, stop_eps)
        base_key = ilqx_key or pgd_key
        if base_key is None:
            continue

        df_base = mean_results[base_key]
        ax.plot(
            df_base["arrival_idx"],
            df_base["cumulative_original"],
            color=TSP_SCENARIO_COLOR[scenario],
            linestyle=DASH_LINESTYLE,
            linewidth=2.0,
            label=f"TSP [{SCENARIO_LABEL[scenario]}]",
            zorder=2,
        )

        if ilqx_key is not None:
            df = mean_results[ilqx_key]
            ax.plot(
                df["arrival_idx"],
                df["cumulative_cvx"],
                color=ILQX_SCENARIO_COLOR[scenario],
                marker=ILQX_MARKER.get(lam, None),
                markevery=(0, max(1, len(df) // 28)),
                markersize=7,
                linewidth=1.4,
                linestyle=DASH_LINESTYLE,
                label=_ilqx_label(scenario, lam),
                zorder=4,
            )

        if pgd_key is not None:
            df = mean_results[pgd_key]
            mk = PGD_MARKER.get((eta, eta_pow), None)
            ax.plot(
                df["arrival_idx"],
                df["cumulative_pgd"],
                color=PGD_SCENARIO_COLOR[scenario],
                marker=mk,
                markevery=(5, max(1, len(df) // 28)),
                markersize=6,
                linewidth=1.2,
                linestyle="-" if _approx_equal(stop_eps, DEFAULT_STOP_EPS_VALUES[0]) else (0, (3, 4)),
                label=_pgd_label_with_stop_eps(scenario, eta, eta_pow, stop_eps),
                zorder=3,
            )

    ax.set_xlabel("Arrival Index (t)", fontsize=16)
    ax.set_ylabel("Regret", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        fontsize=10,
        loc="upper left",
        ncol=2,
        framealpha=0.92,
        handlelength=1.4,
        columnspacing=0.9,
        labelspacing=0.3,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _select_reward_bar_rows(reward_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Fixed bar order per scenario: TSP, ILQX×|Λ|, AMLE-PGD×|Η|×|A|."""
    rows: list[pd.Series] = []
    sub = reward_df.loc[reward_df["scenario"] == scenario]

    tsp = sub.loc[sub["strategy"] == "TSP"]
    if not tsp.empty:
        rows.append(tsp.iloc[0])

    for lam in ILQX_BAR_LAM_VALUES:
        hit = sub.loc[(sub["strategy"] == "ILQX") & sub["lam"].apply(lambda v: _approx_equal(v, lam))]
        if not hit.empty:
            rows.append(hit.iloc[0])

    for eta in PGD_BAR_ETA_VALUES:
        for a in PGD_BAR_A_VALUES:
            for seps in DEFAULT_STOP_EPS_VALUES:
                hit = sub.loc[
                    (sub["strategy"] == "PGD")
                    & sub["eta"].apply(lambda v: _approx_equal(v, eta))
                    & sub["eta_pow"].apply(lambda v: _approx_equal(v, a))
                    & sub["stop_eps"].apply(lambda v: _approx_equal(v, seps))
                ]
                if not hit.empty:
                    rows.append(hit.iloc[0])

    return pd.DataFrame(rows)


def plot_total_reward(
    reward_df: pd.DataFrame,
    out_path: Path,
    show: bool = False,
) -> None:
    """Single-panel total-reward bars with grouped x-axis annotations."""
    scenarios = [s for s in SCENARIO_ORDER if s in set(reward_df["scenario"])]
    if not scenarios:
        print("No reward rows to plot.")
        return

    fig, ax = plt.subplots(figsize=(21, 10.5))
    group_gap = 1.2
    bar_x: list[float] = []
    bar_h: list[float] = []
    bar_c: list[str] = []
    bar_hatch: list[str] = []
    bar_eps: list[float] = []
    group_spans: list[tuple[float, float, str]] = []
    tsp_bands: list[tuple[float, float]] = []
    ilqx_lam_bands: list[tuple[float, float, int]] = []
    pgd_a_pair_bands: list[tuple[float, float, int]] = []
    pgd_pair_boundaries: list[float] = []

    x_cursor = 0.0
    for scenario in scenarios:
        sub = _select_reward_bar_rows(reward_df, scenario)
        if sub.empty:
            continue
        g_start = x_cursor
        pgd_by_eta: dict[float, list[float]] = {}
        pgd_positions_in_scenario: list[float] = []

        for _, row in sub.iterrows():
            bar_x.append(x_cursor)
            bar_h.append(row["total_reward"])
            strategy = row["strategy"]
            if strategy == "TSP":
                bar_c.append(TSP_SCENARIO_COLOR[scenario])
                bar_hatch.append("")
                bar_eps.append(np.nan)
                tsp_bands.append((x_cursor - 0.44, x_cursor + 0.44))
            elif strategy == "ILQX":
                lam_val = float(row["lam"])
                lam_idx = next(
                    (i for i, v in enumerate(ILQX_BAR_LAM_VALUES) if _approx_equal(lam_val, v)),
                    0,
                )
                bar_c.append(_ilqx_bar_color(lam_val))
                bar_hatch.append("")
                bar_eps.append(np.nan)
                ilqx_lam_bands.append((x_cursor - 0.44, x_cursor + 0.44, lam_idx))
            else:
                eta_val = float(row["eta"])
                seps_val = float(row["stop_eps"])
                eta_color = next(
                    (clr for eta_key, clr in PGD_ETA_BAR_COLOR.items() if _approx_equal(eta_val, eta_key)),
                    "#C62828",
                )
                a_blend = next(
                    (b for a_key, b in PGD_A_WHITE_BLEND.items() if _approx_equal(float(row["eta_pow"]), a_key)),
                    0.0,
                )
                eta_color = _blend_hex(eta_color, a_blend)
                hatch = next(
                    (h for eps_key, h in STOP_EPS_HATCH.items() if _approx_equal(seps_val, eps_key)),
                    "xx",
                )
                bar_c.append(eta_color)
                bar_hatch.append(hatch)
                bar_eps.append(seps_val)
                pgd_by_eta.setdefault(eta_val, []).append(x_cursor)
                pgd_positions_in_scenario.append(x_cursor)

            x_cursor += 1.0

        g_end = x_cursor - 1.0
        group_spans.append(((g_start + g_end) / 2.0, g_start, g_end, scenario))

        if pgd_by_eta:
            # Visual separators between adjacent eps-pairs for each a.
            for j in range(2, len(pgd_positions_in_scenario), 2):
                boundary_x = (pgd_positions_in_scenario[j - 1] + pgd_positions_in_scenario[j]) / 2.0
                pgd_pair_boundaries.append(boundary_x)
            # Background bands per (a, eps-pair); eta/a encoded by color + legend.
            n_a = len(PGD_BAR_A_VALUES)
            n_seps = len(DEFAULT_STOP_EPS_VALUES)
            block = n_a * n_seps  # bars per eta
            for eta_idx in range(len(PGD_BAR_ETA_VALUES)):
                for a_idx in range(n_a):
                    i0 = eta_idx * block + a_idx * n_seps
                    i1 = i0 + n_seps - 1
                    if i1 < len(pgd_positions_in_scenario):
                        x0 = pgd_positions_in_scenario[i0]
                        x1 = pgd_positions_in_scenario[i1]
                        pgd_a_pair_bands.append((x0 - 0.44, x1 + 0.44, a_idx))

        x_cursor += group_gap

    y_min, y_max = float(min(bar_h)), float(max(bar_h))
    y_span = y_max - y_min
    y_pad = max(y_span * 0.06, y_max * 0.0015)
    y_lo = y_min - y_pad
    for x0, x1 in tsp_bands:
        ax.axvspan(
            x0,
            x1,
            ymin=0.0,
            ymax=1.0,
            facecolor=TSP_BAND_BG,
            alpha=0.45,
            zorder=0,
            clip_on=True,
        )
    for x0, x1, lam_idx in ilqx_lam_bands:
        ax.axvspan(
            x0,
            x1,
            ymin=0.0,
            ymax=1.0,
            facecolor=ILQX_LAM_BAND_BG[lam_idx % len(ILQX_LAM_BAND_BG)],
            alpha=0.40,
            zorder=0,
            clip_on=True,
        )
    for x0, x1, a_idx in pgd_a_pair_bands:
        ax.axvspan(
            x0,
            x1,
            ymin=0.0,
            ymax=1.0,
            facecolor=PGD_A_PAIR_BG[a_idx % len(PGD_A_PAIR_BG)],
            alpha=0.42,
            zorder=0,
            clip_on=True,
        )

    bars = ax.bar(
        bar_x,
        bar_h,
        color=bar_c,
        hatch=None,
        width=0.78,
        edgecolor="0.35",
        linewidth=0.6,
    )
    # Style only bar rectangles (ax.patches also includes axvspan backgrounds).
    for patch, hatch, seps in zip(bars.patches, bar_hatch, bar_eps):
        patch.set_edgecolor("white")
        if _approx_equal(seps, 5e-1):
            patch.set_linewidth(1.6)
        elif _approx_equal(seps, 1e0):
            patch.set_linewidth(0.8)
        else:
            patch.set_linewidth(0.6)
        if hatch and not _approx_equal(seps, 1.5e0):
            patch.set_hatch(hatch)
        else:
            patch.set_hatch("")

    # eps=1e-2: overlay white hatching so texture is clearer on dark bars
    eps_1e2_idx = [i for i, seps in enumerate(bar_eps) if _approx_equal(seps, 5e-1)]
    if eps_1e2_idx:
        eps_1e2_x = [bar_x[i] for i in eps_1e2_idx]
        eps_1e2_h = [bar_h[i] for i in eps_1e2_idx]
        ax.bar(
            eps_1e2_x,
            eps_1e2_h,
            color="none",
            width=0.78,
            edgecolor="white",
            linewidth=0.0,
            hatch=STOP_EPS_HATCH[5e-1],
            zorder=4,
        )

    ax.set_ylim(y_lo, y_max + y_pad)

    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    for bx in pgd_pair_boundaries:
        ax.axvline(bx, color="0.70", linewidth=0.7, alpha=0.7, zorder=1)

    # Beta-simulator section titles only (no per-bar x-axis text).
    trans = ax.get_xaxis_transform()
    for cx, _x0, _x1, scenario in group_spans:
        ax.text(cx, -0.08, SCENARIO_BETA_PAIR[scenario], transform=trans, ha="center", va="top",
                fontsize=20, clip_on=False)

    legend_items = [
        Patch(facecolor=TSP_BAND_BG, edgecolor="white", label="TSP"),
        Patch(facecolor=ILQX_LAM_BAR_COLOR[5e-2], edgecolor="white",
              label=r"ILQX $\mathring{{\lambda}}=5\mathrm{e}{-2}$"),
        Patch(facecolor=ILQX_LAM_BAR_COLOR[1e-1], edgecolor="white",
              label=r"ILQX $\mathring{{\lambda}}=1\mathrm{e}{-1}$"),
        Patch(facecolor=ILQX_LAM_BAR_COLOR[5e-1], edgecolor="white",
              label=r"ILQX $\mathring{{\lambda}}=5\mathrm{e}{-1}$"),
        Patch(facecolor=PGD_ETA_BAR_COLOR[5e-2], edgecolor="white", label=r"PGD $\mathring{\eta}=5\mathrm{e}{-2}$"),
        Patch(facecolor=PGD_ETA_BAR_COLOR[1e-1], edgecolor="white", label=r"PGD $\mathring{\eta}=1\mathrm{e}{-1}$"),
        Patch(facecolor=PGD_ETA_BAR_COLOR[5e-1], edgecolor="white", label=r"PGD $\mathring{\eta}=5\mathrm{e}{-1}$"),
        Patch(facecolor=PGD_A_PAIR_BG[0], edgecolor="white", label=r"$a=0$"),
        Patch(facecolor=PGD_A_PAIR_BG[1], edgecolor="white", label=r"$a=0.25$"),
        Patch(facecolor=PGD_A_PAIR_BG[2], edgecolor="white", label=r"$a=0.5$"),
        Patch(facecolor="#D9D9D9", edgecolor="white", linewidth=0.8, hatch=STOP_EPS_HATCH[1.5e0],
              label=r"$\mathring{{\epsilon}}=1.5\mathrm{e}{0}$"),
        Patch(facecolor="#D9D9D9", edgecolor="white", linewidth=1.6, hatch=STOP_EPS_HATCH[1e0],
              label=r"$\mathring{{\epsilon}}=1\mathrm{e}{0}$"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        fontsize=14,
        framealpha=0.92,
        ncol=6,
        handlelength=1.8,
        columnspacing=1.0,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Reward", fontsize=22)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.8)

    fig.subplots_adjust(bottom=0.18, top=0.84)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _pick_best_rows_per_scenario(reward_df: pd.DataFrame) -> pd.DataFrame:
    """Per scenario pick: TSP + best ILQX (best λ) + best PGD (best reward, including stop_eps)."""
    rows: list[pd.Series] = []
    for scenario in SCENARIO_ORDER:
        sub = reward_df.loc[reward_df["scenario"] == scenario]
        if sub.empty:
            continue

        tsp = sub.loc[sub["strategy"] == "TSP"]
        if not tsp.empty:
            rows.append(tsp.iloc[0])

        ilqx = sub.loc[sub["strategy"] == "ILQX"]
        if not ilqx.empty:
            # ILQX best by lambda only: pick the λ with highest total_reward.
            ilqx_best = (
                ilqx.sort_values("total_reward", ascending=False)
                .iloc[0]
            )
            rows.append(ilqx_best)

        pgd = sub.loc[sub["strategy"] == "PGD"]
        if not pgd.empty:
            rows.append(pgd.sort_values("total_reward", ascending=False).iloc[0])

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    scenario_rank = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    strat_rank = {"TSP": 0, "ILQX": 1, "PGD": 2}
    out = (
        out.assign(_scn=out["scenario"].map(scenario_rank), _sr=out["strategy"].map(strat_rank))
        .sort_values(["_scn", "_sr"])
        .drop(columns=["_scn", "_sr"])
        .reset_index(drop=True)
    )
    return out


def _subset_mean_results_for_best(
    mean_results: dict[tuple, pd.DataFrame], best_df: pd.DataFrame
) -> dict[tuple, pd.DataFrame]:
    """Filter mean_results down to only the runs needed for best-only plot."""
    if best_df.empty:
        return {}

    keys = list(mean_results.keys())
    out: dict[tuple, pd.DataFrame] = {}

    for scenario in SCENARIO_ORDER:
        # Always include a sample key for TSP curve (cumulative_original lives in any key)
        sample_key = next((k for k in keys if k[0] == scenario), None)
        if sample_key is not None:
            out[sample_key] = mean_results[sample_key]

        # Best ILQX: choose any key with matching (scenario, lam)
        ilqx_row = best_df.loc[(best_df["scenario"] == scenario) & (best_df["strategy"] == "ILQX")]
        if not ilqx_row.empty:
            lam = float(ilqx_row.iloc[0]["lam"])
            ilqx_key = next((k for k in keys if k[0] == scenario and _approx_equal(k[2], lam)), None)
            if ilqx_key is not None:
                out[ilqx_key] = mean_results[ilqx_key]

        # Best PGD: keep the exact stop_eps selected in best_df.
        pgd_row = best_df.loc[(best_df["scenario"] == scenario) & (best_df["strategy"] == "PGD")]
        if not pgd_row.empty:
            eta = float(pgd_row.iloc[0]["eta"])
            eta_pow = float(pgd_row.iloc[0]["eta_pow"])
            stop_eps = float(pgd_row.iloc[0]["stop_eps"])
            # find all keys with this (eta, eta_pow, stop_eps) for the scenario, regardless of lam
            candidates = [
                k for k in keys
                if k[0] == scenario
                and _approx_equal(k[1], eta)
                and _approx_equal(k[4], eta_pow)
                and _approx_equal(k[3], stop_eps)
            ]
            if candidates:
                # If multiple lambdas exist for the same PGD config, prefer lower final PGD regret.
                def _tie_key(k):
                    df = mean_results[k]
                    end_reg = float(df["cumulative_pgd"].iloc[-1]) if "cumulative_pgd" in df.columns else float("inf")
                    return end_reg
                best_key = min(candidates, key=_tie_key)
                out[best_key] = mean_results[best_key]

    return out


def plot_total_reward_best(
    best_df: pd.DataFrame,
    out_path: Path,
    show: bool = False,
) -> None:
    """3 bars per scenario: TSP + best ILQX + best AMLE-PGD."""
    scenarios = [s for s in SCENARIO_ORDER if s in set(best_df["scenario"])]
    if not scenarios:
        print("No best-only reward rows to plot.")
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    group_gap = 1.6
    bar_x: list[float] = []
    bar_h: list[float] = []
    bar_c: list[str] = []
    tick_text: list[str] = []
    pgd_param_labels: list[tuple[float, float, float, float]] = []
    group_spans: list[tuple[float, float, float, str]] = []
    x = 0.0

    for scenario in scenarios:
        sub = best_df.loc[best_df["scenario"] == scenario]
        if sub.empty:
            continue
        g0 = x

        # enforce order
        for strat in ("TSP", "ILQX", "PGD"):
            hit = sub.loc[sub["strategy"] == strat]
            if hit.empty:
                continue
            row = hit.iloc[0]
            bar_x.append(x)
            bar_h.append(float(row["total_reward"]))
            bar_c.append(str(row["color"]))
            if strat == "TSP":
                tick_text.append("TSP")
            elif strat == "ILQX":
                tick_text.append(rf"$\mathring{{\lambda}}={_fmt_coeff_math(float(row['lam']))}$")
            else:
                tick_text.append("")
                pgd_param_labels.append((
                    x,
                    float(row["eta"]),
                    float(row["eta_pow"]),
                    float(row["stop_eps"]),
                ))
            x += 1.0

        g1 = x - 1.0
        group_spans.append(((g0 + g1) / 2.0, g0, g1, scenario))
        x += group_gap

    ax.bar(bar_x, bar_h, color=bar_c, width=0.78, edgecolor="0.35", linewidth=0.6)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(tick_text, rotation=35, ha="right", fontsize=16)

    trans = ax.get_xaxis_transform()
    for px, eta, a_val, eps in pgd_param_labels:
        ax.text(px, -0.10, rf"$\mathring{{\eta}}={_fmt_coeff_math(eta)}$", transform=trans,
                ha="center", va="top", fontsize=14, color="#C62828", clip_on=False)
        ax.text(px, -0.17, rf"$a={a_val:g}$", transform=trans,
                ha="center", va="top", fontsize=14, color="#C62828", clip_on=False)
        ax.text(px, -0.24, rf"$\mathring{{\epsilon}}={_fmt_coeff_math(eps)}$", transform=trans,
                ha="center", va="top", fontsize=14, color="#C62828", clip_on=False)

    for cx, _x0, _x1, scenario in group_spans:
        ax.text(cx, -0.34, SCENARIO_BETA_PAIR[scenario], transform=trans, ha="center", va="top",
                fontsize=18, clip_on=False)

    ax.set_ylabel("Reward", fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.8)

    fig.subplots_adjust(bottom=0.38)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot styled cumulative regret from sim_output.")
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reward-output", type=Path, default=DEFAULT_REWARD_OUT)
    parser.add_argument(
        "--stop-eps",
        type=float,
        action="append",
        default=None,
        help=(
            "Filter runs by stop_eps. Repeatable, e.g. --stop-eps 5e-4 --stop-eps 1e-3. "
            "If omitted, defaults to PGD_ILQX STOP_EPS_VALUES."
        ),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST_OUT)
    parser.add_argument("--best-reward-output", type=Path, default=DEFAULT_BEST_REWARD_OUT)
    args = parser.parse_args()

    stop_eps_values = DEFAULT_STOP_EPS_VALUES if args.stop_eps is None else args.stop_eps
    eta_values = PGD_BAR_ETA_VALUES
    lam_values = ILQX_BAR_LAM_VALUES
    eta_pow_values = PGD_BAR_A_VALUES

    runs = discover_runs(args.input_root, eta_values, lam_values, eta_pow_values, stop_eps_values)
    if not runs:
        raise SystemExit(
            f"No matching runs under {args.input_root}. "
            f"Expected eta in {eta_values}, lam in {lam_values}, eta_pow in {eta_pow_values}, "
            f"stop_eps in {stop_eps_values}."
        )

    mean_results, reward_df = load_simulation_results(runs)
    print(f"Loaded {len(runs)} CSV(s) → {len(mean_results)} unique config(s).")
    plot_cumulative_regret(mean_results, args.output, show=args.show)
    plot_total_reward(reward_df, args.reward_output, show=args.show)

    # Best-only per scenario (TSP + best ILQX + best AMLE-PGD by total_reward)
    best_df = _pick_best_rows_per_scenario(reward_df)
    if not best_df.empty:
        plot_cumulative_regret_best(mean_results, best_df, args.best_output, show=args.show)
        plot_total_reward_best(best_df, args.best_reward_output, show=args.show)
    else:
        print("No best-only rows found; skipping best-only plots.")


if __name__ == "__main__":
    main()
