# Regret Computational Efficiency Experiments

This monorepo collects code for studying **online pricing** algorithms under
passenger acceptance uncertainty, with a focus on **cumulative regret** and
**computational cost** (solver time, per-step updates) across estimators.

Two self-contained subprojects share the same methodological family (MOSEK, PGD,
ILQX, SPO+, AdaGrad-style updates) but differ in the data environment:

| Subproject | Folder | What it covers |
| --- | --- | --- |
| **Synthetic benchmarks** | [`Synthetic/R2_rerun/`](Synthetic/R2_rerun/) | Linear demand: Section 5.1 (Table 2, Figure 4) across SNR levels, the time of the exact solver (Figure 2), the sparse experiment (Table 3), and the sensitivity to `q` (Figure EC.1). |
| **NYC ride-hailing (real data)** | [`NYCTaxi/`](NYCTaxi/) | Section 5.2 (Table 4, Figure 5): willingness-to-pay simulator fitted to NYC TLC ride-hailing (HVFHV) trips, policies estimating 729 parameters, selection on Monday 2026-09-07, timed tests on three September 2026 Mondays, revenue lost to delay under Algorithm 1. |

The code of the first-round version (before the second-round revision) is kept at the tag
[`round1`](https://github.com/RegretComputationalEfficiency/AMLECP/tree/round1).

## Experiment overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Synthetic/ — controlled feature geometry (Gaussian, p-norm budget q)       │
├──────────────────────────────────────────────────────────────────────────┤
│  R2_rerun/       →  revised paper: Figure 2, Section 5.1, Figure EC.1    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  NYCTaxi/ — willingness-to-pay simulator on NYC TLC ride-hailing trips   │
├──────────────────────────────────────────────────────────────────────────┤
│  build/      →  clean HVFHV trips 2025-10 to 2026-09, 20 features        │
│  formal.py   →  select (Sep 7), timed tests (Sep 14, 21, 28), replay     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Where to start

- **Synthetic experiments, Figure 2, Section 5.1 and Figure EC.1:** [`Synthetic/R2_rerun/README.md`](Synthetic/R2_rerun/README.md)
- **NYC ride-hailing case study, Section 5.2:** [`NYCTaxi/README.md`](NYCTaxi/README.md)

Each subfolder documents its own dependencies (MOSEK, Gurobi/PyEPO where applicable),
data downloads, and run commands. Raw TLC trip records and weather data are **not** redistributed; see
the NYCTaxi README for download links.

## Quick start (pointers)

```bash
# Synthetic experiments (Figure 2, Section 5.1 and Figure EC.1); requirements and run commands in Synthetic/R2_rerun/README.md
(cd Synthetic/R2_rerun && python3 config_r2.py)   # settings, grids and result files of Section 5.1

# NYC ride-hailing case study (Section 5.2); data preparation and run commands in NYCTaxi/README.md
(cd NYCTaxi && python3 formal_report.py && python3 figure_paper.py figures)   # Table 4 and Figure 5 from formal.csv
```

For full data-prep and retraining workflows, follow the README in the relevant subfolder.
