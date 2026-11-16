# Regret Computational Efficiency Experiments

This monorepo collects code for studying **online pricing** algorithms under
passenger acceptance uncertainty, with a focus on **cumulative regret** and
**computational cost** (solver time, per-step updates) across estimators.

Two self-contained subprojects share the same methodological family (MOSEK, PGD,
ILQX, SPO+, AdaGrad-style updates) but differ in the data environment:

| Subproject | Folder | What it covers |
| --- | --- | --- |
| **Synthetic benchmarks** | [`Synthetic/R2_rerun/`](Synthetic/R2_rerun/) | Linear demand: Section 5.1 (Table 2, Figure 4) across SNR levels, the time of the exact solver (Figure 2), and the sensitivity to `q` with dense and sparse parameters (Figure EC.1, Appendix B). |
| **NYC taxi (real data)** | [`NYCTaxi/`](NYCTaxi/) | Section 5.2 (Table 4, Figure 5): willingness-to-pay simulator fitted to NYC TLC trips, selection on a November 2014 week, timed tests on four December 2014 weeks, revenue lost to delay under two pricing rules. |

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
│  NYCTaxi/ — willingness-to-pay simulator on NYC TLC taxi trips           │
├──────────────────────────────────────────────────────────────────────────┤
│  build/      →  clean TLC trips 2013-12 to 2014-12, 20 features          │
│  formal.py   →  select (Nov 2014), timed tests (Dec 2014), replay        │
└──────────────────────────────────────────────────────────────────────────┘
```

## Where to start

- **Synthetic experiments, Figure 2, Section 5.1 and Figure EC.1:** [`Synthetic/R2_rerun/README.md`](Synthetic/R2_rerun/README.md)
- **NYC taxi case study, Section 5.2:** [`NYCTaxi/README.md`](NYCTaxi/README.md)

Each subfolder documents its own dependencies (MOSEK, Gurobi/PyEPO where applicable),
data downloads, and run commands. Raw TLC trip records and weather data are **not** redistributed; see
the NYCTaxi README for download links.

## Quick start (pointers)

```bash
# Synthetic experiments (Figure 2, Section 5.1 and Figure EC.1); requirements and run commands in Synthetic/R2_rerun/README.md
(cd Synthetic/R2_rerun && python3 config_r2.py)   # settings, grids and result files of Section 5.1

# NYC taxi case study (Section 5.2); data preparation and run commands in NYCTaxi/README.md
(cd NYCTaxi && python3 formal_report.py)   # Table 4 and Figure 5 from the saved results in formal.csv
```

For full data-prep and retraining workflows, follow the README in the relevant subfolder.
