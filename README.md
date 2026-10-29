# Regret Computational Efficiency Experiments

This monorepo collects code for studying **online pricing** algorithms under
passenger acceptance uncertainty, with a focus on **cumulative regret** and
**computational cost** (solver time, per-step updates) across estimators.

Two self-contained subprojects share the same methodological family (MOSEK, PGD,
ILQX, SPO+, AdaGrad-style updates) but differ in the data environment:

| Subproject | Folder | What it covers |
| --- | --- | --- |
| **Synthetic benchmarks** | [`Synthetic/`](Synthetic/) | Gaussian-feature toy online pricing; SNR sweeps and `q`-sensitivity studies on simulated arrivals. |
| **NYC taxi (real data)** | [`NYCTaxi/`](NYCTaxi/) | Kaggle-trained acceptance simulator + December 2014 NYC TLC out-of-sample online pricing tests. |

For the revised paper (second round), Section 5.1 (Table 2, Figure 4) and Figure EC.1 are produced by
[`Synthetic/R2_rerun/`](Synthetic/R2_rerun/); `Synthetic/Main_Gaussian/` and `Synthetic/q_sensitivity/` hold the
first-round code.

## Experiment overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Synthetic/ — controlled feature geometry (Gaussian, p-norm budget q)       │
├──────────────────────────────────────────────────────────────────────────┤
│  Main_Gaussian/  →  online methods  →  regret & solver-time CSVs         │
│  q_sensitivity/  →  vary q, compare estimators on same synthetic DGP     │
│  R2_rerun/       →  revised paper: Section 5.1 and Figure EC.1           │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  NYCTaxi/ — learned environment on NYC taxi fares                         │
├──────────────────────────────────────────────────────────────────────────┤
│  Simulator/   →  train β on Kaggle features (2013–2014)                  │
│  2014_data/   →  December 2014 TLC test  →  PGD_ILQX + regret plots      │
└──────────────────────────────────────────────────────────────────────────┘
```

## Where to start

- **Revised paper, Section 5.1 and Figure EC.1:** [`Synthetic/R2_rerun/README.md`](Synthetic/R2_rerun/README.md)
- **Synthetic Gaussian benchmark:** [`Synthetic/Main_Gaussian/README.md`](Synthetic/Main_Gaussian/README.md)
- **Synthetic `q` sensitivity:** [`Synthetic/q_sensitivity/README.md`](Synthetic/q_sensitivity/README.md) (if present)
- **NYC taxi pipeline (train + December test):** [`NYCTaxi/README.md`](NYCTaxi/README.md)

Each subfolder documents its own dependencies (MOSEK, Gurobi/PyEPO where applicable),
data downloads, and run commands. Raw TLC/Kaggle data are **not** redistributed; see
the NYCTaxi README for download links.

## Quick start (pointers)

```bash
# Synthetic Gaussian experiment
pip install -r Synthetic/Main_Gaussian/requirements.txt
# See Synthetic/Main_Gaussian/README.md for run instructions

# NYC taxi December test (uses pre-trained simulator artifacts when available)
pip install -r NYCTaxi/Simulator/requirements.txt
pip install -r NYCTaxi/2014_data/requirements.txt
export TEST_DATA_PATH=NYCTaxi/2014_data/data/2014_december_test_arrivals.csv
python NYCTaxi/2014_data/PGD_ILQX.py
```

For full data-prep and retraining workflows, follow the README in the relevant subfolder.
