"""Unpack data/processed/{val,test}.npz (written by build_data.py) into the arrays read by ../../run.py:
data/processed/cache/<split>_t_ns.npy, <split>_fare.npy and <split>_X.npy. Same as backtest.stream in the
original NYC_Major2014Train/rerun_R2 code.

    python make_cache.py
"""
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parent / "data" / "processed"
(PROC / "cache").mkdir(parents=True, exist_ok=True)
for split in ("val", "test"):
    d = np.load(PROC / f"{split}.npz")
    for k in ("t_ns", "fare", "X"):
        np.save(PROC / "cache" / f"{split}_{k}.npy", d[k])
    print(split, "done")
