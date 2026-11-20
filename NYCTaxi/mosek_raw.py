"""AMLE-MOSEK test chains rerun with MOSEK's own solution kept when it is not certified optimal (2026-11-18).

formal.py redid every MOSEK solve whose status was not Optimal by a cold-started APGD solve at eps0 = 1e-6, and
counted that time. Here the solution MOSEK returns (status Feasible or Unknown) is used as it is, and only
MOSEK's own time (model build and solve) is counted. A solve that returns no solution at all is still redone by
APGD and flagged. Per update the run saves the time, MOSEK's status and the repair flag.

The previous runs are moved to results/formal_mosek_repaired/ (same relative paths), then the new ones are written
where formal.evaluate reads them.
    python mosek_raw.py
"""
import run  # noqa: F401  single-thread environment before numpy
import json
import os
import time
import numpy as np
import formal as F
import solvers
import single_update as su
from online import tag

SCENARIO = 'wtp16'
CFG = dict(family='mosek', reg=json.loads((F.OUT / SCENARIO / 'selected.json').read_text())['eta'], eps0=0.)  # selected eta0
OLD = F.OUT.parent / 'formal_mosek_repaired'


def chain(s, cfg):
    thetas, secs, optimal, repaired = [], [], [], []; peak = os.getloadavg()[0]
    for j, (n, t) in enumerate(zip(s['n_at'], s['trig'])):
        phi, y = s['phi'][:n], s['y'][:n]; eta = su.eta_of(cfg, n)
        t0 = time.perf_counter()
        th, info = solvers.amle_mosek(phi, y, eta, su.Q)
        fixed = th is None
        if fixed:                                   # no usable solution at all: redo as formal.update does
            th, _ = su.solve(dict(family='apgd', reg=cfg['reg'], eps0=1e-6), phi, y)
        secs.append(time.perf_counter() - t0); thetas.append(th)
        optimal.append(bool(info['optimal'])); repaired.append(fixed)
        if j % 20 == 0:
            peak = max(peak, os.getloadavg()[0])
    return (np.array(thetas), np.array(secs), np.array(optimal), np.array(repaired),
            max(peak, os.getloadavg()[0]))


def main():
    solvers.warm_up(su.KAPPA)
    for split, day in F.TEST:
        for seed in F.SEEDS:
            folder = F.OUT / SCENARIO / day / f'seed{seed}'; out = folder / (tag(CFG) + '.npz')
            done = out.exists() and 'optimal' in np.load(out).files
            if done:
                continue
            if out.exists():                        # keep the repaired run
                dest = OLD / out.relative_to(F.OUT); dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    out.rename(dest)
                else:                               # never delete: keep a second copy beside the first
                    out.rename(dest.with_name(dest.stem + f'_{int(time.time())}.npz'))
            F.wait_idle()
            s = F.stream(SCENARIO, split, day, seed, F.plan(CFG))
            thetas, secs, optimal, repaired, peak = chain(s, CFG)
            np.savez(out, thetas=thetas, secs=secs, cfg=json.dumps(CFG), repaired=int(repaired.sum()),
                     optimal=optimal, repaired_mask=repaired, load=peak)
            F.log(SCENARIO, day, seed, 'mosek raw total', round(secs.sum(), 1), 's, not optimal',
                  int((~optimal).sum()), 'no solution', int(repaired.sum()), 'peak load', round(peak, 1))


if __name__ == '__main__':
    main()
