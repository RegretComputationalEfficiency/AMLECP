"""Single-update mechanism experiment (design recorded in protocol.txt).

An old estimate, learned from n_old exploration observations, is already deployed.
A block of delta new exploration observations ends at the trigger. The update on the
n_old + delta observations replaces the old estimate when it finishes. The requests
that follow are priced twice: with the new estimate from the trigger on (instant), and
with the old estimate until the measured finish time (deployed). A load factor only
compresses the arrival clock. Request order, features, exploration prices and
responses do not change with it.

    python single_update.py learn [first_rep last_rep]   solve and time every update
    python single_update.py evaluate                     replay, write single_update.csv
"""
import run  # sets the single-thread environment before numpy is imported
import functools
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit
import solvers
from replay import schedule, index_AB, best_price, prob

OUT = run.RESULTS / 'single_update'
SCENARIO = 'wtp16'
REPS = 20
TAU = np.linspace(100_000, 3_400_000, REPS).round().astype(int)  # first evaluated request
N_OLD = (410, 1640, 6560)
DELTA = (128, 512)
LOADS = (1, 7, 28, 112, 448)   # 112 and 448 are stress levels beyond any NYC load
HISTORY = 20_000        # requests before the new block; old observations are drawn from them
HORIZON = 10_000        # requests evaluated after the trigger
WINDOWS = (1_000, 10_000)
ETA0 = (.003, .01, .03)
EPS0 = dict(pgd=(.1, .03, .01, .003, .001, 3e-4, 1e-4),             # plain PGD (Algorithm 2)
            apgd=(.1, .03, .01, .003, .001, 3e-4, 1e-4, 1e-5, 1e-6))  # accelerated, to high precision
LAM0 = (.03, .1, .3)
REPEATS = 3             # timings of an update shorter than 0.3 s; the median is used
KAPPA = 41
Q = 4.


def configs():
    out = [dict(family=s, reg=e, eps0=p) for s in ('pgd', 'apgd') for e in ETA0 for p in EPS0[s]]
    out += [dict(family='mosek', reg=e, eps0=0.) for e in ETA0]
    return out + [dict(family='ilqx', reg=l, eps0=0.) for l in LAM0]


def own_window(n_new, delta):
    """Exploitation requests per block of delta when exploration keeps the density of the
    k^2 schedule with 28 observations per epoch (392/n per request)."""
    return int(round(delta * (n_new / 392 - 1)))


@functools.lru_cache(None)
def schedule_time(n):
    """Request number at which the k^2 schedule has collected n exploration observations."""
    idx, trig = schedule(400_000)
    n_at = np.searchsorted(idx, trig - 1, side='right')
    return int(trig[np.searchsorted(n_at, n)])


def load(rep):
    tau = int(TAU[rep]); lo = tau - max(DELTA) - HISTORY; hi = tau + HORIZON
    src = run.SOURCE / 'data/processed/cache'
    t = np.asarray(np.load(src / 'val_t_ns.npy', mmap_mode='r')[lo:hi])
    x = np.array(np.load(src / 'val_X.npy', mmap_mode='r')[lo:hi], dtype=float)
    x[:, 16] = run.daylight(t)
    fare = np.array(np.load(src / 'val_fare.npy', mmap_mode='r')[lo:hi], dtype=float)
    return x / np.load(run.DATA / 'scale.npy'), t / 1e9, fare


def draw(rep):
    """Exploration data and evaluation requests of one repetition."""
    z, clock, fare = load(rep)
    true = np.load(run.DATA / f'{SCENARIO}.npy')
    rng = np.random.default_rng([20261113, rep])
    start = HISTORY + max(DELTA)                      # first evaluated request
    pos = np.r_[rng.permutation(HISTORY)[:max(N_OLD)], np.arange(HISTORY, start)]
    price = rng.choice([10., 15., 20., 25., 30.], len(pos))
    phi = run.design(z[pos], price)
    y = (rng.random(len(pos)) < expit(phi @ true)).astype(float)
    return dict(phi=phi, y=y, true=true, z=z[start:], fare=fare[start:],
                wait=clock[start:] - clock[start - 1], trigger_time=clock[start - 1])


def sample(data, n_old, delta=0):
    """First n_old history observations, then the last delta of the new block."""
    m = max(N_OLD) + max(DELTA)
    rows = np.r_[np.arange(n_old), np.arange(m - delta, m)]
    return data['phi'][rows], data['y'][rows]


def objective(cfg, phi, y, th):
    u = phi @ th; loss = float(np.sum(np.logaddexp(0, u) - y * u)); n = len(y)
    if cfg['family'] == 'ilqx':
        return loss + lam_of(cfg, n) * np.sum(np.abs(th))
    p = Q / (Q - 1)
    return loss + .5 * eta_of(cfg, n) * np.sum(np.abs(th) ** p) ** (2 / p)


def eta_of(cfg, n):
    return cfg['reg'] * KAPPA ** (1 / Q - .5) * np.sqrt(n)   # eta_k = eta0 kappa^(1/q-1/2) sqrt(n)


def lam_of(cfg, n, t=None):
    t = t or schedule_time(n)
    return cfg['reg'] * t ** .25 * np.sqrt(np.log(KAPPA) + np.log(t))


def solve(cfg, phi, y, warm=None, t=None):
    """One update. warm = (theta, L) of the previous update of the same method.
    t = request number of the trigger (ILQX only; default: the k^2 schedule's time for n)."""
    n = len(y)
    if cfg['family'] in ('pgd', 'apgd'):
        L_max = float(np.sum(phi ** 2)) / 4
        th0, L0 = warm if warm else (np.zeros(KAPPA), L_max)
        fun = solvers.amle_pgd if cfg['family'] == 'pgd' else solvers.amle_apgd
        eps = cfg['eps0'] * np.sqrt(KAPPA * n)
        th, info = fun(phi, y, eta_of(cfg, n), eps, th0, L_max, L0, Q)
        return th, dict(L=info['L'], iters=info['iters'], ok=info['G'] <= eps * (1 + 1e-8))
    if cfg['family'] == 'mosek':
        th, info = solvers.amle_mosek(phi, y, eta_of(cfg, n), Q)
        if th is None:
            raise RuntimeError('MOSEK returned no AMLE solution')
        return th, dict(L=0., iters=info['iters'], ok=bool(info['optimal']))
    lam = lam_of(cfg, n, t)
    th, info = solvers.lasso_mosek(phi, y, lam)
    if th is None:
        raise RuntimeError('MOSEK returned no lasso solution')
    gap, obj = run.lasso_gap(phi, y, th, lam)
    if gap > 1e-4 * max(1., obj):
        th = run.polish_lasso(phi, y, th, lam)
        gap, obj = run.lasso_gap(phi, y, th, lam)
    return th, dict(L=0., iters=info['iters'], ok=gap <= 1e-4 * max(1., obj))


def learn(rep):
    out = OUT / f'rep{rep:02d}.npz'
    if out.exists():
        return
    data = draw(rep); rows = []; olds = []; news = []; secs = []
    for n_old in N_OLD:
        po, yo = sample(data, n_old)
        for cfg in configs():
            th_old, old = solve(cfg, po, yo)
            for delta in DELTA:
                pn, yn = sample(data, n_old, delta)
                times = []
                while len(times) < REPEATS and (not times or times[0] < .3):
                    t0 = time.perf_counter()
                    th_new, new = solve(cfg, pn, yn, (th_old, old['L']))
                    times.append(time.perf_counter() - t0)
                rows.append(dict(cfg, rep=rep, n_old=n_old, delta=delta, iters_old=int(old['iters']),
                                 iters_new=int(new['iters']), ok=bool(old['ok'] and new['ok']),
                                 objective_old=objective(cfg, po, yo, th_old),
                                 objective_new=objective(cfg, pn, yn, th_new)))
                olds.append(th_old); news.append(th_new); secs.append(float(np.median(times)))
        print('rep', rep, 'n_old', n_old, 'done', flush=True)
    np.savez(out, rows=json.dumps(rows), theta_old=np.array(olds), theta_new=np.array(news),
             secs=np.array(secs), load=os.getloadavg()[0])


def evaluate():
    """Per update: revenue of old/new estimate and deployed revenue for every load and window."""
    table = []
    for rep in range(REPS):
        file = OUT / f'rep{rep:02d}.npz'
        if not file.exists():
            continue
        data = draw(rep); saved = np.load(file)
        A, B = index_AB(data['z'], data['true'])
        earn = lambda price: price * prob(A, B, price)
        oracle = earn(best_price(A, B)); tsp = earn(data['fare'])
        rate = (HORIZON - 1) / (data['wait'][-1] - data['wait'][0])
        for row, th_old, th_new, secs in zip(json.loads(str(saved['rows'])), saved['theta_old'],
                                             saved['theta_new'], saved['secs']):
            r_old = earn(best_price(*index_AB(data['z'], th_old)))
            r_new = earn(best_price(*index_AB(data['z'], th_new)))
            sec = float(secs)
            windows = dict(own=own_window(row['n_old'] + row['delta'], row['delta']))
            windows.update({str(h): h for h in WINDOWS})
            for name, h in windows.items():
                base = dict(row, window=name, requests=h, seconds=sec, arrival_rate=rate,
                            oracle=oracle[:h].sum(), tsp=tsp[:h].sum(),
                            old=r_old[:h].sum(), instant=r_new[:h].sum())
                for load in LOADS:
                    stale = data['wait'][:h] / load < sec
                    table.append(dict(base, load=load, stale=int(stale.sum()),
                                      deployed=np.where(stale, r_old[:h], r_new[:h]).sum()))
    frame = pd.DataFrame(table)
    frame.to_csv(run.ROOT / 'single_update.csv', index=False)
    return frame


if __name__ == '__main__':
    if sys.argv[1] == 'learn':
        OUT.mkdir(parents=True, exist_ok=True)
        solvers.prox_half_sq_pnorm(np.ones(KAPPA), 1., Q)   # compile outside measurements
        first, last = (int(v) for v in sys.argv[2:4]) if len(sys.argv) > 2 else (0, REPS - 1)
        for rep in range(first, last + 1):
            learn(rep)
    else:
        print(len(evaluate()), 'rows')
