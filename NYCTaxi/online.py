"""Online algorithm with one update worker (design recorded in protocol.txt).

Every snapshot of the exploration schedule is solved once and timed. The replay then
decides which snapshots the worker actually solves: 'latest' starts the newest waiting
snapshot when the worker becomes free and drops the older ones, 'fifo' solves all in
order. A load factor only compresses the arrival clock. Rules for a request:
    newest   priced with the newest finished estimate (baseline);
    fare     recorded fare while the worker is computing, newest estimate otherwise;
    explore  a random exploration price while the worker is computing (its response is
             not added to the data), newest estimate otherwise;
    lost     no revenue while the worker is computing, newest estimate otherwise.

    python online.py learn SEED [MK]      solve and time every snapshot
    python online.py live SEED [MK]       APGD with true warm starts where snapshots are dropped
    python online.py evaluate             replay all saved runs, write online.csv
"""
import run  # sets the single-thread environment before numpy is imported
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit
import solvers
import single_update as su
from replay import schedule, index_AB, best_price, prob, _policy_prices

OUT = run.RESULTS / 'online'
T = 100_000
M0 = 410
LOADS = (1, 7, 28, 112, 448, 1792, 7168, 28672)
WINDOWS = dict(first5=.05, first20=.2, all=1.)
METHODS = ([dict(family='apgd', reg=.01, eps0=e) for e in (.003, .001, 3e-4)]
           + [dict(family='mosek', reg=.01, eps0=0.)]
           + [dict(family='ilqx', reg=.03, eps0=0.)])


def tag(cfg):
    return f"{cfg['family']}_reg{cfg['reg']:g}_eps{cfg['eps0']:g}"


def stream(seed, mk):
    z, clock, fare, _ = run.load_stream('val', 1, T)
    true = np.load(run.DATA / f'{su.SCENARIO}.npy')
    idx, trig = schedule(T, M0, mk)
    rng = np.random.default_rng([20261113, seed])
    price = rng.choice([10., 15., 20., 25., 30.], len(idx))
    phi = run.design(z[idx], price)
    y = (rng.random(len(idx)) < expit(phi @ true)).astype(float)
    return dict(z=z, clock=clock, fare=fare, true=true, idx=idx, trig=trig, price=price, phi=phi, y=y,
                n_at=np.searchsorted(idx, trig - 1, side='right'))


def learn(seed, mk):
    s = stream(seed, mk)
    folder = OUT / f'mk{mk}' / f'seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    for cfg in METHODS:
        out = folder / (tag(cfg) + '.npz')
        if out.exists():
            continue
        warm = None; thetas = []; secs = []
        for n, t in zip(s['n_at'], s['trig']):
            t0 = time.perf_counter()
            th, info = su.solve(cfg, s['phi'][:n], s['y'][:n], warm, int(t))
            secs.append(time.perf_counter() - t0); thetas.append(th)
            if not info['ok']:
                raise RuntimeError(f'{tag(cfg)} failed at n={n}')
            warm = (th, info['L']) if cfg['family'] == 'apgd' else None
        np.savez(out, thetas=np.array(thetas), secs=np.array(secs), cfg=json.dumps(cfg), load=os.getloadavg()[0])
        print(seed, mk, tag(cfg), 'total', round(sum(secs), 1), 's, last', round(secs[-1], 3), flush=True)


def live(seed, mk):
    """APGD under the latest policy with the true warm starts. Needed only for loads at
    which the worker drops snapshots: there the next update starts from an older estimate
    and takes longer than the time measured in learn()."""
    s = stream(seed, mk); J = len(s['trig'])
    folder = OUT / f'mk{mk}' / f'seed{seed}'; (folder / 'live').mkdir(exist_ok=True)
    for cfg in METHODS:
        if cfg['family'] != 'apgd':
            continue
        secs = np.load(folder / (tag(cfg) + '.npz'))['secs']
        for load in LOADS:
            out = folder / 'live' / f'{tag(cfg)}_load{load}.npz'
            trig_time = s['clock'][s['trig'] - 1] / load
            if out.exists() or len(timeline(trig_time, secs, 'latest')[1]) == J:
                continue
            start = []; finish = []; solved = []; thetas = []; warm = None; free = -np.inf; nxt = 0
            while nxt < J:
                begin = max(free, trig_time[nxt])
                j = int(np.searchsorted(trig_time, begin, side='right')) - 1
                n = s['n_at'][j]; t0 = time.perf_counter()
                th, info = su.solve(cfg, s['phi'][:n], s['y'][:n], warm)
                free = begin + time.perf_counter() - t0
                start.append(begin); finish.append(free); solved.append(j); thetas.append(th)
                warm = (th, info['L']); nxt = j + 1
            np.savez(out, start=start, finish=finish, solved=solved, thetas=np.array(thetas), load=os.getloadavg()[0])
            print(seed, mk, tag(cfg), 'load', load, 'solved', len(solved), 'of', J, flush=True)


def timeline(trig_time, secs, policy):
    """Finish time and snapshot index of the updates the single worker solves."""
    finish = []; solved = []; free = -np.inf; nxt = 0
    while nxt < len(secs):
        start = max(free, trig_time[nxt])
        j = nxt if policy == 'fifo' else int(np.searchsorted(trig_time, start, side='right')) - 1
        free = start + secs[j]; finish.append(free); solved.append(j); nxt = j + 1
    return np.array(finish), np.array(solved)


def evaluate():
    rows = []
    for file in sorted(OUT.glob('mk*/seed*/*.npz')):
        mk = int(file.parts[-3][2:]); seed = int(file.parts[-2][4:])
        s = stream(seed, mk); saved = np.load(file); cfg = json.loads(str(saved['cfg']))
        thetas, secs = saved['thetas'], saved['secs']
        A, B = index_AB(s['z'], s['true'])
        earn = lambda idx, price: price * prob(A[idx], B[idx], price)
        explore = np.zeros(T, bool); explore[s['idx']] = True; ex = np.flatnonzero(~explore)
        oracle = earn(np.arange(T), best_price(A, B)); tsp = earn(np.arange(T), s['fare'])
        target = np.searchsorted(s['trig'], ex + 1, side='right') - 1
        common = np.zeros(T); common[s['idx']] = earn(s['idx'], s['price'])

        def revenue(est, thetas=thetas):
            r = common.copy(); r[ex] = earn(ex, _policy_prices(s['z'][ex], thetas, est, 10., 2.5, 70.))
            return r
        instant = revenue(target)
        # an update is due when the first exploitation request after its trigger arrives
        first = np.searchsorted(ex, s['trig']); due = first < len(ex)
        first = ex[np.minimum(first, len(ex) - 1)]
        due &= first < np.r_[s['trig'][1:] - 1, T]
        wander = earn(np.arange(T), np.random.default_rng([7, seed]).choice([10., 15., 20., 25., 30.], T))
        for policy in ('latest', 'fifo'):
            for load in LOADS:
                clock = s['clock'] / load
                finish, solved = timeline(clock[s['trig'] - 1], secs, policy)
                begin = finish - secs[solved]; used_thetas = thetas
                real = file.parent / 'live' / f'{file.stem}_load{load}.npz'
                if policy == 'latest' and real.exists():      # measured with the true warm starts
                    r = np.load(real); begin, finish, solved = r['start'], r['finish'], r['solved']
                    used_thetas = np.zeros_like(thetas); used_thetas[solved] = r['thetas']
                pos = np.searchsorted(finish, clock[ex], side='right') - 1
                used = np.where(pos >= 0, solved[np.maximum(pos, 0)], -1)
                deployed = revenue(used, used_thetas)
                busy = np.zeros(T, bool)        # worker computing when the request arrives
                busy[ex] = np.searchsorted(begin, clock[ex], side='right') - 1 > pos
                rule = dict(newest=deployed, fare=np.where(busy, tsp, deployed),
                            explore=np.where(busy, wander, deployed), lost=np.where(busy, 0., deployed))
                done = np.full(len(secs), np.inf); done[solved] = finish
                timely = done <= clock[first]
                behind = s['n_at'][np.maximum(target, 0)] - np.where(used >= 0, s['n_at'][np.maximum(used, 0)], 0)
                for window, share in WINDOWS.items():
                    h = int(share * T); m = ex < h
                    rows.append(dict(cfg, mk=mk, seed=seed, policy=policy, load=load, window=window,
                                     rate=(h - 1) / clock[h - 1], oracle=oracle[:h].sum(), tsp=tsp[:h].sum(),
                                     instant=instant[:h].sum(), deployed=deployed[:h].sum(),
                                     **{k: v[:h].sum() for k, v in rule.items() if k != 'newest'},
                                     busy=busy[:h][~explore[:h]].mean(),
                                     timely=timely[due & (s['trig'] <= h)].mean(),
                                     stale=np.mean((used != target)[m]), fallback=np.mean((used < 0)[m]),
                                     behind=np.mean(behind[m] / s['n_at'][np.maximum(target[m], 0)]),
                                     solved=len(solved) / len(secs), seconds=secs.sum(),
                                     machine_load=float(saved['load'])))
    frame = pd.DataFrame(rows); frame.to_csv(run.ROOT / 'online.csv', index=False)
    return frame


if __name__ == '__main__':
    if sys.argv[1] in ('learn', 'live'):
        solvers.prox_half_sq_pnorm(np.ones(su.KAPPA), 1., su.Q)   # compile outside measurements
        (learn if sys.argv[1] == 'learn' else live)(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 28)
    else:
        print(len(evaluate()), 'rows')
