"""Formal Case Study II experiment (design recorded in protocol.txt).

Hyperparameters are selected on the validation week by revenue with instantaneous
updates. The selected policies are then run and timed on four December weeks, and the
replay of online.py (one worker, newest waiting snapshot, newest finished estimate)
gives timeliness and the revenue lost to delay at every request rate.

    python formal.py select SCENARIO    validation week, several processes, times unused
    python formal.py test SCENARIO      December weeks, one run after another, timed
    python formal.py evaluate           replay every saved test run, write formal.csv
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
import single_update as su
from online import timeline, tag
from replay import schedule, index_AB, best_price, prob, _policy_prices

SMOKE = bool(os.environ.get('FORMAL_SMOKE'))            # small end-to-end check of the pipeline
T = 20_000 if SMOKE else 200_000
M0, MK = 410, 28
VAL = ('val', '2014-11-17')
TEST = [('test', d) for d in ('2014-12-01', '2014-12-08', '2014-12-15', '2014-12-22')][:1 if SMOKE else 4]
SEEDS = (1,) if SMOKE else (1, 2, 3)                              # test runs, and ILQX selection
SELECT_SEEDS = (1,) if SMOKE else tuple(range(1, 11))             # APGD selection on the validation week
RATES = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)   # requests per second, after 'native'
WINDOWS = dict(first5=.05, first20=.2, all=1.)
LADDER = [m * 10. ** e for e in range(-6, 2) for m in (1, 3)]       # 1-3-10 grid for extensions
GRID = dict(eta=[.001, .002, .003, .005, .01], eps=[3e-4, .001, .002, .003, .005, .01], lam=[.003, .01, .03, .1])
IDLE = 5.                                                            # one-minute load allowed for timed runs
OUT = run.RESULTS / ('formal_smoke' if SMOKE else 'formal')


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


@functools.lru_cache(maxsize=2)
def stream(scenario, split, day, seed):
    src = run.SOURCE / 'data/processed/cache'
    t = np.load(src / f'{split}_t_ns.npy', mmap_mode='r')
    lo = int(np.searchsorted(t, np.datetime64(day, 'ns').astype('int64')))
    tt = np.asarray(t[lo:lo + T]); assert len(tt) == T
    x = np.array(np.load(src / f'{split}_X.npy', mmap_mode='r')[lo:lo + T], dtype=float)
    x[:, 16] = run.daylight(tt)
    z = x / np.load(run.DATA / 'scale.npy')
    fare = np.array(np.load(src / f'{split}_fare.npy', mmap_mode='r')[lo:lo + T], dtype=float)
    true = np.load(run.DATA / f'{scenario}.npy')
    idx, trig = schedule(T, M0, MK)
    rng = np.random.default_rng([20261114, int(day.replace('-', '')), seed])
    price = rng.choice([10., 15., 20., 25., 30.], len(idx))
    phi = run.design(z[idx], price)
    y = (rng.random(len(idx)) < expit(phi @ true)).astype(float)
    s = dict(T=T, z=z, clock=(tt - tt[0]) / 1e9, fare=fare, true=true, idx=idx, trig=trig, price=price, phi=phi,
             y=y, n_at=np.searchsorted(idx, trig - 1, side='right'))
    # quantities of the replay that do not depend on the policy
    A, B = index_AB(z, true); every = np.arange(T)
    earn = lambda i, p: p * prob(A[i], B[i], p)
    explore = np.zeros(T, bool); explore[idx] = True; ex = np.flatnonzero(~explore)
    common = np.zeros(T); common[idx] = earn(idx, price)
    first = np.searchsorted(ex, trig); due = first < len(ex)      # an update is due at the first
    first = ex[np.minimum(first, len(ex) - 1)]                     # exploitation request after its trigger
    due &= first < np.r_[trig[1:] - 1, T]
    s.update(earn=earn, explore=explore, ex=ex, common=common, first=first, due=due,
             oracle=earn(every, best_price(A, B)), tsp=earn(every, fare),
             target=np.searchsorted(trig, ex + 1, side='right') - 1,
             wander=earn(every, np.random.default_rng([7, seed]).choice([10., 15., 20., 25., 30.], T)))
    return s


def levels(s):
    """Request-rate levels: the recorded clock, then the clock divided so that the mean rate is RATES."""
    native = (s['T'] - 1) / s['clock'][-1]
    return [('native', 1.)] + [(str(r), r / native) for r in RATES]


def update(cfg, phi, y, warm, t):
    """One update. A MOSEK solve of the AMLE problem that is not certified optimal is redone by the
    accelerated proximal gradient method at high precision (repaired). Returns theta, L, repaired."""
    try:
        th, info = su.solve(cfg, phi, y, warm, t)
    except RuntimeError:
        th, info = None, dict(ok=False, L=0.)
    if info['ok']:
        return th, info['L'], 0
    if cfg['family'] == 'mosek':
        th, info = su.solve(dict(family='apgd', reg=cfg['reg'], eps0=1e-6), phi, y)
    if th is None or (cfg['family'] == 'apgd' and not info['ok']):
        raise RuntimeError(f'{tag(cfg)} gave no usable solution at n={len(y)}')
    return th, info['L'], 1


def chain(s, cfg):
    """Solve every snapshot in order and time it. APGD is warm-started from the previous snapshot."""
    warm = None; thetas = []; secs = []; repaired = 0; peak = os.getloadavg()[0]
    for j, (n, t) in enumerate(zip(s['n_at'], s['trig'])):
        t0 = time.perf_counter()
        th, L, fixed = update(cfg, s['phi'][:n], s['y'][:n], warm, int(t))
        secs.append(time.perf_counter() - t0); thetas.append(th); repaired += fixed
        warm = (th, L) if cfg['family'] == 'apgd' else None
        if j % 20 == 0:
            peak = max(peak, os.getloadavg()[0])                 # machine load seen during the run
    return np.array(thetas), np.array(secs), repaired, max(peak, os.getloadavg()[0])


def revenue(s, thetas, est):
    r = s['common'].copy()
    r[s['ex']] = s['earn'](s['ex'], _policy_prices(s['z'][s['ex']], thetas, est, 10., 2.5, 70.))
    return r


# ------------------------------------------------------------------ selection on the validation week
def score(task):
    scenario, cfg, seed = task
    s = stream(scenario, *VAL, seed)
    thetas = chain(s, cfg)[0]
    return dict(cfg, seed=seed, share=revenue(s, thetas, s['target']).sum() / s['oracle'].sum())


def beyond(values, value):
    """Next ladder value outside the tried range when value sits on its edge, else None."""
    i = int(np.argmin(np.abs(np.log(LADDER) - np.log(value))))
    if np.isclose(value, min(values)) and i > 0:
        return LADDER[i - 1]
    if np.isclose(value, max(values)) and i + 1 < len(LADDER):
        return LADDER[i + 1]


def r6(v):
    return float(f'{float(v):.6g}')


def decide(table):
    """Two steps on the APGD runs with all SELECT_SEEDS. (1) eta0 with the smallest mean regret at the
    tightest tolerance of the grid. (2) At that eta0, the largest eps0 whose mean regret is within 1% of
    the smallest regret on its path, the rule of Section 5.1. lam0 by mean regret over SEEDS."""
    t = table.assign(reg=table.reg.map(r6), eps0=table.eps0.map(r6))
    a = t[(t.family == 'apgd') & t.seed.isin(SELECT_SEEDS)].groupby(['reg', 'eps0']).share
    regret = (1 - a.mean())[a.size() == len(SELECT_SEEDS)]
    first = regret.xs(r6(min(GRID['eps'])), level='eps0')
    eta = first.idxmin(); path = regret.xs(eta, level='reg')
    l = t[(t.family == 'ilqx') & t.seed.isin(SEEDS)].groupby('reg').share
    lam = (1 - l.mean())[l.size() == len(SEEDS)]
    return dict(eta=float(eta), eps=float(path[path <= 1.01 * path.min()].index.max()), lam=float(lam.idxmin()),
                etas=sorted(float(v) for v in first.index), epss=sorted(float(v) for v in path.index),
                lams=sorted(float(v) for v in lam.index))


def select(scenario, workers=4):
    import multiprocessing as mp
    folder = OUT / scenario; folder.mkdir(parents=True, exist_ok=True); file = folder / 'selection.csv'
    table = pd.read_csv(file) if file.exists() else pd.DataFrame(columns=['family', 'reg', 'eps0', 'seed', 'share'])
    todo = ([dict(family='apgd', reg=e, eps0=p) for e in GRID['eta'] for p in GRID['eps']]
            + [dict(family='ilqx', reg=l, eps0=0.) for l in GRID['lam']])
    for _ in range(6):
        done = {(r.family, r6(r.reg), r6(r.eps0), r.seed) for r in table.itertuples()}
        tasks = [(scenario, dict(c, reg=r6(c['reg']), eps0=r6(c['eps0'])), sd) for c in todo
                 for sd in (SELECT_SEEDS if c['family'] == 'apgd' else SEEDS)
                 if (c['family'], r6(c['reg']), r6(c['eps0']), sd) not in done]
        if tasks:
            log('select', scenario, len(tasks), 'runs')
            with mp.Pool(workers) as pool:
                for row in pool.imap_unordered(score, tasks):
                    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True); table.to_csv(file, index=False)
        d = decide(table); todo = []
        new = beyond(d['etas'], d['eta'])          # best eta0 on the edge: try one more value
        if new:
            todo += [dict(family='apgd', reg=new, eps0=p) for p in d['epss']]
        new = beyond(d['epss'], d['eps'])          # the loosest tolerance still qualifies: loosen further
        if new and new > d['eps']:
            todo += [dict(family='apgd', reg=d['eta'], eps0=new)]
        new = beyond(d['lams'], d['lam'])
        if new:
            todo += [dict(family='ilqx', reg=new, eps0=0.)]
        if not todo:
            break
    done = {(r.family, r6(r.reg), r6(r.eps0), r.seed) for r in table.itertuples()}
    pair = [(scenario, dict(family='ilqx', reg=d['lam'], eps0=0.), sd) for sd in SELECT_SEEDS
            if ('ilqx', r6(d['lam']), 0., sd) not in done]   # selected ILQX on all seeds, for a paired comparison
    if pair:
        log('select', scenario, len(pair), 'ILQX runs for the paired comparison')
        with mp.Pool(workers) as pool:
            for row in pool.imap_unordered(score, pair):
                table = pd.concat([table, pd.DataFrame([row])], ignore_index=True); table.to_csv(file, index=False)
    d['tight'] = LADDER[int(np.argmin(np.abs(np.log(LADDER) - np.log(d['eps'])))) - 1]
    d['interior'] = not todo
    (folder / 'selected.json').write_text(json.dumps(d, indent=2) + '\n')
    log('selected', scenario, {k: d[k] for k in ('eta', 'eps', 'tight', 'lam', 'interior')})
    return d


# ------------------------------------------------------------------ timed runs on the December weeks
def methods(scenario):
    d = json.loads((OUT / scenario / 'selected.json').read_text())
    return ([dict(family='apgd', reg=d['eta'], eps0=e) for e in (d['eps'], d['tight'])]
            + [dict(family='mosek', reg=d['eta'], eps0=0.), dict(family='ilqx', reg=d['lam'], eps0=0.)])


def wait_idle():
    waited = 0
    while os.getloadavg()[0] > IDLE and waited < 7200:
        if waited % 600 == 0:
            log('machine busy, load', round(os.getloadavg()[0], 1), 'waiting')
        time.sleep(30); waited += 30


def live(s, cfg, secs, folder):
    """APGD with the true warm starts at every level where the worker drops snapshots."""
    J = len(s['trig'])
    for name, factor in levels(s):
        out = folder / 'live' / f'{tag(cfg)}_{name}.npz'
        trig_time = s['clock'][s['trig'] - 1] / factor
        if out.exists() or len(timeline(trig_time, secs, 'latest')[1]) == J:
            continue
        wait_idle()
        start = []; finish = []; solved = []; thetas = []; warm = None; free = -np.inf; nxt = 0
        while nxt < J:
            begin = max(free, trig_time[nxt])
            j = int(np.searchsorted(trig_time, begin, side='right')) - 1
            n = s['n_at'][j]; t0 = time.perf_counter()
            th, L, _ = update(cfg, s['phi'][:n], s['y'][:n], warm, None)
            free = begin + time.perf_counter() - t0
            start.append(begin); finish.append(free); solved.append(j); thetas.append(th)
            warm = (th, L); nxt = j + 1
        np.savez(out, start=start, finish=finish, solved=solved, thetas=np.array(thetas), load=os.getloadavg()[0])


def archive(scenario):
    """Move test runs of settings that are no longer selected to OUT/../formal_superseded."""
    keep = {tag(c) for c in methods(scenario)}
    for f in sorted((OUT / scenario).glob('*/seed*/*.npz')) + sorted((OUT / scenario).glob('*/seed*/live/*.npz')):
        stem = f.stem if f.parent.name != 'live' else f.stem.rsplit('_', 1)[0]
        if stem not in keep:
            dest = OUT.parent / (OUT.name + '_superseded') / f.relative_to(OUT)
            dest.parent.mkdir(parents=True, exist_ok=True); f.rename(dest)
            log('archived', f.relative_to(OUT))


def test(scenario):
    archive(scenario)
    solvers.prox_half_sq_pnorm(np.ones(su.KAPPA), 1., su.Q)       # compile outside measurements
    for split, day in TEST:
        for seed in SEEDS:
            folder = OUT / scenario / day / f'seed{seed}'; (folder / 'live').mkdir(parents=True, exist_ok=True)
            s = stream(scenario, split, day, seed)
            for cfg in methods(scenario):
                out = folder / (tag(cfg) + '.npz')
                try:
                    if not out.exists():
                        wait_idle()
                        thetas, secs, repaired, peak = chain(s, cfg)
                        np.savez(out, thetas=thetas, secs=secs, cfg=json.dumps(cfg), repaired=repaired, load=peak)
                        log(scenario, day, seed, tag(cfg), 'total', round(secs.sum(), 1), 's, repaired', repaired,
                            'peak load', round(peak, 1))
                    if cfg['family'] == 'apgd':
                        live(s, cfg, np.load(out)['secs'], folder)
                except Exception as error:                       # keep the remaining runs going
                    log('FAILED', scenario, day, seed, tag(cfg), repr(error))


# ------------------------------------------------------------------ replay
def evaluate():
    rows = []
    for file in sorted(OUT.glob('*/*/seed*/*.npz')):
        scenario, day, seed = file.parts[-4], file.parts[-3], int(file.parts[-2][4:])
        split = 'val' if day == VAL[1] else 'test'
        s = stream(scenario, split, day, seed); saved = np.load(file); cfg = json.loads(str(saved['cfg']))
        thetas, secs = saved['thetas'], saved['secs']; ex, target = s['ex'], s['target']
        instant = revenue(s, thetas, target)
        for name, factor in levels(s):
            clock = s['clock'] / factor
            finish, solved = timeline(clock[s['trig'] - 1], secs, 'latest')
            begin = finish - secs[solved]; used_thetas = thetas; busy_load = float(saved['load'])
            real = file.parent / 'live' / f'{file.stem}_{name}.npz'
            if real.exists():                                    # measured with the true warm starts
                r = np.load(real); begin, finish, solved = r['start'], r['finish'], r['solved']
                used_thetas = np.zeros_like(thetas); used_thetas[solved] = r['thetas']
                busy_load = max(busy_load, float(r['load']))
            pos = np.searchsorted(finish, clock[ex], side='right') - 1
            used = np.where(pos >= 0, solved[np.maximum(pos, 0)], -1)
            deployed = revenue(s, used_thetas, used)
            # Algorithm 1 as written: a request of epoch k is priced with the estimate of epoch k once it is
            # finished, and with the recorded metered fare before that
            locked = deployed.copy(); late = np.zeros(T, bool); late[ex] = used != target
            locked[late] = s['tsp'][late]
            busy = np.zeros(T, bool)                             # worker computing when the request arrives
            busy[ex] = np.searchsorted(begin, clock[ex], side='right') - 1 > pos
            done = np.full(len(secs), np.inf); done[solved] = finish
            timely = done <= clock[s['first']]
            behind = s['n_at'][np.maximum(target, 0)] - np.where(used >= 0, s['n_at'][np.maximum(used, 0)], 0)
            for window, share in WINDOWS.items():
                h = int(share * T); m = ex < h
                rows.append(dict(cfg, scenario=scenario, day=day, seed=seed, level=name, window=window,
                                 rate=(T - 1) / clock[-1], oracle=s['oracle'][:h].sum(), tsp=s['tsp'][:h].sum(),
                                 instant=instant[:h].sum(), deployed=deployed[:h].sum(),
                                 fare=np.where(busy, s['tsp'], deployed)[:h].sum(),
                                 explore=np.where(busy, s['wander'], deployed)[:h].sum(),
                                 lost=np.where(busy, 0., deployed)[:h].sum(), lock=locked[:h].sum(),
                                 busy=busy[:h][~s['explore'][:h]].mean(),
                                 timely=timely[s['due'] & (s['trig'] <= h)].mean(),
                                 stale=np.mean((used != target)[m]), fallback=np.mean((used < 0)[m]),
                                 behind=np.mean(behind[m] / s['n_at'][np.maximum(target[m], 0)]),
                                 solved=len(solved) / len(secs), seconds=secs.sum(),
                                 repaired=int(saved['repaired']), machine_load=busy_load))
    frame = pd.DataFrame(rows)
    frame.to_csv(run.ROOT / ('formal_smoke.csv' if SMOKE else 'formal.csv'), index=False)
    return frame


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select(sys.argv[2])
    elif sys.argv[1] == 'test':
        test(sys.argv[2])
    else:
        print(len(evaluate()), 'rows')
