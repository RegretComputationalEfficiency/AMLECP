"""Robustness of Case Study II to the median willingness-to-pay multiplier.

The simulators wtp14 and wtp18 (data/sim, see simulator.json) differ from the reported wtp16 only in the median
multiplier, 1.4 and 1.8 instead of 1.6, and are fitted to the same two million trips. Everything else is
formal.py: the same requests, features, exploration schedule (m0 = 410, m_k = 27), seeds and solver.

Only AMLE-PGD and HPF are compared, with updates available at once (the column "% of oracle" of Table 4).
For each scenario, AMLE-PGD is tuned on the validation Monday by the APGD part of formal.decide (eta0 with the
smallest mean regret at the tightest tolerance, then the largest eps0 within 1% of the smallest regret on its
path, with the same edge extensions), and the selected configuration is run on the three test Mondays with three
seeds each. As a check, the configuration selected for wtp16 (eta0 = 1, eps0 = 0.001) is also run on the test
Mondays. Times are not used, so the runs may share the machine.

    python robust_wtp.py select wtp14     # selection.csv and selected.json in results/robust_wtp/wtp14
    python robust_wtp.py test wtp14       # test.csv in the same folder
    python robust_wtp.py report           # robust_wtp_report.txt, with the wtp16 rows of formal.csv
"""
import run  # sets the single-thread environment before numpy is imported
import json
import multiprocessing as mp
import sys
import numpy as np
import pandas as pd
import formal as F

OUT = run.RESULTS / ('robust_wtp_smoke' if F.SMOKE else 'robust_wtp')
MAIN = dict(eta=1.0, eps=0.001)                    # selected for wtp16 (results/formal/wtp16/selected.json)
MEDIAN = dict(wtp14=1.4, wtp16=1.6, wtp18=1.8)


def decide_apgd(table):
    """APGD part of formal.decide."""
    t = table.assign(reg=table.reg.map(F.r6), eps0=table.eps0.map(F.r6))
    a = t[t.seed.isin(F.SELECT_SEEDS)].groupby(['reg', 'eps0']).share
    regret = (1 - a.mean())[a.size() == len(F.SELECT_SEEDS)]
    first = regret.xs(F.r6(min(F.GRID['eps'])), level='eps0')
    eta = first.idxmin(); path = regret.xs(eta, level='reg')
    return dict(eta=float(eta), eps=float(path[path <= 1.01 * path.min()].index.max()),
                etas=sorted(float(v) for v in first.index), epss=sorted(float(v) for v in path.index))


def select(scenario, workers=4):
    folder = OUT / scenario; folder.mkdir(parents=True, exist_ok=True); file = folder / 'selection.csv'
    table = pd.read_csv(file) if file.exists() else pd.DataFrame(columns=['family', 'reg', 'eps0', 'seed', 'share'])
    todo = [dict(family='apgd', reg=e, eps0=p) for e in F.GRID['eta'] for p in F.GRID['eps']]
    for _ in range(8):
        done = {(F.r6(r.reg), F.r6(r.eps0), r.seed) for r in table.itertuples()}
        tasks = [(scenario, dict(c, reg=F.r6(c['reg']), eps0=F.r6(c['eps0'])), sd) for c in todo
                 for sd in F.SELECT_SEEDS if (F.r6(c['reg']), F.r6(c['eps0']), sd) not in done]
        if tasks:
            F.log('select', scenario, len(tasks), 'runs')
            with mp.Pool(workers) as pool:
                for row in pool.imap_unordered(F.score, tasks):
                    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True); table.to_csv(file, index=False)
        d = decide_apgd(table); todo = []
        new = F.beyond(d['etas'], d['eta'])
        if new:
            todo += [dict(family='apgd', reg=new, eps0=p) for p in d['epss']]
        new = F.beyond(d['epss'], d['eps'])
        if new and new > d['eps']:
            todo += [dict(family='apgd', reg=d['eta'], eps0=new)]
        if not todo:
            break
    d['interior'] = not todo
    (folder / 'selected.json').write_text(json.dumps(d, indent=2) + '\n')
    F.log('selected', scenario, {k: d[k] for k in ('eta', 'eps', 'interior')})
    return d


def one(task):
    scenario, cfg, label, day, seed = task
    s = F.stream(scenario, 'test', day, seed, 'alg1')
    thetas = F.chain(s, cfg)[0]
    instant = F.revenue(s, thetas, s['target'])
    return dict(scenario=scenario, label=label, reg=cfg['reg'], eps0=cfg['eps0'], day=day, seed=seed,
                oracle=s['oracle'].sum(), hpf=s['tsp'].sum(), instant=instant.sum())


def test(scenario, workers=4):
    folder = OUT / scenario
    d = json.loads((folder / 'selected.json').read_text())
    cfgs = [('selected', dict(family='apgd', reg=d['eta'], eps0=d['eps']))]
    if not (np.isclose(d['eta'], MAIN['eta']) and np.isclose(d['eps'], MAIN['eps'])):
        cfgs.append(('wtp16 setting', dict(family='apgd', reg=MAIN['eta'], eps0=MAIN['eps'])))
    tasks = [(scenario, cfg, label, day, seed) for label, cfg in cfgs for _, day in F.TEST for seed in F.SEEDS]
    F.log('test', scenario, len(tasks), 'runs')
    rows = []
    with mp.Pool(workers) as pool:
        for row in pool.imap_unordered(one, tasks):
            rows.append(row); pd.DataFrame(rows).to_csv(folder / 'test.csv', index=False)
    F.log('test done', scenario)


def report():
    frames = [pd.read_csv(f) for f in sorted(OUT.glob('*/test.csv'))]
    base = pd.read_csv(run.ROOT / 'formal.csv')
    base = base[(base.family == 'apgd') & np.isclose(base.eps0, MAIN['eps']) & (base.window == 'all')
                & (base.level == 'native') & (base.scenario == 'wtp16')]
    frames.append(pd.DataFrame(dict(scenario='wtp16', label='selected', reg=base.reg, eps0=base.eps0, day=base.day,
                                    seed=base.seed, oracle=base.oracle, hpf=base.tsp, instant=base.instant)))
    t = pd.concat(frames, ignore_index=True)
    lines = ['Case Study II, robustness to the median WTP multiplier. Revenue with updates available at once,',
             'first 200,000 requests of the three test Mondays x three seeds, in % of the oracle revenue.',
             'Gain over HPF = mean AMLE-PGD revenue / mean HPF revenue - 1 (ratio of means, as in Table 4).', '']
    lines.append(f'{"median":>7} {"setting":>14} {"eta0":>6} {"eps0":>7} {"runs":>5} {"HPF":>7} {"AMLE-PGD":>9} '
                 f'{"gain vs HPF":>12} {"min..max per run":>18}')
    for (sc, label), g in t.groupby(['scenario', 'label'], sort=True):
        hpf = 100 * g.hpf / g.oracle; pgd = 100 * g.instant / g.oracle; per = 100 * (g.instant / g.hpf - 1)
        gain = 100 * (pgd.mean() / hpf.mean() - 1)
        lines.append(f'{MEDIAN[sc]:>7.1f} {label:>14} {g.reg.iloc[0]:>6g} {g.eps0.iloc[0]:>7g} {len(g):>5} '
                     f'{hpf.mean():>7.2f} {pgd.mean():>9.2f} {gain:>+11.1f}% {per.min():>+8.1f}..{per.max():+.1f}%')
    text = '\n'.join(lines)
    (run.ROOT / ('robust_wtp_smoke_report.txt' if F.SMOKE else 'robust_wtp_report.txt')).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    if sys.argv[1] == 'select':
        select(sys.argv[2])
    elif sys.argv[1] == 'test':
        test(sys.argv[2])
    else:
        report()
