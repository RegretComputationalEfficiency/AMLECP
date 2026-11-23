"""Tables and figure of the HVFHV case study (2026-11-17). Reads formal.csv.

Revenues are percent of the oracle revenue of the same window, losses are percentage
points (revenue with instantaneous updates minus revenue with the measured update
times). Means and standard errors are over the runs (four September 2026 Mondays x three seeds).
The gain over HPF is the ratio of the mean revenues minus one, as in Table 4 of the paper.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEVELS = ['native', '10', '30', '100', '300', '1000', '3000', '10000', '30000', '100000']
NAME = dict(apgd='AMLE-PGD', mosek='AMLE-MOSEK', ilqx='ILQX')
BUSY = 7.      # peak one-minute machine load above which the update times of a run are not used


def read(scenario):
    d = pd.read_csv(ROOT / 'formal.csv', dtype={'level': str}); d = d[d.scenario == scenario].copy()
    for c in ('instant', 'deployed', 'tsp', 'lock'):
        if c in d:
            d[c] = 100 * d[c] / d.oracle
    d['loss'] = d.instant - d.deployed
    if 'lock' in d:
        d['lock_loss'] = d.instant - d.lock
    d['run'] = d.day + '/' + d.seed.astype(str)
    sel = json.loads((ROOT / 'results/formal' / scenario / 'selected.json').read_text())
    d['policy'] = np.where(d.family == 'apgd', np.where(np.isclose(d.eps0, sel['eps']), 'AMLE-PGD', 'AMLE-PGD tight'),
                           d.family.map(NAME))
    return d, sel


def stat(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def ratio_of_means(x, y):
    """100 (mean(x) / mean(y) - 1) and its delta-method standard error over the runs."""
    r = x.mean() / y.mean()
    return 100 * (r - 1), 100 * np.std(x.to_numpy() - r * y.to_numpy(), ddof=1) / np.sqrt(len(x)) / y.mean()


def crossing(rate, loss, level):
    """Request rate at which the mean loss first reaches level (log-log interpolation)."""
    hit = np.flatnonzero(loss >= level)
    if not len(hit):
        return None
    i = hit[0]
    if i == 0:
        return rate[0]
    y0, y1 = max(loss[i - 1], 1e-4), loss[i]
    return float(np.exp(np.log(rate[i - 1]) + (np.log(level) - np.log(y0)) / (np.log(y1) - np.log(y0))
                        * (np.log(rate[i]) - np.log(rate[i - 1]))))


def validation(scenario, sel, say, seeds=(1, 2, 3)):
    """Two-step selection on the validation week: regret (% of oracle) of APGD, then the paired
    comparison of the selected APGD and ILQX settings over the same seeds."""
    t = pd.read_csv(ROOT / 'results/formal' / scenario / 'selection.csv')
    t = t[t.seed.isin(seeds)].assign(reg=t.reg.map(lambda v: float(f'{v:.6g}')),
                                      eps0=t.eps0.map(lambda v: float(f'{v:.6g}')), regret=lambda x: 100 * (1 - x.share))
    a = t[t.family == 'apgd'].groupby(['reg', 'eps0']).regret.agg(['mean', 'size'])
    a = a[a['size'] == len(seeds)]['mean'].unstack()
    if 'etas' in sel and 'epss' in sel:                      # the grid of the selection being reported
        a = a.loc[[v for v in a.index if np.isclose(sel['etas'], v).any()],
                  [v for v in a.columns if np.isclose(sel['epss'], v).any()]]
    say(f'\n〇、验证周选参（2026-09-07 周一 × 三颗种子 = {len(seeds)} 次，平均遗憾占 oracle 的 %）')
    say('  η̊ \\ ε̊   ' + ''.join(f'{e:>9g}' for e in a.columns))
    for eta, row in a.iterrows():
        mark = f' ← ε̊={sel["eps"]:g}' if np.isclose(eta, sel['eta']) else ''
        say(f'  {eta:<9g}' + ''.join('        -' if np.isnan(v) else f'{v:9.3f}' for v in row) + mark)
    if 'rule' in sel:
        say(f'  联合选参：全部格点中平均遗憾最小的 (η̊, ε̊)=({sel["eta"]:g}, {sel["eps"]:g})，'
            f'{"在网格内部" if sel["interior"] else "在网格边界"}。')
    else:
        say(f'  第一步：ε̊={a.columns.min():g} 下遗憾最小的 η̊={sel["eta"]:g}；'
            f'第二步：该行遗憾不超过最小值 1.01 倍的最大 ε̊={sel["eps"]:g}。')
    pick = t[(t.family == 'apgd') & np.isclose(t.reg, sel['eta']) & np.isclose(t.eps0, sel['eps'])].set_index('seed').regret
    ilqx = t[(t.family == 'ilqx') & np.isclose(t.reg, sel['lam'])].set_index('seed').regret
    both = pick.index.intersection(ilqx.index)
    if len(both) > 1:
        m, se = stat(ilqx[both] - pick[both])
        say(f'  选出的 AMLE-PGD 遗憾 {pick[both].mean():.3f}，ILQX（λ̊={sel["lam"]:g}）{ilqx[both].mean():.3f}，'
            f'PGD 收益 − ILQX 收益（配对，{len(both)} 颗种子）：{m:+.3f} ± {se:.3f}')


def report(scenario):
    every, sel = read(scenario)
    dirty = every[every.machine_load >= BUSY].run.unique()
    d = every[~every.run.isin(dirty)]                       # runs whose update times are usable
    a = d[d.window == 'all']; out = []; say = out.append
    policies = ['AMLE-PGD', 'AMLE-PGD tight', 'AMLE-MOSEK', 'ILQX']
    rate = a.groupby('level').rate.mean()[LEVELS]
    say(f'HVFHV 实验 {scenario}（κ=729）：2026 年 9 月后三个周一 × 三颗种子，每次前 200,000 个请求。')
    say(f'验证周选出 η̊={sel["eta"]:g}，ε̊={sel["eps"]:g}（较紧一档 {sel["tight"]:g}），λ̊={sel["lam"]:g}，'
        f'{"均在网格内部" if sel["interior"] else "有入选值在网格边界"}。')
    say(f'运行数 {every.run.nunique()}，其中计时受干扰（峰值负载 ≥ {BUSY:g}）而不计入延迟指标的 {len(dirty)} 次。'
        f'MOSEK 未认证而改用 APGD 高精度重解的更新共 {int(every[every.level == "native"].drop_duplicates(["run", "policy"]).repaired.sum())} 次。')
    say(f'原始到达率 {rate["native"]:.2f} 请求/秒。历史报价（base_passenger_fare）收益 {stat(every[(every.window == "all") & (every.level == "native")].drop_duplicates("run").tsp)[0]:.2f}。')
    validation(scenario, sel, say)

    say('\n一、即时收益（占 oracle，%）、相对 HPF 的提升（平均收益之比）与全程更新时间')
    base = every[(every.window == 'all') & (every.level == 'native')]
    for p in policies:
        g = base[base.policy == p]; m, se = stat(g.instant); lift = ratio_of_means(g.instant, g.tsp)
        t = stat(a[(a.policy == p) & (a.level == 'native')].seconds)
        say(f'  {p:<15} {m:6.2f} ± {se:.2f}   较 HPF {lift[0]:+.1f}% ± {lift[1]:.1f}   更新时间 {t[0]:7.1f} ± {t[1]:.1f} 秒')
    pair = base.pivot_table(index='run', columns='policy', values='instant')
    for other in ('AMLE-MOSEK', 'ILQX'):
        m, se = stat(pair['AMLE-PGD'] - pair[other])
        say(f'  AMLE-PGD − {other}（配对）：{m:+.3f} ± {se:.3f}')

    for window, title in (('all', '全部请求'), ('first20', '前 20%'), ('first5', '前 5%')):
        w = d[d.window == window]
        say(f'\n二、延迟造成的收益损失（百分点，{title}，均值 ± 标准误）')
        say('  请求/秒        ' + ''.join(f'{("原始" if l == "native" else l):>12}' for l in LEVELS))
        for p in policies:
            g = w[w.policy == p].pivot_table(index='run', columns='level', values='loss')[LEVELS]
            say(f'  {p:<15}' + ''.join(f'{g[l].mean():7.2f}±{g[l].std(ddof=1) / np.sqrt(len(g)):.2f}' for l in LEVELS))
            cross = [crossing(rate.values, g.mean().values, v) for v in (.1, .5, 1.)]
            say('  ' + ' ' * 15 + '损失达到 0.1 / 0.5 / 1 个百分点的请求率：' +
                ' / '.join('未达到' if c is None else f'{c:,.0f}' for c in cross))

    say('\n三、及时率（%）、所用估计缺少的探索样本占比（%）、实际求解的快照占比（%）')
    for column, text in (('timely', '及时率'), ('behind', '缺少样本'), ('solved', '求解快照')):
        say(f'  {text}  ' + ''.join(f'{("原始" if l == "native" else l):>9}' for l in LEVELS))
        for p in policies:
            g = a[a.policy == p].groupby('level')[column].mean()[LEVELS]
            say(f'  {p:<15}' + ''.join(f'{100 * v:9.1f}' for v in g))

    say('\n四、按周（全部请求）：即时收益；原始到达率下的及时率；10,000 请求/秒下的损失')
    for day, g in a.groupby('day'):
        row = []
        for p in ('AMLE-PGD', 'AMLE-MOSEK', 'ILQX'):
            h = g[g.policy == p]
            row.append(f'{p} {h[h.level == "native"].instant.mean():.2f} / {100 * h[h.level == "native"].timely.mean():.1f}% / '
                       f'{h[h.level == "10000"].loss.mean():.2f}')
        say(f'  {day}：' + '；'.join(row))

    rules = [('deployed', '用最新已完成估计（基准）')]
    if 'lock' in d:
        rules.append(('lock', '按 Algorithm 1（本周期估计未完成时按计价表车费）'))
    for column, title in rules:
        loss = 'loss' if column == 'deployed' else 'lock_loss'
        say(f'\n五、{title}')
        for window, text in (('first5', '前 5%'), ('all', '全部请求')):
            w = d[d.window == window]
            say(f'  {text}：延迟损失（百分点）')
            say('  请求/秒        ' + ''.join(f'{("原始" if l == "native" else l):>12}' for l in LEVELS))
            for p in ('AMLE-PGD', 'AMLE-MOSEK', 'ILQX'):
                g = w[w.policy == p].pivot_table(index='run', columns='level', values=loss)[LEVELS]
                say(f'  {p:<15}' + ''.join(f'{g[l].mean():7.2f}±{g[l].std(ddof=1) / np.sqrt(len(g)):.2f}' for l in LEVELS))
            pair = w.pivot_table(index=['run', 'level'], columns='policy', values=column)
            say(f'  {text}：部署收益差（百分点，配对）')
            for other in ('ILQX', 'AMLE-MOSEK'):
                x = (pair['AMLE-PGD'] - pair[other]).unstack('level')[LEVELS]
                say(f'  PGD − {other:<9}' + ''.join(f'{x[l].mean():+7.2f}±{x[l].std(ddof=1) / np.sqrt(len(x)):.2f}' for l in LEVELS))
    f5 = d[d.window == 'first5']
    say('\n六、前 5% 的及时率（%）与所用估计缺少的探索样本占比（%）')
    for column, text in (('timely', '及时率'), ('behind', '缺少样本')):
        say(f'  {text}  ' + ''.join(f'{("原始" if l == "native" else l):>9}' for l in LEVELS))
        for p in ('AMLE-PGD', 'AMLE-MOSEK', 'ILQX'):
            g = f5[f5.policy == p].groupby('level')[column].mean()[LEVELS]
            say(f'  {p:<15}' + ''.join(f'{100 * v:9.1f}' for v in g))
    return '\n'.join(out)


def figure(scenario='wtp16'):
    """(a) timeliness of the updates in the first 5% of requests, (b) revenue lost to delay over all
    requests with the newest finished estimate, (c) the same under the epoch rule of Algorithm 1.
    Against the request rate. Readable in grayscale."""
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nyc-delay-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    every, _ = read(scenario)
    d = every[~every.run.isin(every[every.machine_load >= BUSY].run.unique())]
    rate = d[d.window == 'all'].groupby('level').rate.mean()[LEVELS]
    style = {'AMLE-PGD': ('#111111', 's', '-'), 'AMLE-MOSEK': ('#555555', 'D', '--'), 'ILQX': ('#888888', 'o', ':')}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    floor = .003
    panels = [('first5', 'timely', 'Timely updates, first 5% of requests (%)'),
              ('all', 'loss', 'Revenue lost to delay (points)\nnewest finished estimate'),
              ('all', 'lock_loss', 'Revenue lost to delay (points)\nestimate of the own epoch')]
    for ax, (window, column, label) in zip(axes, panels):
        w = d[d.window == window]
        for p, (color, marker, line) in style.items():
            g = w[w.policy == p].groupby('level')[column].mean()[LEVELS]
            y = 100 * g.values if column == 'timely' else np.maximum(g.values, floor)
            ax.plot(rate.values, y, color=color, marker=marker, ls=line, ms=5, lw=1.5, mfc='white', mew=1.1, label=p)
        ax.set_xscale('log'); ax.set_xlabel('Requests per second'); ax.set_ylabel(label)
        ax.grid(axis='y', color='#dddddd', lw=.6)
        if column != 'timely':
            ax.set_yscale('log')
    axes[0].legend(frameon=False, fontsize=8.5)
    for ax, tag in zip(axes, 'abc'):
        ax.set_title(f'({tag})', loc='left', fontsize=9.5)
    fig.tight_layout(w_pad=2.2)
    path = ROOT / 'figures' / f'formal_pressure_{scenario}.png'
    fig.savefig(path, dpi=200); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)
    for k, (window, column, label) in enumerate(panels, 1):    # one file per panel, for subcaptions
        one, ax = plt.subplots(figsize=(3.0, 2.45)); w = d[d.window == window]
        for p, (color, marker, line) in style.items():
            g = w[w.policy == p].groupby('level')[column].mean()[LEVELS]
            y = 100 * g.values if column == 'timely' else np.maximum(g.values, floor)
            ax.plot(rate.values, y, color=color, marker=marker, ls=line, ms=5, lw=1.5, mfc='white', mew=1.1, label=p)
        ax.set_xscale('log'); ax.set_xlabel('Requests per second'); ax.grid(axis='y', color='#dddddd', lw=.6)
        ax.set_ylabel('Timely updates (%)' if column == 'timely' else 'Revenue lost to delay (points)')
        if column != 'timely':
            ax.set_yscale('log')
        if k == 1:
            ax.legend(frameon=False, fontsize=8.5)
        one.tight_layout(); one.savefig(path.with_name(f'formal_pressure_{scenario}_{k}.pdf')); plt.close(one)
    return path


if __name__ == '__main__':
    done = sorted(pd.read_csv(ROOT / 'formal.csv', usecols=['scenario']).scenario.unique(),
                  key=['wtp16', 'wtp14', 'wtp18'].index)
    text = '\n\n\n'.join(report(s) for s in (sys.argv[1:] or done))
    (ROOT / 'formal_report.txt').write_text(text + '\n')
    print(text)
    for s in (sys.argv[1:] or done):
        print(figure(s))
