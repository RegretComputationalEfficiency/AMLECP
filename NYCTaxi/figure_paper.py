"""Figure 5 of the revised manuscript (Section 5.2), from formal.csv (2026-11-19).

(a) revenue lost to delay, (b) revenue with delay, both against the request rate, under the rule of
Algorithm 1 (estimate fixed at the first exploitation request of an epoch, column 'freeze').
Top axis: request rate. Bottom axis: the equivalent factor by which every update time is multiplied at the recorded rate (rate / recorded rate).
    python figure_paper.py OUTDIR
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
L = [10, 30, 100, 300, 1000, 3000]
S = [('apgd', 'AMLE-PGD', '#2a78d6', 'o', '-'), ('mosek', 'AMLE-MOSEK', '#6f6e69', 's', '--'),
     ('ilqx', 'ILQX', '#eb6834', '^', '-.')]


def data():
    d = pd.read_csv(HERE / 'formal.csv')
    d = d[(d.window == 'all') & ((d.family != 'apgd') | np.isclose(d.eps0, .001))]
    native = d[d.level == 'native'].rate.mean()
    d = d[d.level.isin([str(v) for v in L])].copy(); d['r'] = d.level.astype(float)
    d['inst'] = 100 * d.instant / d.oracle; d['net'] = 100 * d.freeze / d.oracle; d['loss'] = d.inst - d.net
    g = d.groupby(['family', 'r']).agg(inst=('inst', 'mean'), net=('net', 'mean'), nse=('net', 'sem'),
                                       loss=('loss', 'mean'), se=('loss', 'sem')).reset_index()
    return g, native


def axes(ax, native):
    ax.set_xscale('log'); ax.set_xticks(L); ax.set_xticklabels([f'{v / native:.0f}' for v in L]); ax.minorticks_off()
    ax.set_xlabel('update time multiplier at the recorded rate'); ax.grid(axis='y', color='#e6e5e0', lw=.6)
    top = ax.secondary_xaxis('top'); top.set_xticks(L); top.set_xticklabels([f'{v:,}' for v in L])
    top.minorticks_off(); top.set_xlabel('requests per second', fontsize=8)


def main(out):
    g, native = data(); out = Path(out)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, a = plt.subplots(figsize=(3.3, 2.7))
    for f, lab, c, m, ls in S:
        x = g[g.family == f].sort_values('r')
        a.errorbar(x.r, x.loss, yerr=x.se, color=c, marker=m, ms=4.5, lw=1.6, ls=ls, capsize=2, label=lab)
    axes(a, native); a.set_ylabel('revenue lost to delay (% of oracle revenue)')
    a.legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(out / 'fig_taxi_delay_a.pdf')
    fig, b = plt.subplots(figsize=(3.3, 2.7))
    for (f, lab, c, m, ls), dx in zip(S, (.94, 1., 1.06)):
        x = g[g.family == f].sort_values('r')
        b.plot(x.r, x.inst, color=c, lw=.9, ls=':')
        b.fill_between(x.r, x.net, x.inst, color=c, alpha=.18, lw=0)
        b.plot(x.r, x.net, color=c, lw=1.6, ls=ls)
        b.errorbar(x.r * dx, x.net, yerr=x.nse, color=c, marker=m, ms=4.5, lw=0, elinewidth=1, capsize=2)
    axes(b, native); b.set_ylabel('revenue (% of oracle revenue)')
    fig.tight_layout(); fig.savefig(out / 'fig_taxi_delay_b.pdf')
    print(g.round(2).to_string(), '\nnative rate', round(native, 2))


if __name__ == '__main__':
    main(sys.argv[1])
