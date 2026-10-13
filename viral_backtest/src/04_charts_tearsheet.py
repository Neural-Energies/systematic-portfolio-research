import sys, pickle, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import HB
res = pickle.load(open('outputs/results.pkl', 'rb')); S = res['s']; SPY = res['spy']
BG, INK, MUT, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C = dict(a='#2a78d6', b='#eb6834', c='#1baf7a', spy='#8a8984')
def style(ax):
    ax.set_facecolor(BG); [ax.spines[k].set_visible(False) for k in ('top', 'right')]
    for k in ('left', 'bottom'): ax.spines[k].set_color(GRID)
    ax.grid(axis='y', color=GRID, lw=.8); ax.tick_params(colors=MUT, labelsize=9)

for y in (1997, 2012):
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG); style(ax)
    ser = [('Top-1, no stop', S[f'1|none|{y}']['eq'], C['a']), ('Top-1, 35% stop', S[f'1|35%|{y}']['eq'], C['b']),
           ('EW top-10, no stop', S[f'10|none|{y}']['eq'], C['c']), ('S&P 500 (SPY)', SPY[y]['eq'], C['spy'])]
    for lab, e, col in ser:
        ax.plot(e.index, 100 * e.values, color=col, lw=2, label=f'{lab}  (${100*e.iloc[-1]:,.0f})')
    ax.set_yscale('log'); ax.set_ylabel('Value of $100 (log scale)', color=MUT)
    ax.set_title(f'Growth of $100 from {y}: last year\'s best S&P 500 stock\nPIT membership, Yahoo total-return prices, 10bps/side. {HB}.', loc='left', color=INK, fontsize=12)
    ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK)
    fig.text(.01, .01, 'Price data omits many delisted members (survivorship-limited). ' + HB + '.', color=MUT, fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(f'outputs/log_equity_from_{y}.png', dpi=150, facecolor=BG); plt.close(fig)

# year-by-year bars (1997 sample covers 2012 sample); complete years only
fig, axs = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
yrs = S['1|none|1997']['years']; x = np.arange(len(yrs)); w = .38
for ax, key, lab, col in ((axs[0], '1|none|1997', 'Top-1, no stop', C['a']), (axs[1], '1|35%|1997', 'Top-1, 35% stop', C['b'])):
    style(ax); r = S[key]['yearly'] * 100; sp = S[key]['spy_yearly'] * 100
    ax.bar(x - w / 2, r, w, color=col, label=lab); ax.bar(x + w / 2, sp, w, color=C['spy'], label='S&P 500 (SPY)')
    ax.axhline(0, color=MUT, lw=.8); ax.set_ylabel('Year return, %', color=MUT); ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK)
    ax.set_title(f'{lab} vs SPY by year (beat SPY {int((r>sp).sum())}/{len(r)}). {HB}.', loc='left', color=INK, fontsize=11)
axs[1].set_xticks(x); axs[1].set_xticklabels(yrs, rotation=60, fontsize=8)
fig.text(.01, .005, 'Complete calendar years 1997-2025; 2026 YTD omitted. ' + HB + '.', color=MUT, fontsize=8)
fig.tight_layout(rect=(0, .02, 1, 1)); fig.savefig('outputs/yearly_bars_vs_spy.png', dpi=150, facecolor=BG); plt.close(fig)

# QuantStats tear sheets (main variant = top-1, no stop, 1997 start; plus the 35%-stop variant)
import quantstats as qs
spy_r = SPY[1997]['eq'].pct_change().dropna(); spy_r.index = pd.DatetimeIndex(spy_r.index); spy_r.name = 'SPY'
for key, nm in (('1|none|1997', 'top1_nostop'), ('1|35%|1997', 'top1_stop35')):
    r = S[key]['eq'].pct_change().dropna(); r.index = pd.DatetimeIndex(r.index)
    qs.reports.html(r, benchmark=spy_r, output=f'outputs/quantstats_{nm}.html', title=f'{nm} - {HB}', rf=0.0)
    print('tearsheet', nm)
