"""Fresh $1,000 invested each January (no compounding): top-1 ($1,000 in #1) vs top-10 ($100 each), 2016 -> latest. Each year's money is
sold at that year's end (2026: latest close) and is NOT reinvested. Reuses the yearly return series from 10_top10_vs_top1_yearly.py."""
import sys, io, runpy, contextlib; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()): g = runpy.run_path('src/10_top10_vs_top1_yearly.py')
res, spy, qqq = g['res'], np.array(g['spy']), np.array(g['qqq'])
yrs = list(res[(1, True, .001)][0].year)
S = {'Top-1 (A)': res[(1, True, .001)][0].ret.values, 'Top-1 (B, full prior yr)': res[(1, False, .001)][0].ret.values,
     'Top-10 (A)': res[(10, True, .001)][0].ret.values, 'Top-10 (B, full prior yr)': res[(10, False, .001)][0].ret.values, 'SPY': spy, 'QQQ': qqq}
pf = lambda v: f'{100*v:+.1f}%'
t = pd.DataFrame({'Year': yrs, 'Top-1 $1,000 -> ': [f'${1000*(1+r):,.0f}' for r in S['Top-1 (A)']], 'Top-10 ($100 each) $1,000 -> ': [f'${1000*(1+r):,.0f}' for r in S['Top-10 (A)']],
                  'SPY $1,000 ->': [f'${1000*(1+r):,.0f}' for r in S['SPY']], 'QQQ $1,000 ->': [f'${1000*(1+r):,.0f}' for r in S['QQQ']]})
print('### Each year: a fresh $1,000 invested in January, value at that year-end (reading A, 10 bps/side). Hypothetical backtest, not live results.'); print(t.to_markdown(index=False))
for lab, n in (('2016-2026 YTD (11 deposits = $11,000 put in)', 11), ('2016-2025 (10 complete years, 10 deposits = $10,000 put in)', 10)):
    rows = []
    for k, r in S.items():
        r = r[:n]; v = 1000 * (1 + r); rows.append({'Strategy': k, 'Total put in': f'${1000*n:,.0f}', 'Total value': f'${v.sum():,.0f}', 'Profit': f'${v.sum()-1000*n:,.0f}', 'Money multiple': f'{v.sum()/(1000*n):.2f}x',
                                                     'Avg yearly return': pf(r.mean()), 'Median yearly': pf(np.median(r)), 'Worst year $': f'${1000*r.min():,.0f}', 'Best year $': f'${1000*r.max():,.0f}', 'Years up': f'{(r>0).sum()}/{n}'})
    print(f'\n### {lab}. Hypothetical backtest, not live results.'); print(pd.DataFrame(rows).to_markdown(index=False))
r1, r10 = S['Top-1 (A)'], S['Top-10 (A)']
print('\nTop-1 without the 2026 SNDK year (2016-2025):  profit $%s of $10,000 put in' % f'{1000*r1[:10].sum():,.0f}')
print('Share of top-1 total profit coming from the single best year: %.0f%%;  top-10: %.0f%%' % (100 * r1.max() / r1.sum(), 100 * r10.max() / r10.sum()))
print('Years top-1 beat top-10: %d/11' % (r1 > r10).sum())
pd.DataFrame(S, index=yrs).to_csv('outputs/fixed_1000_per_year_returns.csv')
