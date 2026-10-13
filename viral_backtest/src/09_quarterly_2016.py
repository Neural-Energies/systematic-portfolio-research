"""Quarterly version, 2016Q1 -> latest. Each calendar quarter: rank PIT S&P 500 members (as of prior quarter-end) on the prior quarter's total return;
buy #1 at the first trading day's open, sell at the quarter's last trading day close (latest close for the current partial quarter). 100% in one stock."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, yfinance as yf
from engine import Data
D = Data(); cal = D.cal
q = yf.download('QQQ', start='2015-09-01', auto_adjust=True, progress=False); q.columns = [c[0] for c in q.columns]; q.index = pd.DatetimeIndex(q.index).tz_localize(None)
qk = pd.Series(np.arange(D.T), index=cal).groupby([cal.year, (cal.month - 1) // 3])
FIRST = qk.first().to_dict(); LAST = qk.last().to_dict()
Q = [(y, i) for y in range(2016, 2027) for i in range(4) if (y, i) in FIRST]
prev = lambda y, i: (y - 1, 3) if i == 0 else (y, i - 1)
def run(mode, cost):
    E = 1.0; eqs = {}; rows = []
    for (y, i) in Q:
        e = FIRST[(y, i)]; x = LAST[(y, i)]; b = LAST[prev(y, i)]; a = LAST[prev(*prev(y, i))]
        cand = []
        for t in D.members(cal[b]):
            k = D.k.get(t)
            if k is None or not all(np.isfinite(v) for v in (D.C[a, k], D.C[b, k], D.O[e, k], D.C[e, k])): continue
            cand.append((D.C[b, k] / D.C[a, k] - 1, t))
        cand.sort(reverse=True); r0, t = cand[0]; k = D.k[t]
        t0 = e if mode == 'open' else b; ep = D.O[e, k] if mode == 'open' else D.C[b, k]
        for j in range(t0, x + 1):
            c = D.C[j, k] if np.isfinite(D.C[j, k]) else ep
            eqs[cal[j]] = E * c / ep * (1 - cost) * ((1 - cost) if j == x else 1)
        xc = D.C[x, k] if np.isfinite(D.C[x, k]) else ep; g = xc / ep * (1 - cost) ** 2
        allg = [D.C[x, D.k[u]] / (D.O[e, D.k[u]] if mode == 'open' else D.C[b, D.k[u]]) - 1 for _, u in cand if np.isfinite(D.C[x, D.k[u]])]
        sp = D.spyC[x] / (D.spyO[e] if mode == 'open' else D.spyC[b]) - 1
        qq = q.Close.loc[cal[x]] / (q.Open.loc[cal[e]] if mode == 'open' else q.Close.loc[cal[b]]) - 1
        rows.append(dict(quarter=f'{y}Q{i+1}' + ('*' if (y, i) == Q[-1] else ''), pick=t, prior=r0, ret=g - 1, spy=sp, qqq=qq, univ_avg=np.mean(allg), n_elig=len(cand), n_mem=len(D.members(cal[b]))))
        E *= g
    return pd.DataFrame(rows), pd.Series(eqs).sort_index()
pf = lambda v: f'{100*v:+.1f}%'
res = {(m, c): run(m, c) for m in ('open', 'close') for c in (0, .001, .005)}
base, eq0 = res[('open', .001)]; base.to_csv('outputs/quarterly_since_2016_picks.csv', index=False)
yrs = (cal[LAST[Q[-1]]] - cal[FIRST[Q[0]]]).days / 365.25
rows = []
for (m, c), (R, eq) in res.items():
    end = 100 * (1 + R.ret).prod(); r = R.ret
    rows.append({'Variant': f'{m}-entry, {int(c*1e4)}bps/side', 'End $ (to 2026-10-13)': f'${end:,.0f}', 'CAGR': pf((end / 100) ** (1 / yrs) - 1), 'Vol (ann.)': f'{100*r.std()*2:.1f}%', 'Sharpe': f'{r.mean()/r.std()*2:.2f}',
                 'Max DD (daily)': pf((eq / eq.cummax() - 1).min()), 'Worst qtr': pf(r.min()), 'Best qtr': pf(r.max()), 'Up qtrs': f'{100*(r>0).mean():.0f}%', 'Beat SPY': f'{(r>R.spy).sum()}/{len(r)}'})
print(f'{len(base)} quarterly trades ({base.quarter.iloc[0]} to {base.quarter.iloc[-1]}; * = partial quarter to {cal[-1].date()})')
print(pd.DataFrame(rows).to_markdown(index=False))
print(f'SPY same quarterly windows: ${100*(1+base.spy).prod():,.0f} | QQQ: ${100*(1+base.qqq).prod():,.0f}')
print('\nQuarter by quarter (open-entry, 10bps):')
print(base.assign(prior=base.prior.map(pf), ret=base.ret.map(pf), spy=base.spy.map(pf), qqq=base.qqq.map(pf), univ_avg=base.univ_avg.map(pf))[['quarter', 'pick', 'prior', 'ret', 'spy', 'qqq', 'univ_avg']].rename(columns={'prior': 'prior-qtr ret', 'univ_avg': 'avg S&P stock'}).to_markdown(index=False))
base['year'] = base.quarter.str[:4]
yr = base.groupby('year').apply(lambda d: pd.Series({'strategy': (1 + d.ret).prod() - 1, 'SPY': (1 + d.spy).prod() - 1, 'QQQ': (1 + d.qqq).prod() - 1}))
print('\nBy calendar year:'); print(yr.map(pf).to_markdown())
lr = np.log1p(base.ret); top = lr.sort_values(ascending=False).index[:3]
print('\nBest 3 quarters:', [(base.quarter[i], base.pick[i], pf(base.ret[i])) for i in top], '= %.0f%% of log-growth; without them $%.0f' % (100 * lr[top].sum() / lr.sum(), 100 * np.exp(lr.drop(top).sum())))
print('Avg qtr return: pick %s | avg eligible stock %s | SPY %s | QQQ %s; pick beat avg stock in %d%% of quarters; picks\' avg prior-qtr return %s' % (pf(base.ret.mean()), pf(base.univ_avg.mean()), pf(base.spy.mean()), pf(base.qqq.mean()), 100 * (base.ret > base.univ_avg).mean(), pf(base.prior.mean())))
print('Coverage (eligible/members) 2016: %.2f, 2025: %.2f' % (base[base.year == '2016'].eval('n_elig/n_mem').mean(), base[base.year == '2025'].eval('n_elig/n_mem').mean()))
