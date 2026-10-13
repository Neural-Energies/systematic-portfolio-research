"""Monthly version, Jan-2016 -> latest. Each month: rank PIT S&P 500 members (as of prior month-end) on that prior month's total return;
buy $ of #1 at the first trading day's OPEN, sell at the month's last trading day CLOSE (latest close for the current month). 100% in one stock.
Variant: close-to-close (buy at prior month-end close). Stocks with no price data are skipped (next one down)."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, yfinance as yf
from engine import Data
D = Data(); cal = D.cal; T = D.T
q = yf.download('QQQ', start='2015-11-01', auto_adjust=True, progress=False); q.columns = [c[0] for c in q.columns]; q.index = pd.DatetimeIndex(q.index).tz_localize(None)
ym = pd.Series(np.arange(T), index=cal).groupby([cal.year, cal.month])
FIRST = ym.first().to_dict(); LAST = ym.last().to_dict()
months = [(y, m) for y in range(2016, 2027) for m in range(1, 13) if (y, m) in FIRST and (y < 2026 or m <= 10)]
prev = lambda y, m: (y - 1, 12) if m == 1 else (y, m - 1)
def pick(y, m, mode):
    py, pm = prev(y, m); ppy, ppm = prev(py, pm); b = LAST[(py, pm)]; a = LAST[(ppy, ppm)]; e = FIRST[(y, m)]; out = []
    for t in D.members(cal[b]):
        k = D.k.get(t)
        if k is None: continue
        if not (np.isfinite(D.C[a, k]) and np.isfinite(D.C[b, k]) and np.isfinite(D.O[e, k]) and np.isfinite(D.C[e, k])): continue
        out.append((D.C[b, k] / D.C[a, k] - 1, t))
    out.sort(reverse=True); return out
def run(mode, cost):
    E = 1.0; eqs = {}; rows = []
    for (y, m) in months:
        e = FIRST[(y, m)]; x = LAST[(y, m)]; cand = pick(y, m, mode); r0, t = cand[0]; k = D.k[t]
        ep = D.O[e, k] if mode == 'open' else D.C[e - 1, k]
        t0 = e if mode == 'open' else e - 1                       # close-to-close: bought at prior day's (month-end) close
        eqs[cal[t0 - 1] if t0 > 0 else cal[0]] = eqs.get(cal[t0 - 1], E)
        for i in range(t0, x + 1):
            c = D.C[i, k] if np.isfinite(D.C[i, k]) else ep
            eqs[cal[i]] = E * c / ep * (1 - cost) * ((1 - cost) if i == x else 1)
        g = (D.C[x, k] if np.isfinite(D.C[x, k]) else ep) / ep * (1 - cost) ** 2
        mem = D.members(cal[LAST[prev(y, m)]]); allg = [D.C[x, D.k[u]] / (D.O[e, D.k[u]] if mode == 'open' else D.C[e - 1, D.k[u]]) - 1 for _, u in cand[:400] if np.isfinite(D.C[x, D.k[u]])]
        sp = D.spyC[x] / (D.spyO[e] if mode == 'open' else D.spyC[e - 1]) - 1
        qq = q.Close.loc[cal[x]] / (q.Open.loc[cal[e]] if mode == 'open' else q.Close.loc[cal[e - 1]]) - 1
        rows.append(dict(month=f'{y}-{m:02d}', pick=t, prior=r0, ret=g - 1, spy=sp, qqq=qq, univ_avg=np.mean(allg), n_elig=len(cand), n_mem=len(mem)))
        E *= g
    return pd.DataFrame(rows), pd.Series(eqs).sort_index()
def st(R, eq):
    r = R.ret; yrs = (cal[LAST[months[-1]]] - cal[FIRST[months[0]]]).days / 365.25; end = 100 * (1 + r).prod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(end=end, cagr=(end / 100) ** (1 / yrs) - 1, vol=r.std() * np.sqrt(12), sharpe=r.mean() / r.std() * np.sqrt(12), mdd=dd, worst=r.min(), best=r.max(), beat=(r > R.spy).mean(), up=(r > 0).mean())
pf = lambda v: f'{100*v:+.1f}%'
res = {}
for mode in ('open', 'close'):
    for c in (0, .001, .005): res[(mode, c)] = run(mode, c)
base, eq0 = res[('open', .001)]; base.to_csv('outputs/monthly_since_2016_picks.csv', index=False)
rows = []
for (mode, c), (R, eq) in res.items():
    s = st(R, eq); rows.append({'Variant': f'{mode}-entry, {int(c*1e4)}bps/side', 'End $ (to 2026-10-13)': f"${s['end']:,.0f}", 'CAGR': pf(s['cagr']), 'Vol (ann.)': pf(s['vol']).lstrip('+'), 'Sharpe': f"{s['sharpe']:.2f}",
                                'Max DD (daily)': pf(s['mdd']), 'Worst month': pf(s['worst']), 'Best month': pf(s['best']), 'Up months': f"{100*s['up']:.0f}%", 'Beat SPY': f"{100*s['beat']:.0f}%"})
n = len(base); spyv = 100 * (1 + base.spy).prod(); qv = 100 * (1 + base.qqq).prod()
print(f'{n} monthly trades, {base.month.iloc[0]} to {base.month.iloc[-1]} (last month partial, to {cal[-1].date()})')
print(pd.DataFrame(rows).to_markdown(index=False)); print(f'SPY (same months, open->close): ${spyv:,.0f} | QQQ: ${qv:,.0f}  [SPY/QQQ by compounding the same monthly windows]')
base['year'] = base.month.str[:4]
yr = base.groupby('year').apply(lambda d: pd.Series({'strategy': (1 + d.ret).prod() - 1, 'SPY': (1 + d.spy).prod() - 1, 'QQQ': (1 + d.qqq).prod() - 1, 'worst month': d.ret.min(), 'months beating SPY': f'{(d.ret > d.spy).sum()}/{len(d)}'}))
print('\nBy calendar year (open-entry, 10bps):'); print(yr.assign(**{c: yr[c].map(pf) for c in ('strategy', 'SPY', 'QQQ', 'worst month')}).to_markdown())
lr = np.log1p(base.ret); top = lr.sort_values(ascending=False).index[:5]; t3 = top[:3]
print('\nBest 5 months:', [(base.month[i], base.pick[i], pf(base.ret[i])) for i in top])
print('Best 3 months = %.0f%% of total log-growth; without them $%.0f; without best 5: $%.0f' % (100 * lr[t3].sum() / lr.sum(), 100 * np.exp(lr.drop(t3).sum()), 100 * np.exp(lr.drop(top).sum())))
print('Worst 5 months:', [(base.month[i], base.pick[i], pf(base.ret[i])) for i in base.ret.sort_values().index[:5]])
print('\nAvg monthly return: pick %s | avg eligible S&P stock %s | SPY %s | QQQ %s ; median pick %s' % (pf(base.ret.mean()), pf(base.univ_avg.mean()), pf(base.spy.mean()), pf(base.qqq.mean()), pf(base.ret.median())))
print('Avg prior-month return of the picks: %s; pick beat the average eligible stock in %d%% of months' % (pf(base.prior.mean()), 100 * (base.ret > base.univ_avg).mean()))
print('Months where pick had prior-month return > +100%% (audit for data glitches/spin-offs): ', [(r.month, r.pick, pf(r.prior), pf(r.ret)) for r in base.itertuples() if r.prior > 1])
print('Most frequent picks:', base.pick.value_counts().head(8).to_dict())
print('Eligible/members coverage by year (avg):', (base.groupby('year').apply(lambda d: (d.n_elig / d.n_mem).mean())).round(2).to_dict())
yy = pd.read_csv('outputs/since_2016.csv'); print('\nFor comparison, yearly version (A, open-entry, no costs) 2016-2026 YTD: $118,777 (SNDK) / $42,161 (strict); 2016-2025: $20,000')
