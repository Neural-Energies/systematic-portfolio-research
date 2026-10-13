"""Yearly, 2016 -> latest: $1,000 split equally ($100 each) across the prior calendar year's top-10 S&P 500 stocks vs $1,000 all-in on #1.
Buy at the first trading day's open of January, sell at the December last close (2026: latest close). Compounded, no stop, no intra-year rebalance.
Reading A = names with partial prior-year history may rank (viral post's rule, SNDK in 2026). Reading B = full prior year required."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, yfinance as yf
from engine import Data
D = Data(); cal = D.cal; last = D.T - 1
q = yf.download('QQQ', start='2015-12-01', auto_adjust=True, progress=False); q.columns = [c[0] for c in q.columns]; q.index = pd.DatetimeIndex(q.index).tz_localize(None)
dec = lambda y: np.where(cal.year == y)[0][-1]; first = lambda y: np.where(cal.year == y)[0][0]
def ranked(Y, partial):
    b = dec(Y - 1); a = dec(Y - 2); out = []
    for t in D.members(cal[b]):
        k = D.k.get(t)
        if k is None or not np.isfinite(D.C[b, k]) or not np.isfinite(D.O[first(Y), k]): continue
        c0 = D.C[a, k]; p = False
        if not np.isfinite(c0):
            if not partial: continue
            f = np.where(np.isfinite(D.C[a + 1:b + 1, k]))[0]
            if not len(f): continue
            c0 = D.C[a + 1:b + 1, k][f[0]]; p = True
        out.append((D.C[b, k] / c0 - 1, t, p))
    return sorted(out, reverse=True)
def path(t, Y, cost):
    k = D.k[t]; e = first(Y); x = dec(Y) if Y < 2026 else last
    c = pd.Series(D.C[e:x + 1, k]).ffill().fillna(D.O[e, k]).values
    v = c / D.O[e, k] * (1 - cost); v[-1] *= (1 - cost); return v        # value per $1
def run(N, partial, cost):
    E = 1000.0; eq = {}; rows = []
    for Y in range(2016, 2027):
        e = first(Y); x = dec(Y) if Y < 2026 else last; r = ranked(Y, partial)[:N]
        P = np.mean([path(t, Y, cost) for _, t, _ in r], axis=0)               # equal $ per name, buy & hold
        for i, v in zip(range(e, x + 1), P): eq[cal[i]] = E * v
        g = P[-1]; rows.append(dict(year=Y if Y < 2026 else '2026 YTD', names=', '.join(t + ('*' if p else '') for _, t, p in r), ret=g - 1,
                                    best=max(path(t, Y, cost)[-1] for _, t, _ in r) - 1, worst=min(path(t, Y, cost)[-1] for _, t, _ in r) - 1,
                                    n_up=sum(path(t, Y, cost)[-1] > 1 for _, t, _ in r)))
        E *= g
    return pd.DataFrame(rows), pd.Series(eq)
spy = [D.spyC[(dec(Y) if Y < 2026 else last)] / D.spyO[first(Y)] - 1 for Y in range(2016, 2027)]
qqq = [q.Close.loc[cal[dec(Y) if Y < 2026 else last]] / q.Open.loc[cal[first(Y)]] - 1 for Y in range(2016, 2027)]
pf = lambda v: f'{100*v:+.1f}%'; yrs = (cal[last] - cal[first(2016)]).days / 365.25
def line(lab, r, eq=None, upto=None):
    r = np.asarray(r)[:upto]; end = 1000 * np.prod(1 + r); y = yrs if upto is None else upto
    d = {'Portfolio ($1,000 start)': lab, 'End value': f'${end:,.0f}', 'CAGR': pf((end / 1000) ** (1 / y) - 1), 'Worst year': pf(r.min()), 'Beat SPY': f'{(r > np.asarray(spy)[:len(r)]).sum()}/{len(r)}', 'Yearly vol': f'{100*r.std():.0f}%'}
    if eq is not None: d['Max DD (daily)'] = pf((eq / eq.cummax() - 1).min())
    return d
res = {(N, p, c): run(N, p, c) for N in (1, 10) for p in (True, False) for c in (0, .001)}
for lab, upto in (('2016-2026 YTD (to %s)' % cal[last].date(), None), ('2016-2025 (10 complete years)', 10)):
    rows = []
    for (N, p, c), (R, eq) in res.items():
        if c != .001 or (not p and N == 1 and False): continue
        eq_ = eq if upto is None else eq[eq.index <= cal[dec(2025)]]
        rows.append(line(f"{'Top-1 (all $1,000 in #1)' if N == 1 else 'Top-10 ($100 each)'}, reading {'A' if p else 'B'}", R.ret.values, eq_, upto))
    rows.append(line('SPY $1,000', spy, None, upto)); rows.append(line('QQQ $1,000', qqq, None, upto))
    print(f'\n### {lab} - 10 bps/side, buy Jan open, sell Dec close. Hypothetical backtest, not live results.'); print(pd.DataFrame(rows).fillna('').to_markdown(index=False))
R1, _ = res[(1, True, .001)]; R10, e10 = res[(10, True, .001)]; R10b, _ = res[(10, False, .001)]
cmp_ = pd.DataFrame({'year': R1.year, 'Top-1 pick': R1.names, 'Top-1 return': R1.ret.map(pf), 'Top-10 return': R10.ret.map(pf), 'Top-10 best/worst name': [f'{pf(b)} / {pf(w)}' for b, w in zip(R10.best, R10.worst)], 'names up': R10.n_up.astype(str) + '/10', 'SPY': [pf(v) for v in spy], 'QQQ': [pf(v) for v in qqq]})
print('\n### Year by year (reading A, 10 bps). Hypothetical backtest, not live results.'); print(cmp_.to_markdown(index=False))
print('\n### Value of $1,000 at each year-end'); v1 = 1000 * np.cumprod(1 + R1.ret.values); v10 = 1000 * np.cumprod(1 + R10.ret.values)
print(pd.DataFrame({'year': R1.year, 'Top-1': [f'${v:,.0f}' for v in v1], 'Top-10': [f'${v:,.0f}' for v in v10], 'SPY': [f'${v:,.0f}' for v in 1000 * np.cumprod(1 + np.array(spy))], 'QQQ': [f'${v:,.0f}' for v in 1000 * np.cumprod(1 + np.array(qqq))]}).to_markdown(index=False))
print('\nTop-10 holdings by year (* = partial prior-year history):'); [print(f'  {y}: {n}') for y, n in zip(R10.year, R10.names)]
for lab, R in (('Top-1', R1), ('Top-10', R10)):
    lr = np.log1p(R.ret.values); o = np.argsort(-lr)[:3]
    print(f'{lab}: best 3 years {[str(R.year[i]) + " " + pf(R.ret[i]) for i in o]} = {100*lr[o].sum()/lr.sum():.0f}% of log-growth; end without them ${1000*np.exp(np.delete(lr,o).sum()):,.0f}')
R10.assign(top1_ret=R1.ret).to_csv('outputs/top10_vs_top1_yearly_2016.csv', index=False)
