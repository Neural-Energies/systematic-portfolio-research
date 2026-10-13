"""Simple version: each year, buy $100 of last calendar year's best-performing S&P 500 member at the first trading day's
open of January; sell at the last trading day's close of December. Names without Yahoo prices are skipped (next one down)."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, yfinance as yf
from engine import Data, HB
D = Data(); cal = D.cal
qqq = yf.download('QQQ', start='2014-12-01', auto_adjust=True, progress=False)
qqq.columns = [c[0] for c in qqq.columns]; qqq.index = pd.DatetimeIndex(qqq.index).tz_localize(None)
rows = []; top3 = []
for Y in range(2016, 2027):
    idx = np.where(cal.year == Y)[0]; e = idx[0]; x = idx[-1]            # first trading day, last trading day (latest data for 2026)
    prev = np.where(cal.year == Y - 1)[0]; a = np.where(cal.year == Y - 2)[0][-1]; b = prev[-1]   # prior-year Dec close -> Dec close
    mem = D.members(cal[b]); cand = []; missing = 0
    for t in mem:
        k = D.k.get(t)
        if k is None: missing += 1; continue
        c0, c1 = D.C[a, k], D.C[b, k]
        if np.isfinite(c0) and np.isfinite(c1) and np.isfinite(D.O[e, k]): cand.append((c1 / c0 - 1, t, k))
        else: missing += 1
    cand.sort(reverse=True)
    r_, t, k = cand[0]
    ex = D.C[x, k] if np.isfinite(D.C[x, k]) else pd.Series(D.C[e:x + 1, k]).dropna().iloc[-1]
    ret = ex / D.O[e, k] - 1
    spy = D.spyC[x] / D.spyO[e] - 1
    q = qqq.Close.iloc[qqq.index.get_indexer([cal[x]], method='pad')[0]] / qqq.Open.loc[cal[e]] - 1
    rows.append(dict(year=Y, pick=t, prior_year_return=r_, buy_open=D.O[e, k], sell_close=ex, ret=ret, spy=spy, qqq=q,
                     runner_ups=', '.join(f'{c[1]} ({c[0]:+.0%})' for c in cand[1:3]), members_without_prices=f'{missing}/{len(mem)}',
                     sold_on=str(cal[x].date())))
R = pd.DataFrame(rows); R.to_csv('outputs/simple_10y_picks.csv', index=False)
full = R[R.year <= 2025]
def comp(s): return 100 * np.prod(1 + s)
pf = lambda x: f'{100*x:+.1f}%'
out = R.assign(**{'$100 becomes': lambda d: (100 * (1 + d.ret)).map('${:,.0f}'.format), 'prior yr': lambda d: d.prior_year_return.map(pf), 'return': lambda d: d.ret.map(pf),
                  'SPY': lambda d: d.spy.map(pf), 'QQQ': lambda d: d.qqq.map(pf), 'buy $': lambda d: d.buy_open.map('{:.2f}'.format), 'sell $': lambda d: d.sell_close.map('{:.2f}'.format)})
out['year'] = out.year.astype(str).replace('2026', '2026 YTD')
print(out[['year', 'pick', 'prior yr', 'buy $', 'sell $', 'return', '$100 becomes', 'SPY', 'QQQ', 'runner_ups', 'members_without_prices']].to_markdown(index=False))
for lab, s in (('Strategy', full.ret), ('SPY', full.spy), ('QQQ', full.qqq)):
    print(f'{lab}: 2016-2025 compounded $100 -> ${comp(s):,.0f}; avg yr {pf(s.mean())}; worst {pf(s.min())}; beat-count n/a' if lab != 'Strategy' else
          f'{lab}: 2016-2025 compounded $100 -> ${comp(s):,.0f}; avg yr {pf(s.mean())}; worst {pf(s.min())}; beat SPY {(s.values > full.spy.values).sum()}/10; beat QQQ {(s.values > full.qqq.values).sum()}/10')
    print(f'   $100 invested fresh each year (10 x $100): profit ${100*s.sum():,.0f} on $1,000')
s = (1 + full.ret) * 0.999 ** 2 - 1; print('with 10bps/side costs: compounded $%s' % f'{comp(s):,.0f}')
lr = np.log1p(full.ret); top = lr.sort_values(ascending=False)[:3]
print('best 3 yrs', [(int(full.year[i]), pf(full.ret[i])) for i in top.index], 'share of log-growth %.0f%%' % (100 * top.sum() / lr.sum()), 'without them: $%.0f' % (100 * np.exp(lr.drop(top.index).sum())))
y26 = R[R.year == 2026].iloc[0]; print('2026 YTD to', y26.sold_on, pf(y26.ret))
