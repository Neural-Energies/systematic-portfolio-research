"""Simplest version. Each year: buy $1,000 of the S&P 500 symbol with the best return in the PRIOR calendar year at the first session's open;
sell all at the last session's close; report % return and $ profit for that year only. No compounding, no costs, no stop. 2026 = to latest close."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from engine import Data
D = Data(); cal = D.cal; last = D.T - 1
dec = lambda y: np.where(cal.year == y)[0][-1]; first = lambda y: np.where(cal.year == y)[0][0]
rows = []
for Y in range(2016, 2027):
    b, a, e = dec(Y - 1), dec(Y - 2), first(Y); x = dec(Y) if Y < 2026 else last; best = None
    for t in D.members(cal[b]):
        k = D.k.get(t)
        if k is None or not np.isfinite(D.C[b, k]) or not np.isfinite(D.O[e, k]): continue
        c0 = D.C[a, k]
        if not np.isfinite(c0):                                      # listed during the prior year: use first close of that year
            f = np.where(np.isfinite(D.C[a + 1:b + 1, k]))[0]
            if not len(f): continue
            c0 = D.C[a + 1:b + 1, k][f[0]]
        r = D.C[b, k] / c0 - 1
        if best is None or r > best[0]: best = (r, t, k)
    r0, t, k = best; ret = D.C[x, k] / D.O[e, k] - 1
    rows.append(dict(year=str(Y) if Y < 2026 else '2026 YTD', symbol=t, prior_year_return=r0, buy_date=str(cal[e].date()), sell_date=str(cal[x].date()), ret=ret, profit=1000 * ret,
                     spy_ret=D.spyC[x] / D.spyO[e] - 1))
R = pd.DataFrame(rows); R['spy_profit'] = 1000 * R.spy_ret; R['cum_profit'] = R.profit.cumsum()
R.to_csv('outputs/simple_profit_per_year.csv', index=False)
pf = lambda v: f'{100*v:+.1f}%'; usd = lambda v: f'{"-" if v<0 else "+"}${abs(v):,.0f}'
print(pd.DataFrame({'Year': R.year, 'Symbol (best prior-yr return)': R.symbol + ' (' + R.prior_year_return.map(pf) + ')', 'Buy (open)': R.buy_date, 'Sell (close)': R.sell_date,
                    'Return': R.ret.map(pf), 'Profit on $1,000': R.profit.map(usd), 'SPY profit on $1,000': R.spy_profit.map(usd), 'Running total profit': R.cum_profit.map(usd)}).to_markdown(index=False))
n = 10; c = R.iloc[:n]
print(f'\n2016-2025 ({n} years, ${1000*n:,} put in over time): total profit {usd(c.profit.sum())}; SPY {usd(c.spy_profit.sum())}; winning years {(c.profit>0).sum()}/{n}; years beating SPY {(c.ret>c.spy_ret).sum()}/{n}')
print(f'2016-2026 YTD (11 years, $11,000 put in over time): total profit {usd(R.profit.sum())}; SPY {usd(R.spy_profit.sum())}; winning years {(R.profit>0).sum()}/11; years beating SPY {(R.ret>R.spy_ret).sum()}/11')
print(f'Average return per year {pf(R.ret.mean())}, median {pf(R.ret.median())}; best {R.year[R.ret.idxmax()]} {pf(R.ret.max())}, worst {R.year[R.ret.idxmin()]} {pf(R.ret.min())}')
