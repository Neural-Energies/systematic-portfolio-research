"""Compounded version of the simple rule from 2022: start with $1,000 on the first session of 2022; each year buy the best-prior-year stock with the WHOLE balance
at the first session's open, sell all at the last session's close, reinvest everything. No costs, no stop. 2026 = to latest close."""
import numpy as np, pandas as pd
R = pd.read_csv('outputs/simple_profit_bull_markets.csv'); R['y'] = R.year.str[:4].astype(int); d = R[R.y >= 2022].reset_index(drop=True)
usd = lambda v: f'${v:,.0f}'; pf = lambda v: f'{100*v:+.1f}%'
bal = 1000 * np.cumprod(1 + d.ret); start = np.r_[1000, bal[:-1]]; spy = 1000 * np.cumprod(1 + d.spy_ret); spy_start = np.r_[1000, spy[:-1]]
t = pd.DataFrame({'Year': d.year, 'Stock': d.symbol, 'Start balance': start.map(usd) if hasattr(start, 'map') else [usd(v) for v in start], 'Return': d.ret.map(pf),
                  'Profit that year': [f'{"-" if b<s else "+"}${abs(b-s):,.0f}' for s, b in zip(start, bal)], 'End balance': [usd(v) for v in bal], 'SPY end balance': [usd(v) for v in spy]})
print(t.to_markdown(index=False))
n4 = d.y <= 2025
print(f'\n2022-2025 (4 yrs): $1,000 -> {usd(bal[n4].iloc[-1])}  (profit {usd(bal[n4].iloc[-1]-1000)}; SPY {usd(spy[n4].iloc[-1])})')
print(f'2022-2026 YTD (5 yrs): $1,000 -> {usd(bal.iloc[-1])}  (profit {usd(bal.iloc[-1]-1000)}; SPY {usd(spy.iloc[-1])})')
w = 1000 * np.prod(1 + d.ret[n4]) * 2.108; print(f'Strict full-prior-year rule (2026 = WDC +110.8%): {usd(w)}')
lr = np.log1p(d.ret); print(f'Share of total log-growth from 2026 SNDK: {100*lr.iloc[-1]/lr.sum():.0f}%; without the 2026 year, 2022-2025 ends at {usd(bal[n4].iloc[-1])}')
t.to_csv('outputs/compound_from_2022.csv', index=False)
