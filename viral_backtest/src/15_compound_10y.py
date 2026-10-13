"""Compounded 10-year version of the simple rule: $1,000 at the first session's open of the first year; each year the whole balance goes into the prior
year's best S&P 500 stock (open of first session -> close of last session), then everything is reinvested. No costs, no stop.
Two windows: 2016-2025 (10 complete years) and 2017-2026 YTD (last 10 calendar years, 2026 open to latest close)."""
import numpy as np, pandas as pd
R = pd.read_csv('outputs/simple_profit_bull_markets.csv'); R['y'] = R.year.str[:4].astype(int)
usd = lambda v: f'${v:,.0f}'; pf = lambda v: f'{100*v:+.1f}%'
def show(lab, y0, y1, wdc=False):
    d = R[(R.y >= y0) & (R.y <= y1)].reset_index(drop=True); r = d.ret.copy()
    if wdc and y1 == 2026: r.iloc[-1] = 1.108                               # strict full-prior-year rule: 2026 pick = WDC (+110.8% YTD)
    bal = 1000 * np.cumprod(1 + r); st = np.r_[1000, bal[:-1].values]; spy = 1000 * np.cumprod(1 + d.spy_ret); mdd = (bal / np.maximum.accumulate(np.r_[1000, bal.values])[1:] - 1).min()
    t = pd.DataFrame({'Year': d.year, 'Stock': d.symbol if not wdc else d.symbol.where(d.y < 2026, 'WDC'), 'Start': [usd(v) for v in st], 'Return': r.map(pf), 'Profit that year': [f'{"-" if b<s else "+"}${abs(b-s):,.0f}' for s, b in zip(st, bal)],
                      'End balance': [usd(v) for v in bal], 'SPY end balance': [usd(v) for v in spy]})
    print(f'\n### {lab}. Hypothetical backtest, not live results.'); print(t.to_markdown(index=False))
    yrs = len(d) if y1 < 2026 else len(d) - 1 + (pd.Timestamp('2026-10-13') - pd.Timestamp('2026-01-02')).days / 365.25
    print(f'$1,000 -> {usd(bal.iloc[-1])} (profit {usd(bal.iloc[-1]-1000)}, {bal.iloc[-1]/1000:.1f}x; CAGR {pf((bal.iloc[-1]/1000)**(1/yrs)-1)}) | SPY $1,000 -> {usd(spy.iloc[-1])} (CAGR {pf((spy.iloc[-1]/1000)**(1/yrs)-1)}) | up years {(r>0).sum()}/{len(r)}, beat SPY {(r>d.spy_ret).sum()}/{len(r)} | worst year {pf(r.min())} | worst year-end drawdown {pf(mdd)}')
    lr = np.log1p(r); o = np.argsort(-lr.values)[:3]; print(f'Best 3 years {[(d.year[i], pf(r[i])) for i in o]} = {100*lr.values[o].sum()/lr.sum():.0f}% of log-growth; without them {usd(1000*np.exp(np.delete(lr.values,o).sum()))}')
    return bal.iloc[-1]
show('Window 1: 2016-2025 (10 complete years)', 2016, 2025)
show('Window 2: 2017-2026 YTD (last 10 calendar years)', 2017, 2026)
show('Window 2, strict full-prior-year rule (2026 = WDC instead of SNDK)', 2017, 2026, wdc=True)
