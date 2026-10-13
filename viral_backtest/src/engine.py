"""Backtest engine for the 'buy last year's best S&P 500 stock' claim.

Conventions (fixed up-front, never tuned):
  * Membership: point-in-time list from fja05680/sp500 (Updated file), members as of the signal date.
  * Prices: Yahoo auto-adjusted (split + dividend-reinvested) open/close.
  * Signal: close on the last trading day of the month before the start month; entry at the NEXT trading day's open
    (variant: that day's close).  Equal weight across the top-N names, buy-and-hold within the period, no intra-period
    rebalance.  Positions are sold at the next period's entry price.
  * Stop: if a daily CLOSE is <= entry*(1-s), exit at the NEXT trading day's open (gap goes through the stop), then cash
    (0% yield) until the next entry date.
  * A stock whose price series ends mid-period (delisting / acquisition) is sold at its last close, then cash.
  * Costs: `cost` per side, applied to every buy and sell.
"""
import numpy as np, pandas as pd, pickle
from collections import OrderedDict

HB = "Hypothetical backtest, not live results"


class Data:
    def __init__(self, pkl='data/prices.pkl', hist='data/hist_updated.csv'):
        px = pickle.load(open(pkl, 'rb'))
        spy = px['SPY']
        self.cal = spy.index                                  # trading calendar = SPY dates
        self.cal = pd.DatetimeIndex(self.cal).tz_localize(None) if self.cal.tz is not None else pd.DatetimeIndex(self.cal)
        names = [t for t in px if t != 'SPY']
        O = pd.DataFrame({t: px[t]['Open'] for t in names}); C = pd.DataFrame({t: px[t]['Close'] for t in names})
        for df in (O, C):
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        O = O.reindex(self.cal); C = C.reindex(self.cal)
        bad = (C <= 0) | (O <= 0)                              # corrupt prints only; nothing performance-related
        O = O.mask(bad | C.isna()); C = C.mask(bad)
        self.names = list(C.columns); self.k = {t: i for i, t in enumerate(self.names)}
        self.O = O.values; self.C = C.values
        self.Cff = C.ffill(limit=5).values                    # for ranking only (tolerate short gaps)
        self.spyO = spy['Open'].reindex(self.cal).values; self.spyC = spy['Close'].reindex(self.cal).values
        self.T = len(self.cal)
        # Membership: the ORIGINAL file (to 2019-01-11) tags every company that later left the index with a '-YYYYMM'
        # suffix, which makes ticker identity unambiguous (Yahoo re-uses tickers, e.g. SNT, AAL).  Suffixed tokens
        # never match a Yahoo ticker => those members are 'unavailable' (counted, never silently dropped).
        # After 2019-01-11 we use the Updated file (bare tickers).
        import re
        h0 = pd.read_csv('data/hist.csv', parse_dates=['date']); h1 = pd.read_csv(hist, parse_dates=['date'])
        self.cut = h0.date.max()
        h1 = h1[h1.date > self.cut]
        fix = lambda s_: set(x if re.search(r'-\d{6}$', x) else x.replace('.', '-') for x in s_.split(','))
        self.hdates = np.concatenate([h0.date.values, h1.date.values])
        self.hmem = [fix(s_) for s_ in h0.tickers] + [fix(s_) for s_ in h1.tickers]
        # month-end trading-day indices
        s = pd.Series(np.arange(self.T), index=self.cal)
        self.me = s.groupby([self.cal.year, self.cal.month]).last()
        self.me_keys = list(self.me.index); self.me_idx = self.me.values

    def members(self, d):
        i = np.searchsorted(self.hdates, np.datetime64(d), side='right') - 1
        return self.hmem[i]

    def month_pos(self, y, m):
        return self.me_keys.index((y, m))


def periods(D, start_month, start_year, last_idx=None):
    """List of (signal_idx, entry_idx, exit_idx, is_final)."""
    out = []
    last = D.T - 1 if last_idx is None else last_idx
    ents = []
    y = start_year
    while True:
        sm = start_month - 1; sy = y
        if sm == 0: sm = 12; sy = y - 1
        if (sy, sm) not in D.me_keys: break
        sig = D.me[(sy, sm)]
        e = sig + 1
        if e > last: break
        ents.append((sig, e)); y += 1
    res = []
    for i, (sig, e) in enumerate(ents):
        if i + 1 < len(ents): res.append((sig, e, ents[i + 1][1], False))
        else: res.append((sig, e, last, True))
    return res


def rank_candidates(D, sig, lookback, skip, topn):
    """Return list of (ticker_idx, lookback_return) top-N among PIT members, ranked on data up to the signal month-end."""
    mk = D.me_keys; pos = None
    sd = D.cal[sig]; pos = mk.index((sd.year, sd.month))
    end_i = D.me_idx[pos - skip]; st_i = D.me_idx[pos - lookback]
    mem = D.members(sd); cand = []
    for t in mem:
        k = D.k.get(t)
        if k is None: continue
        c0 = D.Cff[st_i, k]; c1 = D.Cff[end_i, k]
        if not (np.isfinite(c0) and np.isfinite(c1)): continue
        # price series must be alive at the signal date (not already delisted) and tradable at next open
        if not np.isfinite(D.Cff[sig, k]) or not np.isfinite(D.O[sig + 1, k]) or not np.isfinite(D.C[sig + 1, k]): continue
        cand.append((k, c1 / c0 - 1))
    cand.sort(key=lambda z: (-z[1], D.names[z[0]]))
    return cand[:topn], len(cand), len(mem)


def sim_position(D, k, e, x, final, stop, entry, cost):
    """Simulate one position over a period. Returns (daily values for days e..last_day, growth, stopped).
    Daily value = position value per $1 allocated at entry, net of the buy cost; after an exit it is the realised
    proceeds net of both sides' costs (cash earns 0%). Non-final periods cover days e..x-1; the position (or cash) is
    realised at the open of day x (close for the close-entry variant) when the next period starts."""
    O, C = D.O[:, k], D.C[:, k]
    ep = O[e] if entry == 'open' else C[e]
    last_day = x if final else x - 1
    n = last_day - e + 1
    vals = np.empty(n); lastc = ep
    exit_idx = None; exit_px = None; stopped = False
    lvl = ep * (1 - stop) if stop is not None else -1.0
    for t in range(e, last_day + 1):
        if exit_idx is not None and t >= exit_idx:
            vals[t - e] = exit_px / ep * (1 - cost) ** 2; continue
        c = C[t]
        if not np.isfinite(c):
            if not np.isfinite(C[t:x + 1]).any():      # series is over -> delisted / acquired: sold at last close
                exit_idx = t; exit_px = lastc
                vals[t - e] = exit_px / ep * (1 - cost) ** 2; continue
            c = lastc
        else:
            lastc = c
        vals[t - e] = c / ep * (1 - cost)
        armed = (entry == 'open') or t > e
        if stop is not None and armed and c <= lvl and t < last_day:
            o = O[t + 1]
            exit_idx = t + 1; exit_px = o if np.isfinite(o) else c      # next open: gaps go through the stop
            stopped = (t + 1 < x) or final
    if exit_idx is None:
        if final: exit_px = lastc
        else:
            o = O[x] if entry == 'open' else C[x]
            exit_px = o if np.isfinite(o) else lastc
    growth = exit_px / ep * (1 - cost) ** 2
    if final and exit_idx is None: vals[-1] = growth
    return vals, growth, stopped


def run(D, topn=1, stop=None, entry='open', cost=0.001, start_month=1, start_year=1997, lookback=12, skip=0, last_idx=None):
    """Run one configuration; returns dict with daily equity (indexed by calendar idx), period table."""
    pers = periods(D, start_month, start_year, last_idx)
    first_e = pers[0][1]
    eq = np.full(D.T, np.nan); E = 1.0; rows = []
    for (sig, e, x, final) in pers:
        picks, ncand, nmem = rank_candidates(D, sig, lookback, skip, topn)
        end_day = x if final else x - 1
        if not picks:
            eq[e:end_day + 1] = E; rows.append(dict(sig=sig, entry=e, exit=x, E0=E, picks='', ret=0.0, stops=0, ncand=ncand, nmem=nmem, final=final)); continue
        V = []; G = []; S = 0
        for (k, r) in picks:
            v, g, st = sim_position(D, k, e, x, final, stop, entry, cost)
            V.append(v); G.append(g); S += st
        Vm = np.mean(V, axis=0); eq[e:e + len(Vm)] = E * Vm
        rows.append(dict(sig=sig, entry=e, exit=x, E0=E, picks=','.join(D.names[k] for k, _ in picks), lbret=','.join(f'{r:.3f}' for _, r in picks),
                         ret=float(np.mean(G) - 1), stops=S, ncand=ncand, nmem=nmem, final=final, gross=float(np.mean(G))))
        E = E * np.mean(G) if not final else E
    eq[first_e - 1] = 1.0                                   # base value at the close before the first entry
    ser = pd.Series(eq, index=D.cal).iloc[first_e - 1:].ffill()
    return dict(eq=ser, per=pd.DataFrame(rows), first=first_e, cfg=dict(topn=topn, stop=stop, entry=entry, cost=cost, sm=start_month, sy=start_year, lb=lookback, skip=skip))


def spy_bench(D, entry, first_e, last_idx=None):
    last = D.T - 1 if last_idx is None else last_idx
    base = D.spyO[first_e] if entry == 'open' else D.spyC[first_e]
    return pd.Series(D.spyC[first_e:last + 1] / base, index=D.cal[first_e:last + 1])


# ---------------------------------------------------------------- statistics
def stats(eq: pd.Series, per_ret=None, spy_per=None):
    """Daily-equity based stats. Start value is 1.0 at the first valid date (equity includes day-1 intraday move)."""
    e = eq.dropna(); r = e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / 1.0) ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    sortino = r.mean() * 252 / dd if dd > 0 else np.nan
    mdd = (e / e.cummax() - 1).min()
    out = dict(end=100 * e.iloc[-1], cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino, mdd=mdd)
    if per_ret is not None:
        out['worst_yr'] = per_ret.min(); out['n_yrs'] = len(per_ret)
        if spy_per is not None: out['beat'] = float((per_ret.values > spy_per.values).mean()); out['beat_n'] = int((per_ret.values > spy_per.values).sum())
    return out


def spy_period_returns(D, per, entry):
    arr = D.spyO if entry == 'open' else D.spyC
    return pd.Series([arr[x] / arr[e] - 1 for e, x in zip(per.entry, per.exit)], index=[D.cal[e].year for e in per.entry])


def sub_equity(eq, start_ts, end_ts=None, E0=None):
    """Sub-sample starting at entry date start_ts, rebased to 1.0 at the entry OPEN (E0 = equity at that period's start),
    so the prior holding's overnight gap is not inherited. Base point is stamped on the prior trading day."""
    prev = eq[eq.index < start_ts]
    base = E0 if E0 is not None else prev.iloc[-1]
    e = eq[eq.index >= start_ts]
    if end_ts is not None: e = e[e.index <= end_ts]
    return pd.concat([pd.Series([1.0], index=[prev.index[-1]]), e / base])
