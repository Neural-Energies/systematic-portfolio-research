"""2016 -> present. Buy $100 of last calendar year's best S&P 500 member at the first open of January, sell at the last December close
(2026: latest close). Two readings of 'best': (A) partial-history allowed (what the viral post did: SNDK for 2026), (B) full prior year required."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, yfinance as yf
from engine import Data
D = Data(); cal = D.cal; last = D.T - 1
try:
    q = yf.download('QQQ', start='2015-12-01', auto_adjust=True, progress=False); q.columns = [c[0] for c in q.columns]; q.index = pd.DatetimeIndex(q.index).tz_localize(None)
except Exception: q = None
dec = lambda y: np.where(cal.year == y)[0][-1]; first = lambda y: np.where(cal.year == y)[0][0]
def top(Y, partial):
    b = dec(Y - 1); a = dec(Y - 2); out = []
    for t in D.members(cal[b]):
        k = D.k.get(t)
        if k is None or not np.isfinite(D.C[b, k]) or not np.isfinite(D.O[first(Y), k]): continue
        c0 = D.C[a, k]; p = False
        if not np.isfinite(c0):
            if not partial: continue
            col = D.C[a + 1:b + 1, k]; f = np.where(np.isfinite(col))[0]
            if not len(f): continue
            c0 = col[f[0]]; p = True
        out.append((D.C[b, k] / c0 - 1, t, p))
    return max(out)
rows = []
for Y in range(2016, 2027):
    e = first(Y); x = dec(Y) if Y < 2026 else last
    rA = top(Y, True); rB = top(Y, False)
    g = lambda t: D.C[x, D.k[t]] / D.O[e, D.k[t]] - 1
    qq = q.Close.iloc[q.index.get_indexer([cal[x]], method='pad')[0]] / q.Open.loc[cal[e]] - 1 if q is not None else np.nan
    rows.append(dict(year=Y if Y < 2026 else '2026 YTD', pick_A=rA[1] + ('*' if rA[2] else ''), prior=rA[0], ret_A=g(rA[1]), pick_B=rB[1], ret_B=g(rB[1]), spy=D.spyC[x] / D.spyO[e] - 1, qqq=qq))
R = pd.DataFrame(rows); R.to_csv('outputs/since_2016.csv', index=False)
pf = lambda v: f'{100*v:+.1f}%'
print(R.assign(prior=R.prior.map(pf), ret_A=R.ret_A.map(pf), ret_B=R.ret_B.map(pf), spy=R.spy.map(pf), qqq=R.qqq.map(pf)).to_markdown(index=False))
cum = lambda r: 100 * np.prod(1 + np.asarray(r))
c = lambda r: 100 * np.prod((1 + np.asarray(r)) * .999 ** 2)
print('\n(*) = partial prior-year history (listed mid-year)')
for lab, sl in (('2016-2025 (10 complete years)', slice(0, 10)), ('2016-2026 YTD (to %s)' % cal[last].date(), slice(0, 11))):
    print(f'{lab}: A ${cum(R.ret_A[sl]):,.0f} (10bps ${c(R.ret_A[sl]):,.0f}) | B ${cum(R.ret_B[sl]):,.0f} | SPY ${cum(R.spy[sl]):,.0f}' + (f' | QQQ ${cum(R.qqq[sl]):,.0f}' if q is not None else ''))
f = R.iloc[:10]; print('Beat SPY (complete yrs): %d/10;  beat QQQ: %d/10' % ((f.ret_A > f.spy).sum(), (f.ret_A > f.qqq).sum()))
lr = np.log1p(R.ret_A.values); o = np.argsort(-lr)[:3]
print('Best 3 yrs (A):', [(str(R.year[i]), pf(R.ret_A[i])) for i in o], 'share of log-growth %.0f%%' % (100 * lr[o].sum() / lr.sum()), '-> without them $%.0f' % (100 * np.exp(np.delete(lr, o).sum())))
print('Worst year A:', pf(f.ret_A.min()), '| vol of yearly returns: %.0f%%' % (100 * f.ret_A.std()))
