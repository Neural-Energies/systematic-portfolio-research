"""Reconcile with the viral post's exhibits (picks + returns visible in the screenshots, 2012 -> 2026 YTD to Oct 9).
Post convention: rank on Dec31->Dec31 total return among year-end S&P members; hold Dec31 close -> next Dec31 close."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from engine import Data, HB
D = Data(); cal = D.cal
END = pd.Timestamp('2026-10-09'); i_end = np.searchsorted(cal.values, np.datetime64(END), side='right') - 1
POST = {2012: ('COG', .31), 2013: ('PHM', .13), 2014: ('NFLX', -.07), 2015: ('LUV', .02), 2016: ('NFLX', .08), 2017: ('NVDA', .82), 2018: ('NRG', .40), 2019: ('AMD', 1.48),
        2020: ('AMD', 1.00), 2021: ('TSLA', .50), 2022: ('DVN', .50), 2023: ('OXY', -.04), 2024: ('NVDA', 1.71), 2025: ('PLTR', 1.35), 2026: ('SNDK', 5.66)}   # read off the screenshots
dec = lambda y: np.where((cal.year == y))[0][-1]
first = lambda y: np.where((cal.year == y))[0][0]
def rank(Y, allow_partial):
    """Top stock for holding year Y (ranked on year Y-1)."""
    b = dec(Y - 1); a = dec(Y - 2); mem = D.members(cal[b]); out = []
    for t in mem:
        k = D.k.get(t)
        if k is None or not np.isfinite(D.C[b, k]): continue
        c0 = D.C[a, k]; partial = False
        if not np.isfinite(c0):
            if not allow_partial: continue
            col = D.C[a + 1:b + 1, k]; fin = np.where(np.isfinite(col))[0]
            if len(fin) == 0: continue
            c0 = col[fin[0]]; partial = True
        out.append((D.C[b, k] / c0 - 1, t, partial))
    out.sort(reverse=True); return out
def hold(t, Y, mode):
    k = D.k[t]; b = dec(Y - 1); e = first(Y); x = dec(Y) if Y < 2026 else i_end
    ep = D.C[b, k] if mode == 'close' else D.O[e, k]
    return D.C[x, k] / ep - 1
def spy(Y, mode):
    b = dec(Y - 1); e = first(Y); x = dec(Y) if Y < 2026 else i_end
    return D.spyC[x] / (D.spyC[b] if mode == 'close' else D.spyO[e]) - 1
rows = []
for Y in range(2012, 2027):
    top_p = rank(Y, True)[0]; top_f = rank(Y, False)[0]
    p_t, p_r = POST[Y]
    rows.append(dict(year=Y, post_pick=p_t, post_ret=p_r, my_top_partial_ok=top_p[1] + ('*' if top_p[2] else ''), my_top_full_year=top_f[1],
                     my_ret_post_pick_close=hold(p_t, Y, 'close') if p_t in D.k else np.nan, my_ret_post_pick_open=hold(p_t, Y, 'open') if p_t in D.k else np.nan,
                     spy_close=spy(Y, 'close'), spy_open=spy(Y, 'open')))
R = pd.DataFrame(rows)
pf = lambda x: f'{100*x:+.0f}%'
print(R.assign(post_ret=R.post_ret.map(pf), my_ret_post_pick_close=R.my_ret_post_pick_close.map(pf), my_ret_post_pick_open=R.my_ret_post_pick_open.map(pf), spy_close=R.spy_close.map(pf), spy_open=R.spy_open.map(pf)).to_markdown(index=False))
R.to_csv('outputs/reconcile_post.csv', index=False)
# chains (100 start). 2012 COG not in Yahoo => use post's +31% (unverifiable) in the 'post' chain
cum = lambda r: 100 * np.prod(1 + np.asarray(r))
post_chain = cum(R.post_ret); r_close = R.my_ret_post_pick_close.fillna(R.post_ret); r_open = R.my_ret_post_pick_open.fillna(R.post_ret)
print('\nPost\'s own reported yearly returns compounded: $%s (post says $176,114; $26,428 at end-2025: mine $%s)' % (f'{post_chain:,.0f}', f'{cum(R.post_ret[:-1]):,.0f}'))
print('Post picks, MY prices, Dec31-close entry (2012 = post +31%% as COG unavailable): $%s;  to end-2025 $%s' % (f'{cum(r_close):,.0f}', f'{cum(r_close[:-1]):,.0f}'))
print('Post picks, MY prices, Jan-open entry  (2012 = post +31%%):  $%s;  to end-2025 $%s' % (f'{cum(r_open):,.0f}', f'{cum(r_open[:-1]):,.0f}'))
print('SPY same window: close-entry $%s, open-entry $%s' % (f'{cum(R.spy_close):,.0f}', f'{cum(R.spy_open):,.0f}'))
# sensitivity: strict full-prior-year rule => 2026 pick is WDC instead of SNDK
w = hold(R.my_top_full_year.iloc[-1], 2026, 'close'); print('\n2026 pick under a full-prior-year rule:', R.my_top_full_year.iloc[-1], pf(w), '-> total', f'${cum(list(r_close[:-1]) + [w]):,.0f}', '(vs SNDK', f'${cum(r_close):,.0f})')
print('2026 SNDK Jan-open entry:', pf(hold('SNDK', 2026, 'open')), ' Dec-31 close entry:', pf(hold('SNDK', 2026, 'close')))
# without the SNDK year / without the best 3 years
lr = np.log1p(r_close.values); o = np.argsort(-lr)[:3]
print('Best 3 years (close conv):', [(int(R.year[i]), R.post_pick[i], pf(r_close[i])) for i in o], 'share of log growth %.0f%%' % (100 * lr[o].sum() / lr.sum()), '-> without them $%.0f' % (100 * np.exp(np.delete(lr, o).sum())))
# last 10 yrs (2017-2026) on the post's picks
for lab, r_ in (('close', r_close), ('open', r_open)):
    s = R.year >= 2017; print(f'Last 10 yrs 2017-2026 ({lab} entry, post picks): ${cum(r_[s]):,.0f}  SPY ${cum(R["spy_"+lab][s]):,.0f}; ex-2026 YTD (2017-2025): ${cum(r_[(R.year>=2017)&(R.year<=2025)]):,.0f}')
# sanity: how many of the post's picks are the top of MY data in the same year
print('\nMy top pick (partial history allowed) equals the post pick in', int((R.post_pick == R.my_top_partial_ok.str.rstrip('*')).sum()), 'of 15 years; mismatches:', R[R.post_pick != R.my_top_partial_ok.str.rstrip('*')][['year','post_pick','my_top_partial_ok']].values.tolist())
