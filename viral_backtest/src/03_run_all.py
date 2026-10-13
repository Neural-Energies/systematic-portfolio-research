"""Runs every test. Nothing is tuned: all configurations are fixed up-front and every one is registered as a trial."""
import sys, json, io, zipfile, math, warnings, urllib.request
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats as st
from engine import *

D = Data()
OUT = 'outputs'; import os; os.makedirs(OUT, exist_ok=True)
STOPS = [None, .15, .20, .25, .30, .35, .40, .45, .50]; RANKS = [1, 3, 5, 10]
sname = lambda s: 'none' if s is None else f'{int(round(s*100))}%'
R = {}                                          # registry of ALL runs (== trials)
def go(topn=1, stop=None, entry='open', cost=.001, sm=1, lb=12, skip=0):
    key = (topn, stop, entry, cost, sm, lb, skip)
    if key not in R: R[key] = run(D, topn, stop, entry, cost, sm, 1997, lb, skip)
    return R[key]

# ---- 1. core grid (rank x stop x entry x cost)
for n in RANKS:
    for s in STOPS:
        for en in ('open', 'close'):
            for c in (.001, .005): go(n, s, en, c)
# ---- 6. robustness runs
for m in range(1, 13):
    for s in (None, .35): go(1, s, sm=m)
for (lb, sk) in ((6, 0), (12, 1)):
    for s in (None, .35): go(1, s, lb=lb, skip=sk)
NTRIALS = len(R); print('total trials registered:', NTRIALS, flush=True)

LAST = D.T - 1
def sample_start(r, y):
    """entry idx of first period with entry year >= y"""
    p = r['per']; p = p[[D.cal[e].year >= y for e in p.entry]]; return p.entry.iloc[0]

def summarize(r, y, entry='open'):
    """Stats for sample starting in entry-year y. Annual stats use COMPLETE periods only; 'end_latest' includes the open YTD period."""
    p = r['per']; p = p[[D.cal[e].year >= y for e in p.entry]].copy()
    full = p[~p.final]
    e0 = D.cal[p.entry.iloc[0]]
    eq = sub_equity(r['eq'], e0, E0=p.E0.iloc[0])
    last_full_end = D.cal[full.exit.iloc[-1] - 1]
    eq_cy = eq[eq.index <= last_full_end]
    spy_full = spy_period_returns(D, full, entry)
    s = stats(eq_cy, full.ret, spy_full)
    s['end_latest'] = 100 * eq.iloc[-1]; s['stops'] = int(p.stops.sum()); s['n_with_stop_years'] = int((p.stops > 0).sum())
    s['start'] = str(e0.date()); s['yearly'] = full.ret.values; s['years'] = [D.cal[e].year for e in full.entry]
    s['spy_yearly'] = spy_full.values
    s['eq'] = eq; s['eq_cy'] = eq_cy; s['per'] = p
    return s

def spy_summary(y, entry='open'):
    r = go(1, None, entry); p = r['per']; p = p[[D.cal[e].year >= y for e in p.entry]]
    e0 = p.entry.iloc[0]; full = p[~p.final]
    arr = D.spyC; base = D.spyO[e0] if entry == 'open' else D.spyC[e0]
    idx = D.cal[e0 - 1:LAST + 1]; v = np.concatenate([[1.0], arr[e0:LAST + 1] / base]); eq = pd.Series(v, index=idx)
    lfe = D.cal[full.exit.iloc[-1] - 1]
    sp = spy_period_returns(D, full, entry)
    s = stats(eq[eq.index <= lfe], sp, sp); s['end_latest'] = 100 * eq.iloc[-1]; s['eq'] = eq; s['yearly'] = sp.values; s['years'] = list(sp.index)
    return s

pct = lambda x: f'{100*x:.1f}%'
def fmt_row(label, s, extra=None):
    d = {'Variant': label, 'End $ (to 2026-10-13)': f"${s['end_latest']:,.0f}", 'End $ (to 12/2025)': f"${s['end']:,.0f}", 'CAGR': pct(s['cagr']), 'Vol': pct(s['vol']),
         'Sharpe': f"{s['sharpe']:.2f}", 'Sortino': f"{s['sortino']:.2f}", 'Max DD': pct(s['mdd'])}
    if 'worst_yr' in s: d['Worst yr'] = pct(s['worst_yr'])
    if 'beat_n' in s: d['Beat SPY'] = f"{s['beat_n']}/{s['n_yrs']}"
    if 'stops' in s: d['Stops hit'] = s['stops']
    if extra: d.update(extra)
    return d
def md(df): return df.to_markdown(index=False).replace(' nan ', '     ')
TAGLINE = f'*{HB}.*'
md_out = []
def section(title, df, note=''):
    md_out.append(f'\n### {title}\n\n**{HB}.** {note}\n\n' + md(df) + f'\n\n*{HB}.*\n')

# ---- 1. replication
rows = []; claim = {(2012, None): '$176,114 / 11 of 15 yrs', (1997, None): '$89,418 / -97% DD', (1997, .35): '$1,514,581 / -59% DD / 4 stops', (2012, .35): '(no claim)'}
for y in (2012, 1997):
    sp = spy_summary(y); rows.append(fmt_row(f'S&P 500 (SPY) from {y}', sp))
    for s in (None, .35):
        for (en, c) in (('open', .001), ('open', .005), ('close', .001)):
            r = summarize(go(1, s, en, c), y, en)
            rows.append(fmt_row(f'Top-1, stop {sname(s)}, {en}-entry, {int(c*1e4)}bps, from {y}', r, {'Viral claim': claim[(y, s)] if (en, c) == ('open', .001) else ''}))
T1 = pd.DataFrame(rows).fillna(''); section('Test 1 - Replication of the four viral variants (top-1, 100% in one stock)', T1,
    'Costs are per side. Last row-group entries marked (to 12/2025) use complete calendar years only.')

# ---- 2. rank sensitivity
rows = []
for y in (2012, 1997):
    for s in (None, .35):
        for n in RANKS:
            r = summarize(go(n, s), y); rows.append(fmt_row(f'{"Top-"+str(n) if n==1 else "EW top-"+str(n)}, stop {sname(s)}, from {y}', r))
T2 = pd.DataFrame(rows).fillna(''); section('Test 2 - Rank sensitivity (open-entry, 10bps/side)', T2)

# ---- 3. stop sweep (whole grid)
grid = []
for y in (2012, 1997):
    for n in RANKS:
        for s in STOPS:
            for c in (.001, .005):
                r = summarize(go(n, s, 'open', c), y)
                grid.append(dict(sample=y, topn=n, stop=sname(s), cost_bps=int(c * 1e4), end_latest=r['end_latest'], end_to_2025=r['end'], cagr=r['cagr'], vol=r['vol'],
                                 sharpe=r['sharpe'], sortino=r['sortino'], mdd=r['mdd'], worst_yr=r['worst_yr'], beat_n=r['beat_n'], n_yrs=r['n_yrs'], stops=r['stops']))
G = pd.DataFrame(grid); G.to_csv(f'{OUT}/stop_sweep_full_grid.csv', index=False)
for y in (2012, 1997):
    g = G[(G['sample'] == y) & (G.cost_bps == 10)]
    for n in RANKS:
        t = g[g.topn == n]
        df = pd.DataFrame({'Stop': t.stop, 'End $ (to 2026-10-13)': t.end_latest.map(lambda v: f'${v:,.0f}'), 'CAGR': t.cagr.map(pct), 'Sharpe': t.sharpe.map('{:.2f}'.format),
                           'Max DD': t.mdd.map(pct), 'Worst yr': t.worst_yr.map(pct), 'Beat SPY': t.beat_n.astype(str) + '/' + t.n_yrs.astype(str), 'Stops hit': t.stops})
        section(f'Test 3 - Stop sweep, {"top-1" if n==1 else "EW top-"+str(n)}, from {y} (10bps/side; all stops shown, none selected)', df)

# ---- 4. statistics: DSR + bootstrap
def monthly(eq):
    m = eq.resample('ME').last().pct_change().dropna(); return m
def sr_stats(m):
    return m.mean() / m.std(), st.skew(m), st.kurtosis(m, fisher=False), len(m)
def pool_srs(y):
    out = []
    for key, r in R.items():
        e = sub_equity(r['eq'], D.cal[sample_start(r, y)], E0=r['per'].set_index('entry').E0[sample_start(r, y)]); m = monthly(e)
        if len(m) > 12 and m.std() > 0: out.append(m.mean() / m.std())
    return np.array(out)
def dsr(m, pool, N):
    sr, sk, ku, T = sr_stats(m); V = np.var(pool, ddof=1); g = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - g) * st.norm.ppf(1 - 1 / N) + g * st.norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return dict(sr_m=sr, sr_ann=sr * math.sqrt(12), sr0_ann=sr0 * math.sqrt(12), dsr=st.norm.cdf(z), psr0=st.norm.cdf(sr * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)), T=T)
rng = np.random.default_rng(20260101)
def boot(yearly, start=100, B=10000):
    n = len(yearly); idx = rng.integers(0, n, (B, n)); ends = start * np.prod(1 + np.asarray(yearly)[idx], axis=1)
    return np.percentile(ends, [5, 50, 95]), (ends < start).mean()
rows = []; N_EFF = len(RANKS) * len(STOPS)
for y in (2012, 1997):
    pool = pool_srs(y)
    sp = spy_summary(y); b = boot(sp['yearly']); rows.append({'Variant': f'SPY from {y}', 'Sharpe': f"{sp['sharpe']:.2f}", 'Sortino': f"{sp['sortino']:.2f}", 'DSR': '-', 'Bootstrap 5th / median / 95th ($100 start)': ' / '.join(f'${v:,.0f}' for v in b[0]), 'P(end<$100)': pct(b[1])})
    for n in (1, 3, 5, 10):
        for s in (None, .35):
            r = summarize(go(n, s), y); m = monthly(r['eq_cy']); d = dsr(m, pool, NTRIALS); d2 = dsr(m, pool, N_EFF); b = boot(r['yearly'])
            rows.append({'Variant': f'{"Top-1" if n==1 else "EW top-"+str(n)}, stop {sname(s)}, from {y}', 'CAGR': pct(r['cagr']), 'Vol': pct(r['vol']), 'Sharpe': f"{r['sharpe']:.2f}", 'Sortino': f"{r['sortino']:.2f}",
                         'Max DD': pct(r['mdd']), 'Worst yr': pct(r['worst_yr']), 'Beat SPY': f"{r['beat_n']}/{r['n_yrs']}",
                         f'DSR (N={NTRIALS})': f"{d['dsr']:.2f}", f'DSR (N={N_EFF})': f"{d2['dsr']:.2f}", 'PSR(SR>0)': f"{d['psr0']:.2f}",
                         'Bootstrap 5th / median / 95th ($100 start)': ' / '.join(f'${v:,.0f}' for v in b[0]), 'P(end<$100)': pct(b[1])})
T4 = pd.DataFrame(rows).fillna('')
section('Test 4 - Full statistics, Deflated Sharpe, bootstrap (10,000 resamples of complete yearly returns)', T4,
        f'DSR uses monthly returns; trial pool = all {NTRIALS} registered configurations (N={N_EFF} = rank x stop grid only). DSR is the probability that the true Sharpe exceeds the best-of-N luck benchmark. Sharpe/Sortino use rf=0 and daily returns.')

# ---- 5. concentration
rows = []
for y in (2012, 1997):
    items = [('SPY', spy_summary(y))] + [(f'Top-1, stop {sname(s)}', summarize(go(1, s), y)) for s in (None, .35)]
    for lab, r in items:
        lr = np.log1p(np.asarray(r['yearly'])); tot = lr.sum(); order = np.argsort(-lr); top3 = order[:3]
        rest = np.delete(lr, top3); yrs = np.array(r['years'])
        rows.append({'Variant': f'{lab}, from {y}', 'Complete yrs': len(lr), 'Best 3 yrs': ', '.join(f'{yrs[i]} ({100*np.expm1(lr[i]):+.0f}%)' for i in top3),
                     '% of total log-growth from best 3': pct(lr[top3].sum() / tot) if tot > 0 else 'n/a (total log-growth <= 0)',
                     'End $100 as reported': f'${100*np.exp(tot):,.0f}', 'End $100 w/o best 3': f'${100*np.exp(rest.sum()):,.0f}', 'CAGR w/o best 3': pct(np.exp(rest.mean()) - 1)})
T5 = pd.DataFrame(rows); section('Test 5 - Concentration of log-growth in the best 3 years', T5, 'Annual figures are entry-to-entry complete years; the open 2026 YTD period is excluded.')

# ---- 6. robustness
rows = []
sp_full = spy_summary(1997)
for m in range(1, 13):
    for s in (None, .35):
        r = summarize(go(1, s, sm=m), 1997); rows.append(fmt_row(f'Start month {m:02d}, stop {sname(s)}, 12M lookback', r))
for (lb, sk, nm) in ((12, 0, '12M lookback'), (6, 0, '6M lookback'), (12, 1, '12M lookback skipping latest month (12-1)')):
    for s in (None, .35):
        r = summarize(go(1, s, lb=lb, skip=sk), 1997); rows.append(fmt_row(f'{nm}, stop {sname(s)} (January starts)', r))
T6 = pd.DataFrame(rows); section('Test 6 - Robustness: start month and lookback definition (top-1, 1997 start, open-entry, 10bps)', T6,
    f'Start-month rows beat-count compares to SPY over each strategy\'s own entry-to-entry periods. For reference SPY from 1997 ended at ${sp_full["end_latest"]:,.0f}.')
sm_tbl = pd.DataFrame([{**dict(start_month=m, stop=sname(s)), **{k: summarize(go(1, s, sm=m), 1997)[k] for k in ('end_latest', 'cagr', 'mdd', 'sharpe')}} for m in range(1, 13) for s in (None, .35)])
sm_tbl.to_csv(f'{OUT}/start_month_robustness.csv', index=False)

# ---- 7. out-of-sample
IS_END_Y = 2011
rows = []; picks = {}
for s in STOPS:
    r = go(1, s)
    p = r['per']; e12 = p.entry[[D.cal[e].year >= 2012 for e in p.entry]].iloc[0]
    eq_is = sub_equity(r['eq'], D.cal[p.entry.iloc[0]], D.cal[e12 - 1], E0=1.0)
    ret = eq_is.pct_change().dropna(); shp = ret.mean() / ret.std() * np.sqrt(252); picks[s] = shp
    oos = summarize(r, 2012)
    rows.append({'Stop': sname(s), 'IS 1997-2011 Sharpe': f'{shp:.2f}', 'IS CAGR': pct(eq_is.iloc[-1] ** (1 / ((eq_is.index[-1] - eq_is.index[0]).days / 365.25)) - 1),
                 'IS Max DD': pct((eq_is / eq_is.cummax() - 1).min()), 'OOS 2012-26 End $': f"${oos['end_latest']:,.0f}", 'OOS CAGR': pct(oos['cagr']), 'OOS Sharpe': f"{oos['sharpe']:.2f}",
                 'OOS Max DD': pct(oos['mdd']), 'OOS beat SPY': f"{oos['beat_n']}/{oos['n_yrs']}"})
best = max(picks, key=lambda k: picks[k]); T7 = pd.DataFrame(rows)
section('Test 7 - Out-of-sample (design on 1997-2011, score 2012-2026 once, no changes)', T7,
        f'Pre-declared selection rule: choose the stop (from the fixed grid, top-1) with the highest in-sample daily Sharpe. IS-selected stop = **{sname(best)}**; its OOS row is the single scored result. Other OOS columns are shown for completeness only and were NOT used to select.')

# ---- supplementary: luck vs momentum diagnostics -----------------------------------------------------------------
# (a) every eligible stock's forward growth each year -> random-pick Monte Carlo, decile table, EW top decile
base = go(1, None); pp = base['per']; full = pp[~pp.final]
cand_g = []; dec_rows = []; ew_rows = []
for _, row in full.iterrows():
    c, nc, nm = rank_candidates(D, row.sig, 12, 0, 10000)
    g = np.array([sim_position(D, k, row.entry, row.exit, False, None, 'open', .001)[1] - 1 for k, _ in c]); pr = np.array([r_ for _, r_ in c])
    cand_g.append(g)
    q = pd.qcut(pd.Series(pr).rank(method='first'), 10, labels=False)
    for d_ in range(10): dec_rows.append(dict(year=D.cal[row.entry].year, decile=d_ + 1, fwd=g[(q == d_).values].mean()))
    k10 = max(1, int(round(0.1 * len(c)))); ew_rows.append(dict(year=D.cal[row.entry].year, top1=g[0], topdec=g[:k10].mean(), allEW=g.mean(), median=np.median(g), pct_rank_of_pick=(g < g[0]).mean()))
dec = pd.DataFrame(dec_rows).groupby('decile').fwd.agg(['mean', 'median']); ew = pd.DataFrame(ew_rows).set_index('year')
nmc = 10000; draws = np.array([[rng.choice(g) for g in cand_g] for _ in range(nmc)]); mc_end = 100 * np.prod(1 + draws, axis=1)
s_top1 = summarize(go(1, None), 1997); my_end = 100 * np.prod(1 + np.array(s_top1['yearly']))
mc = dict(my_end=my_end, mc_pct=(mc_end < my_end).mean(), mc_5=np.percentile(mc_end, 5), mc_50=np.percentile(mc_end, 50), mc_95=np.percentile(mc_end, 95),
          mc_mean_geo_cagr=np.exp(np.log1p(draws).mean()) - 1, p_beat_spy_year=float(np.mean([ (g > 0).mean() for g in cand_g])))
ew['SPY'] = pd.Series(s_top1['spy_yearly'], index=s_top1['years'])
ew.to_csv(f'{OUT}/yearly_top1_vs_decile_vs_universe.csv'); dec.to_csv(f'{OUT}/prior_return_decile_forward_returns.csv')
cum = lambda s_: 100 * np.prod(1 + s_)
sup = pd.DataFrame([{'Portfolio (annual rebalance, no stop, 10bps not applied to decile/universe - gross of costs here)': k, 'End $100 (1997-2025, complete yrs)': f'${cum(ew[k]):,.0f}', 'Geo mean/yr': pct(np.exp(np.log1p(ew[k]).mean()) - 1),
                     'Stdev of yearly ret': pct(ew[k].std())} for k in ('top1', 'topdec', 'allEW', 'SPY')])
section('Supplement A - Is it momentum? Top-1 vs top-decile vs equal-weight universe (same survivorship-limited data)', sup,
        'Top-decile = the best 10% of eligible members by prior 12M return, equal-weighted. top1/topdec/allEW here use the same gross forward-return calc (top-1 net of 10bps/side).')
section('Supplement B - Average next-year return by prior-return decile (pooled 1997-2025; decile 10 = prior winners)', dec.reset_index().assign(**{'mean': lambda d: d['mean'].map(pct), 'median': lambda d: d['median'].map(pct)}))
mcdf = pd.DataFrame([{'Top-1 actual ending $100 (no stop)': f"${mc['my_end']:,.0f}", 'Random-pick MC 5th/50th/95th': f"${mc['mc_5']:,.0f} / ${mc['mc_50']:,.0f} / ${mc['mc_95']:,.0f}",
                      'Share of random-pick paths below actual': pct(mc['mc_pct']), 'Avg pick pct-rank of fwd return': pct(ew.pct_rank_of_pick.mean())}])
section('Supplement C - Luck check: random single-stock pick each January from the same eligible lists (10,000 paths, no stop)', mcdf,
        'Each January a random eligible S&P 500 member (not the top performer) is held 100% for the year, 10bps/side.')

# (b) factor regression with Fama-French (monthly)
try:
    import statsmodels.api as sm
    def ff(url, skip_to):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read())); txt = z.read(z.namelist()[0]).decode('latin1').split('\n')
        i0 = next(i for i, l in enumerate(txt) if l.strip()[:6].isdigit() and len(l.strip().split(',')[0].strip()) == 6)
        rows_ = []
        for l in txt[i0:]:
            p_ = [x.strip() for x in l.split(',')]
            if len(p_[0]) != 6 or not p_[0].isdigit(): break
            rows_.append(p_)
        df = pd.DataFrame(rows_).set_index(0).astype(float); df.index = pd.to_datetime(df.index, format='%Y%m') + pd.offsets.MonthEnd(0); return df / 100
    f3 = ff('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip', 0); f3.columns = ['MKT', 'SMB', 'HML', 'RF']
    um = ff('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip', 0); um.columns = ['UMD']
    F = f3.join(um, how='inner'); rows = []
    for s in (None, .35):
        for n in (1, 10):
            m = monthly(sub_equity(go(n, s)['eq'], D.cal[sample_start(go(n, s), 1997)], E0=1.0)); m.index = m.index + pd.offsets.MonthEnd(0)
            d = pd.concat([m.rename('r'), F], axis=1, join='inner'); d = d[d.index <= '2025-12-31']
            X = sm.add_constant(d[['MKT', 'SMB', 'HML', 'UMD']]); res = sm.OLS(d.r - d.RF, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
            rows.append({'Variant': f'{"Top-1" if n==1 else "EW top-10"}, stop {sname(s)}', 'Alpha (ann.)': pct(res.params['const'] * 12), 'Alpha t': f"{res.tvalues['const']:.2f}",
                         'MKT beta': f"{res.params['MKT']:.2f} (t={res.tvalues['MKT']:.1f})", 'SMB': f"{res.params['SMB']:.2f} (t={res.tvalues['SMB']:.1f})",
                         'HML': f"{res.params['HML']:.2f} (t={res.tvalues['HML']:.1f})", 'UMD (momentum)': f"{res.params['UMD']:.2f} (t={res.tvalues['UMD']:.1f})", 'R2': f'{res.rsquared:.2f}', 'Months': int(res.nobs)})
    section('Supplement D - Fama-French + Momentum (UMD) regression of monthly excess returns, 1997-2025 (HAC t-stats)', pd.DataFrame(rows))
except Exception as e:
    md_out.append(f'\n(Supplement D factor regression skipped: {e})\n'); print('FF failed', e)

# ---- picks, coverage
pk = go(1, None)['per'].copy(); pk['entry_date'] = [str(D.cal[e].date()) for e in pk.entry]
s35 = go(1, .35)['per']
pk['prior_12m_return'] = pk.lbret.astype(float).map(pct); pk['fwd_no_stop'] = pk.ret.map(pct); pk['fwd_35pct_stop'] = s35.ret.map(pct).values; pk['stopped'] = s35.stops.values
pk['spy'] = [pct(D.spyO[x] / D.spyO[e] - 1) for e, x in zip(pk.entry, pk.exit)]
pk['usable_members'] = pk.ncand.astype(str) + '/' + pk.nmem.astype(str)
pkt = pk[['entry_date', 'picks', 'prior_12m_return', 'fwd_no_stop', 'fwd_35pct_stop', 'stopped', 'spy', 'usable_members']]
pkt.to_csv(f'{OUT}/yearly_picks.csv', index=False)
section('Appendix - Every annual top-1 pick (1997 start, open-entry, 10bps) and data coverage', pkt.rename(columns={'usable_members': 'members with Yahoo data / PIT members'}), 'Last row (2026) is year-to-date, still open.')

# ---- save numerics for charts / report
pickle_out = {'s': {f'{n}|{sname(s)}|{y}': summarize(go(n, s), y) for n in RANKS for s in STOPS for y in (1997, 2012)}, 'spy': {y: spy_summary(y) for y in (1997, 2012)}, 'mc': mc, 'ntrials': NTRIALS, 'best_is_stop': sname(best)}
import pickle; pickle.dump(pickle_out, open('outputs/results.pkl', 'wb'))
for key, nm in (((1, None), 'top1_nostop'), ((1, .35), 'top1_stop35')):
    r = go(*key); r['eq'].to_csv(f'{OUT}/equity_{nm}_daily.csv'); r['per'].to_csv(f'{OUT}/periods_{nm}.csv', index=False)
open(f'{OUT}/results_tables.md', 'w').write(f'# Results tables\n\n**{HB}.**\n' + '\n'.join(md_out))
print('done; best IS stop', sname(best), 'MC', mc)
