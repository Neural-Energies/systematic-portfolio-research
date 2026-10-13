"""Download adjusted daily OHLC (auto_adjust=True => dividends reinvested / total-return) for every ticker that was ever
in the PIT S&P 500 membership file, plus SPY. Delisted names that Yahoo no longer serves simply come back empty;
that gap is quantified in 02_coverage and is a *price-data* survivorship bias we cannot remove with yfinance."""
import pandas as pd, yfinance as yf, pickle, sys, time
h = pd.read_csv('data/hist_updated.csv', parse_dates=['date'])
tick = set()
for t in h.tickers: tick.update(t.split(','))
tick = sorted(tick); ymap = {t: t.replace('.', '-') for t in tick}
print(len(tick), 'tickers', flush=True)
ys = sorted(set(ymap.values()) | {'SPY'})
out = {}; B = 40
for i in range(0, len(ys), B):
    b = ys[i:i+B]
    for attempt in range(3):
        try:
            d = yf.download(b, start='1995-06-01', auto_adjust=True, progress=False, threads=True, group_by='ticker')
            break
        except Exception as e:
            print('retry', e, flush=True); time.sleep(5); d = None
    if d is None: continue
    for t in b:
        try:
            x = d[t][['Open', 'Close']].dropna(how='all')
            if len(x): out[t] = x
        except Exception: pass
    print(i, len(out), flush=True)
pickle.dump(out, open('data/prices.pkl', 'wb'))
print('done', len(out), 'of', len(ys))
