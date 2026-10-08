from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import quantstats as qs

R=Path('work/nq_long_ml/shortlist_quant_reports')
def export(cid):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    d=pd.read_csv(R/f'{cid}_final_confirmation_daily.csv',parse_dates=['date']).set_index('date')
    qs.reports.html(d['return'],output=str(R/'quantstats'/f'{cid}.html'),title=f'{cid} | NQ one contract | $100k reference account | 2026',rf=0.,periods_per_year=252,compounded=True)
    metrics=pd.read_csv(R/'final_period_comparison.csv').set_index('candidate').loc[cid]
    p=R/'quantstats'/f'{cid}.html';text=p.read_text(encoding='utf-8')
    panel=f'''<section style="margin:25px auto;padding:20px;max-width:1000px;background:#eef4f9;border-radius:10px"><h2>Individual-trade metrics</h2><p>Trades: {int(metrics.trades)} · Trade profit factor: {metrics.profit_factor:.2f} · Net-winning trades: {metrics.net_win_rate:.1%} · Target hits: {metrics.target_hit_rate:.1%} · Net profit: ${metrics.net_profit:,.0f} · Minute-close drawdown: ${metrics.max_minute_drawdown_dollars:,.0f}</p><p>The standard QuantStats metrics below use daily account returns, including cash sessions without trades. Its profit factor and win rate therefore describe daily returns, and differ from the individual-trade metrics above. Account NAV starts at $100,000; exposure is one NQ contract throughout, without position-size compounding; assumed $25 round-trip costs; zero risk-free rate. This separate 2026 period was previously seen, and contract volume-roll provenance remains unverified.</p></section>'''
    import re
    text=re.sub(r'(<body[^>]*>)',lambda match:match.group(1)+panel,text,count=1)
    p.write_text(text,encoding='utf-8')
    return cid
if __name__=='__main__':
    ids=pd.read_csv(R/'final_period_comparison.csv').candidate.tolist();errors=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(export,cid):cid for cid in ids}
        for job in as_completed(jobs):
            try:print('EXPORTED',job.result(),flush=True)
            except Exception as error:
                errors.append({'candidate':jobs[job],'error':str(error)});print('FAILED',jobs[job],repr(error),flush=True)
    (R/'quantstats_export_status.json').write_text(json.dumps({'completed':len(ids)-len(errors),'requested':len(ids),'errors':errors},indent=2))
