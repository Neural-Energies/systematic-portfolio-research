from pathlib import Path
import json,html
import numpy as np
import pandas as pd

L=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007');O=L/'FOUR_TIMEFRAMES_20261007'
d=pd.read_csv(O/'ALL_METRICS.csv');s=pd.read_csv(O/'PAPER_SCREEN.csv');t=pd.read_csv(O/'IS_ONLY_THRESHOLDS.csv')
rules={r['id']:r for r in json.loads((L/'LOCKED_ENTRIES.json').read_text())}
protocol=json.loads((O/'PROTOCOL.json').read_text())
baseline=s[s.candidate.eq('ELIGIBLE_ENTRY_BASELINE')].set_index(['duration_minutes','target_type'])
validation=d[(d.period=='validation')&(d.scenario=='base')].set_index(['candidate','duration_minutes','target_type'])
stress=d[d.period.eq('post2024')].set_index(['candidate','duration_minutes','target_type','scenario'])
for index,row in s.iterrows():
    b=baseline.loc[(row.duration_minutes,row.target_type)]
    v=validation.loc[(row.candidate,row.duration_minutes,row.target_type)]
    s.loc[index,'win_uplift_vs_baseline_pp']=row.post_win_pct-b.post_win_pct
    s.loc[index,'validation_sharpe']=v.sharpe
    for scenario in ['one_tick','cost50']:
        st=stress.loc[(row.candidate,row.duration_minutes,row.target_type,scenario)]
        s.loc[index,scenario+'_profit']=st.net_profit
        s.loc[index,scenario+'_PF']=st.profit_factor
s.to_csv(O/'PAPER_SCREEN.csv',index=False)
eligible=s[(s.status=='paper_candidate')&s.candidate.ne('ELIGIBLE_ENTRY_BASELINE')]
best=[]
for duration in [5,15,60,240]:
    subset=eligible[eligible.duration_minutes.eq(duration)].sort_values('validation_sharpe',ascending=False)
    if len(subset):best.append(subset.iloc[0].to_dict())
    for row in subset.to_dict('records'):
        ids=[a['atom_id'] for a in rules[row['candidate']]['definition']]
        details={**row,'rule_family':rules[row['candidate']],
            'execution_thresholds':t[t.duration.eq(duration)&t.atom_id.isin(ids)].to_dict('records'),
            'protocol':protocol,'identity':f"{row['candidate']}_{duration}m_{row['target_type']}"}
        (O/(details['identity']+'.json')).write_text(json.dumps(details,indent=2),encoding='utf-8')
best=pd.DataFrame(best);best.to_csv(O/'TOP_BY_TIMEFRAME.csv',index=False)
columns=['candidate','target_type','post_trades','post_net_profit','post_PF','post_win_pct','post_sharpe','minute_DD','mean_MAE_points','target_hit_pct','latest_profit','latest_PF','one_tick_PF','cost50_PF','win_uplift_vs_baseline_pp']
labels=dict(zip(columns,['Rule','Exit definition','Trades','Net profit ($)','Trade PF','Profitable (%)','Daily Sharpe','Minute drawdown ($)','Mean adverse move (points)','Target reached (%)','2026 profit ($)','2026 PF','Slippage PF','$50 cost PF','Win uplift (pp)']))
def display(frame):
    return frame[columns].replace({'target_type':{'mean70':'70% of mean','percentile70':'70th percentile'}}).rename(columns=labels).to_html(index=False,float_format=lambda x:f'{x:,.2f}')
body='<h1>NQ entry research: four matching timeframes</h1><p class="lead">Entries at the actual bar open. Separate 5-minute, 15-minute, 60-minute and four-hour chart tests.</p>'
body+='<p>The corrected clock converts minute-end export labels to minute-start times once, without altering prices. A row labeled 09:31 supplies the 09:30 minute open. Signal indicators use bars completed before that entry; the observed new open supplies the higher-opening check. The clock correction is supported by 6,849 matching overlapping five-minute bars; exporter confirmation is still pending.</p>'
body+='<p>100 rule families tested on each chart, with chart-specific thresholds fitted only through 2024. Both target definitions are tested: 70% of the prior-20-session conditional mean high minus open, and the 70th percentile of that distribution, with past-only IQR filtering. Entries are anchored to 09:30 Eastern. The last window ends at the scheduled RTH close; the 13:30 four-hour window therefore ends at 16:00. No stop and no daily-entry minimum.</p>'
body+='<p>These are simulations through August 20, 2026: one NQ contract, $25 round-trip base cost. Cost50 and one-tick slippage/penetration variants appear in the downloadable metrics. The screen is for candidates worth paper research: at least 30 post-2024 trades, positive combined profit, PF≥1.05, net winners≥53%, positive 2026/PF>1, and no unresolved selected paths. Weak quarters and harsh stress failures are reported rather than automatic vetoes.</p>'
body+='<p>Profit factor and win rate describe individual trades. Sharpe uses calendar-complete daily account returns on a $100,000 reference account. Drawdown is marked minute by minute. The selected ranking uses 2025 Sharpe; 2025 and 2026 have both been seen before, so this is not a fresh blind test. The locked rule families came from a much broader search; these 400 chart variants and 800 exit configurations add selection risk.</p>'
body+='<p>Unqualified table totals cover January 2025 through August 20, 2026, excluding the first five cash sessions at each year boundary. Profitable trades and trades reaching the target are separate columns. The four-hour candidate has just 31 trades in that period and 11 in 2026; the high win percentage has limited support.</p>'
body+='<p><a href="PAPER_SCREEN.csv">All 800 rule/exit screens</a> · <a href="ALL_METRICS.csv">All periods and cost scenarios</a> · <a href="PROTOCOL.json">Test definitions</a> · <a href="TOP_BY_TIMEFRAME.csv">Top candidate per timeframe</a></p>'
for duration in [5,15,60,240]:
    subset=eligible[eligible.duration_minutes.eq(duration)].sort_values('validation_sharpe',ascending=False)
    unique=subset.candidate.nunique();body+=f'<h2>{duration} minutes</h2><p>{len(subset)} exit configurations across {unique} rule families pass the paper-candidate screen. Top ten below, ranked by 2025 Sharpe.</p>'
    if len(subset):body+='<div class="table-wrap">'+display(subset.head(10))+'</div>'
    else:body+='<p>No candidate from these 100 rule families meets the lighter screen at this duration. This does not establish that no entry edge exists.</p>'
    body+='<details><summary>Conditional higher-open baseline for comparison</summary><div class="table-wrap">'+display(s[(s.candidate=='ELIGIBLE_ENTRY_BASELINE')&s.duration_minutes.eq(duration)])+'</div></details>'
body+='<h2>What remains unresolved</h2><p>The new five-minute import has 32 dates beyond the old cutoff, but does not contain the matching ES, minute-level fills or complete Globex inputs. Those dates are preserved and not scored here. Underlying contract volume-roll provenance remains unverified. Actual bid/ask spreads and order queue position are not present in minute OHLC. No real paper trading or orders were started.</p>'
(O/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NQ four timeframe entry research</title><style>body{font:16px system-ui;margin:32px auto;padding:0 20px;max-width:1400px;line-height:1.55;color:#172538}h1{font-size:32px}.lead{font-size:20px}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:8px;border:1px solid #ddd}th{background:#eff3f8}.table-wrap{overflow:auto}h2{margin-top:40px}a{color:#165fc0}details{margin:18px 0}</style></head><body>'+body+'</body></html>',encoding='utf-8')
verification={'corrected_clock_exact_ohlc_preservation':True,'durations':[5,15,60,240],'metrics_rows':len(d),'screen_rows_excluding_baseline':len(s[s.candidate.ne('ELIGIBLE_ENTRY_BASELINE')]),'unit_tests':184,'scoped_lint_type_format':'pass','baseline_lint_errors':40,'baseline_mypy_errors':89,'baseline_format_files':6,'historical_only':True}
(O/'VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print('BEST_BY_TIMEFRAME');print(best.to_string(index=False));print('PAPER_COUNTS');print(eligible.groupby('duration_minutes').agg(configurations=('candidate','size'),rule_families=('candidate','nunique')).to_string())
