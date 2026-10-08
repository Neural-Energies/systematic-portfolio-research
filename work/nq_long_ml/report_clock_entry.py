"""Summarize the completed local RTH entry gauntlet and audit causal references."""
from pathlib import Path
import numpy as np
import pandas as pd
from clock_entry_gauntlet import metrics
from systematic_research.clock_excursion import excursion_summary

root=Path('work/nq_long_ml')
p=sorted((root/'clock_entry_runs_1000').iterdir())[-1]
if not (p/'protocol.json').exists():
    raise RuntimeError('Wait for the completed gauntlet')
r=pd.read_csv(p/'results.csv')
b=pd.read_csv(p/'ordinary_opening_up_baseline.csv')
rules=pd.read_csv(p/'rules.csv')
z=r[r.split.eq('later_test')].merge(rules,on='signal')
old=r[r.split.eq('development')][['signal','trim','reference','incremental_expectancy','success_rate']].rename(columns={'incremental_expectancy':'earlier_excess','success_rate':'earlier_hit_rate'})
z=z.merge(old,on=['signal','trim','reference'])
z.to_csv(p/'later_results_with_definitions.csv',index=False)
# Evidence screen: both periods positive, adequate later support, confidence interval and multiplicity.
short=z[z.events.ge(100)&z.incremental_ci_low.gt(0)&z.fdr_q.le(.05)&z.earlier_excess.gt(0)].sort_values('incremental_ci_low',ascending=False)
short.to_csv(p/'research_shortlist.csv',index=False)
frame=pd.read_parquet(p/'opportunities.parquet')
signals=pd.read_parquet(p/'signals.parquet')
sessions=pd.DatetimeIndex(frame.anchor.unique()).sort_values()
checks=[]
for trim in ('iqr','raw'):
    h=pd.read_parquet(p/f'historical_statistics_{trim}.parquet')
    valid=h[h.q70.notna()]
    for timestamp,row in valid.iloc[::max(1,len(valid)//25)].head(25).iterrows():
        current=frame.loc[timestamp]
        position=sessions.get_loc(current.anchor)
        past=sessions[position-20:position]
        observations=frame[frame.anchor.isin(past)&frame.slot.eq(current.slot)&frame.opening_up&frame.scorable]
        independent=excursion_summary(observations.excursion.to_numpy(float),trim_iqr=trim=='iqr')
        for field in ('mean','median','std','q25','q70','q75','mean70','retained_samples'):
            assert np.isclose(row[field],independent[field],equal_nan=True),(timestamp,field)
        assert (observations.index<timestamp).all()
        checks.append({'history':trim,'timestamp':timestamp,'qualified':len(observations),'reference_matches':True})
pd.DataFrame(checks).to_csv(p/'reference_audit.csv',index=False)
# Restrict selected candidates to actual session-anchored hourly opens as a sensitivity.
sensitivity=[]
selected=short.head(8)
if selected.empty:
    selected=z[z.events.ge(100)].sort_values('incremental_ci_low',ascending=False).head(8)
for _,row in selected.iterrows():
    scope=pd.read_parquet(p/f'outcomes_{row.trim}_{row.reference}.parquet')
    scope=scope[scope.split.eq('later_test')&((scope.slot-570)%60).eq(0)]
    days=pd.DatetimeIndex(frame[frame.split.eq('later_test')].anchor.unique()).sort_values()
    entry=scope[signals[row.signal].reindex(scope.index)]
    sensitivity.append({'signal':row.signal,'trim':row.trim,'reference':row.reference,**metrics(entry,'hit',days)})
pd.DataFrame(sensitivity).to_csv(p/'hourly_grid_sensitivity.csv',index=False)

lines=['# NQ RTH: 1,000 entry rules, one-hour excursions','',
'This experiment measures entries. It does not train a high forecast or stop model. Each entry needs open > the prior observed hourly-stream open. One-hour forward windows begin on five-minute signal boundaries and stop at the RTH close. The hourly-open-only sensitivity is saved separately.','',
'Each reference uses the previous 20 trading-session positions at the same New York clock. Only qualifying higher-opening, complete historical bars contribute high minus open. There are typically about 10 qualifying samples. Both unfiltered and past-only 1.5-IQR-filtered histories were tested. Future failures and large drawdowns were kept.','',
'## Ordinary eligible entries: later-period reference','',
'| History | Exit reference | Events | Target hit | Average target points | Average full-window MAE | Hypothetical points/entry |','|---|---|---:|---:|---:|---:|---:|']
for _,v in b[b.split.eq('later_test')].iterrows():
    lines.append(f'| {v.trim} | {v.reference} | {int(v.events):,} | {v.success_rate:.1%} | {v.mean_target_points:.2f} | {v.mean_mae_points:.2f} | {v.mean_points_to_hypothetical_exit:.2f} |')
lines.extend(['','P70 means the 70th percentile. Mean70 means 70% of the historical mean; it is a smaller target, so its higher hit rate is not evidence of a better entry. Hypothetical exits take the reference when touched and otherwise the end-window close, with no stop. These are overlapping events, not a realizable independent-trade equity curve.','',
f'## Research shortlist: {len(short)} rule/reference combinations','',
'Requires at least 100 later events, positive matched uplift in both periods, positive lower confidence bound, and FDR q <= 0.05 across all 4,000 later comparisons. This is an exploratory screen using repeatedly studied data, not fresh confirmation.','',
'| Entry | Context | History / exit | Hits/events | Matched uplift | Earlier uplift | Session coverage | Mean target | Mean drawdown to exit | Hypothetical points |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|'])
for _,v in short.head(15).iterrows():
    lines.append(f'| {v.signal}: {v.base_rule} | {v.context} | {v.trim}/{v.reference} | {int(v.hit_events)}/{int(v.events)} ({v.success_rate:.1%}) | {v.incremental_expectancy:.1%} | {v.earlier_excess:.1%} | {v.session_coverage:.1%} | {v.mean_target_points:.2f} | {v.mean_drawdown_until_hypothetical_exit:.2f} | {v.mean_points_to_hypothetical_exit:.2f} |')
lines.extend(['','## Daily frequency','',f'Later combinations with an entry every available mature RTH session: {int(z.session_coverage.eq(1).sum())}. The ordinary opening-up baseline covers every mature later session. High-frequency candidates are listed below; uncertainty must still be checked.','',
'| Entry | Context | Exit/history | Session coverage | Hit rate | Matched uplift | Lower confidence bound |','|---|---|---|---:|---:|---:|---:|'])
for _,v in z[z.session_coverage.ge(.95)].sort_values('incremental_expectancy',ascending=False).head(6).iterrows():
    lines.append(f'| {v.signal}: {v.base_rule} | {v.context} | {v.reference}/{v.trim} | {v.session_coverage:.1%} | {v.success_rate:.1%} | {v.incremental_expectancy:.1%} | {v.incremental_ci_low:.1%} |')
lines.extend(['','## Scope and verification','',f'Exactly 1,000 distinct earlier-period event streams were selected without performance ranking from 6,864 definitions. These combine 156 base concepts and context filters; they are not 1,000 independent economic explanations. {len(checks)} causal reference checks matched the stored statistics. Rules, all results, outcomes, history statistics, matched comparisons, unknown-outcome rates and hourly-grid sensitivities are saved alongside this report.','',
'No sealed year was used. Data: approximately August 2023–August 2025, split into earlier development and later evaluation. NQ continuous-contract volume rollover remains unverified. Profiles reconstructed from minute OHLCV are approximations. Therefore any surviving rule needs fresh contract-verified data before an institutional-grade claim.','',
'IQR definition: [NIST box-plot fences](https://www.itl.nist.gov/div898/handbook/eda/section3/boxplot.htm).','',f'Full run: `{p.resolve()}`'])
report=root/'NQ_RTH_1000_ENTRY_RESULTS.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('REPORT',report.resolve())
print('BASELINES\n',b[b.split.eq('later_test')][['trim','reference','success_rate','mean_target_points']].to_string(index=False))
print('SHORTLIST',len(short),'UNIQUE RULES',short.signal.nunique())
print(short[['signal','base_rule','context','trim','reference','events','success_rate','incremental_expectancy','earlier_excess','session_coverage','fdr_q']].head(8).to_string(index=False))
print('AUDIT',len(checks))
