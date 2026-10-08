"""Frozen E0044 versus ordinary higher-opening entries; one NQ position at a time."""
from pathlib import Path
from datetime import datetime, UTC
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment import load_minutes
from systematic_research.reference_backtest import backtest_reference

ROOT=Path('work/nq_long_ml')
SOURCE=ROOT/'clock_entry_runs_1000'/'20261007T001139Z'
OUT=ROOT/'reference_backtests'/datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True)
minute=load_minutes('NQ')
scope=pd.read_parquet(SOURCE/'outcomes_iqr_mean70.parquet')
source_signals=pd.read_parquet(SOURCE/'signals.parquet')
strategies={'ES_confirmed_NQ_recovery_E0044':source_signals.E0044,'ordinary_opening_up':pd.Series(True,index=scope.index)}
variants={'base':(25.,0,0.),'double_cost':(50.,0,0.),'one_minute_latency':(25.,1,0.),'one_tick_penetration':(25.,0,.25)}

def summary(trades,marks,days):
    valid=trades[trades.status.isin(['target','timeout'])].copy()
    pnl=valid.net_dollars
    cumulative=np.r_[0.,pnl.cumsum().to_numpy()]
    closed_dd=float(np.max(np.maximum.accumulate(cumulative)-cumulative))
    mark_values=np.r_[0.,marks.equity.to_numpy()] if len(marks) else np.array([0.])
    minute_dd=float(np.max(np.maximum.accumulate(mark_values)-mark_values))
    daily=valid.groupby('anchor').net_dollars.sum().reindex(days,fill_value=0.)
    losers=pnl[pnl<0].sum()
    consecutive=0; worst=0
    for value in pnl:
        consecutive=consecutive+1 if value<0 else 0
        worst=max(worst,consecutive)
    return {'selected_entries':int(trades.status.ne('already_at_target').sum()),'resolved_trades':len(valid),'unresolved_trades':int(trades.status.eq('unresolved').sum()),'already_at_target':int(trades.status.eq('already_at_target').sum()),'target_hit_rate':float(valid.status.eq('target').mean()),'net_win_rate':float(pnl.gt(0).mean()),'net_dollars_observed':float(pnl.sum()),'mean_net_dollars':float(pnl.mean()),'profit_factor':float(pnl[pnl>0].sum()/-losers) if losers<0 else np.nan,'max_closed_drawdown_dollars':closed_dd,'max_minute_close_drawdown_dollars':minute_dd,'mean_mae_lower_points':float(valid.mae_lower_points.mean()),'mean_mae_upper_points':float(valid.mae_upper_points.mean()),'worst_trade_dollars':float(pnl.min()),'best_trade_dollars':float(pnl.max()),'longest_losing_streak':worst,'mean_hold_minutes':float(valid.holding_minutes.mean()),'session_coverage_selected':trades[trades.status.ne('already_at_target')].anchor.nunique()/len(days),'candidate_sessions':len(days),'mean_resolved_trades_per_session':len(valid)/len(days),'daily_sharpe_observed':float(daily.mean()/daily.std(ddof=1)*np.sqrt(252)) if daily.std(ddof=1)>0 else np.nan}

rows=[]; quarterly=[]
for name,signal in strategies.items():
    for variant,(cost,latency,penetration) in variants.items():
        trades,marks=backtest_reference(minute,scope,signal,cost=cost,latency_minutes=latency,penetration=penetration)
        trades.to_csv(OUT/f'trades_{name}_{variant}.csv',index=False)
        marks.to_parquet(OUT/f'equity_{name}_{variant}.parquet')
        for split in ['development','later_test','all']:
            subset=trades if split=='all' else trades[trades.split.eq(split)]
            opportunities=scope if split=='all' else scope[scope.split.eq(split)]
            days=pd.DatetimeIndex(opportunities.anchor.unique()).sort_values()
            if len(subset):
                a=pd.Timestamp(opportunities.index.min()); b=pd.Timestamp(opportunities.planned_exit.max())
                mm=marks[(marks.time>a)&(marks.time<=b)].copy()
                if len(mm):
                    prior=marks[marks.time<=a]
                    if len(prior): mm.equity-=prior.equity.iloc[-1]
            else: mm=marks.iloc[:0]
            rows.append({'strategy':name,'variant':variant,'split':split,**summary(subset,mm,days)})
        valid=trades[trades.status.isin(['target','timeout'])].copy()
        valid['quarter']=pd.DatetimeIndex(valid.entry_time).tz_convert('America/New_York').tz_localize(None).to_period('Q').astype(str)
        for quarter,group in valid.groupby('quarter'):
            quarterly.append({'strategy':name,'variant':variant,'quarter':quarter,'trades':len(group),'net_dollars':float(group.net_dollars.sum()),'target_hit_rate':float(group.status.eq('target').mean())})
        print(name,variant,'resolved',len(valid),'unknown',int(trades.status.eq('unresolved').sum()),flush=True)
result=pd.DataFrame(rows)
result.to_csv(OUT/'summary.csv',index=False)
pd.DataFrame(quarterly).to_csv(OUT/'quarterly.csv',index=False)
protocol={'source_run':str(SOURCE),'frozen_rule':'E0044, no retuning','comparison':'ordinary eligible opening-up opportunities, NOT the weighted matched 57.3% control statistic','reference':'IQR-filtered previous-20-session 70% mean high-minus-open; same clock, qualifying opening-up history','execution':'one NQ contract; one position at a time; check signals each five minutes; frozen price at decision; target touch or terminal minute close; no stop; no overnight holding','costs':'$25 round trip assumed all-in; $50 stress; exact fees and spread not available','robustness':'one-minute entry delay keeps target and deadline fixed; one-tick penetration requires high >= target + 0.25 but fill remains target','unknown':'selected entries with missing minute coverage remain unresolved; block new entries until their deadline; dollar and equity summaries cover resolved trades only, therefore incomplete if unresolved > 0','data':'same reused 2023-2025 development source; no sealed-year reads; NQ volume-roll unverified; not fresh holdout validation','equity':'minute-close mark-to-market plus realized exits, observed resolved paths; OHLC exit-minute MAE lower/upper bounds rather than invented tick ordering'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
for name in strategies:
    trades=pd.read_csv(OUT/f'trades_{name}_base.csv',parse_dates=['exit_time'])
    valid=trades[trades.status.isin(['target','timeout'])]
    axes[0].plot(valid.exit_time,valid.net_dollars.cumsum(),label=name.replace('_',' '))
    cumulative=np.r_[0.,valid.net_dollars.cumsum().to_numpy()]
    axes[1].plot(valid.exit_time,(cumulative-np.maximum.accumulate(cumulative))[1:])
axes[0].set_title('NQ RTH: resolved-trade equity, one contract, $25 costs')
axes[0].set_ylabel('Cumulative dollars');axes[0].legend()
axes[1].set_ylabel('Closed-trade drawdown ($)')
axes[1].set_title('Missing-data trades remain unresolved; no stops')
fig.savefig(OUT/'equity.png',dpi=150);plt.close(fig)
lines=['# NQ RTH full chronological backtest','',
'Frozen E0044 and ordinary higher-opening entries were replayed as one position at a time on the available research data. The original 57.3% was a matched control statistic, not a separate deployable strategy. The executable comparator here takes ordinary eligible entries and uses the identical exit rule.','',
'Long one NQ contract; evaluate every five minutes; buy at minute open; frozen target = decision open + IQR-filtered 70% historical mean excursion; sell at that reference when reached, otherwise at the final minute close of the one-hour window, capped at 16:00 RTH. No stops. No retuning. $25 assumed total round-trip costs; $50 cost stress.','',
'## Results','',
'| Strategy | Period | Execution | Resolved / unknown | Target hit | Net wins | Observed net $ | Profit factor | Minute-close DD $ | Coverage |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for _,v in result.iterrows():
    if v.split=='all': continue
    lines.append(f'| {v.strategy} | {v.split} | {v.variant} | {int(v.resolved_trades)}/{int(v.unresolved_trades)} | {v.target_hit_rate:.1%} | {v.net_win_rate:.1%} | {v.net_dollars_observed:,.0f} | {v.profit_factor:.2f} | {v.max_minute_close_drawdown_dollars:,.0f} | {v.session_coverage_selected:.1%} |')
lines.extend(['','Target hits differ from the earlier overlapping-event counts because an open position blocks additional signals. Net-win rate differs from target-hit rate because timeout exits can win or lose and costs can absorb small target gains.','',
'## Important limits','',
'Any unresolved trades make the reported totals and equity incomplete; they are not silently erased from selected-entry counts or treated as zero P&L. Position scheduling includes those trades. These are observed resolved-trade sums, not a complete audited account return. No target or entry parameters were optimized in this replay. The period was already used to select the signal, so this is a realistic execution check, not untouched out-of-sample proof. Contract-level NQ rollover is not verified. No margin, capital return or risk sizing claim is made. Minute OHLC cannot determine exact adverse movement before an exit within the same minute; both MAE bounds are supplied in the trade logs.','',
'[Equity and closed-trade drawdown](equity.png)','',f'Run folder: `{OUT.resolve()}`'])
report=ROOT/'NQ_RTH_FULL_BACKTEST.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('COMPLETE',OUT)
print(result[(result.variant=='base')][['strategy','split','resolved_trades','unresolved_trades','target_hit_rate','net_win_rate','net_dollars_observed','profit_factor','max_minute_close_drawdown_dollars','session_coverage_selected']].to_string(index=False))
