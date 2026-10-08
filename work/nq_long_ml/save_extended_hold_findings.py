from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd

O=Path('work/nq_long_ml/extended_hold_pilot')
D=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007');D.mkdir(exist_ok=True)
source=O/'LOCKED_ENTRIES.json';target=D/'LOCKED_ENTRIES.json'
if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('Do not replace a different lock')
shutil.copyfile(source,target)
protocol=json.loads((O/'PROTOCOL.json').read_text())
protocol['maximums']='Both120 elapsed hours and five cash sessions tested; entry session counts as session one, cap at fifth cash-session close. Missing executable quote at deadline makes selected trade unresolved; no late fill beyond cap is invented.'
protocol['first_pass_conclusion']='Keep original exits. Pilot did not improve win rates and forecast RMSE did not beat zero-change baseline in2026. Longer-profit result remains preliminary.'
(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));(D/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
shutil.copyfile('data/processed/nq_entry_discovery_202210/import_manifest.json',D/'SOURCE_DATA_MANIFEST.json')
lock=json.loads(target.read_text())
pd.DataFrame([{'candidate':x['id'],'definition':' AND '.join(q['definition'] for q in x['definition']),'entry_rules_frozen':True,'exit_upgrade_accepted':False} for x in lock]).to_csv(D/'ENTRY_CATALOG.csv',index=False)
result=pd.read_csv(O/'exit_comparison.csv');selection=json.loads((O/'FROZEN_EXIT_SELECTION.json').read_text())['selected_variant']
selected=result[(result.period=='final_confirmation')&(result.variant==selection)]
complete=selected[selected.unresolved.eq(0)]
status=json.loads((O/'STATUS.json').read_text())
status.update({'complete_candidate_replays':len(complete),'positive_complete_candidates':int(complete.net_dollars.gt(0).sum()),'profit_above_original_complete_candidates':int(complete.net_dollars.gt(complete.net_profit).sum()),'win_rate_above_original_complete_candidates':int(complete.win_rate.gt(complete.net_win_rate).sum()),'exit_upgrade_accepted':False,'verification':'Four deterministic execution-boundary checks passed;177 existing tests passed. Unchanged baseline40 lint errors,89 type errors,6 formatting files remain. Full calendar/roll/overnight execution verification not complete.'})
(O/'STATUS.json').write_text(json.dumps(status,indent=2));(D/'EXIT_RESEARCH_STATUS.json').write_text(json.dumps(status,indent=2))
shutil.copyfile(O/'exit_comparison.csv',D/'PRELIMINARY_EXIT_COMPARISON.csv')
shutil.copyfile(O/'regression_diagnostics.csv',D/'REGRESSION_DIAGNOSTICS.csv')
focus=selected[selected.candidate.eq('NQ_000926441771')].iloc[0]
trades=pd.read_csv(O/'NQ_000926441771_final_confirmation_ridge_multi_120h_trades.csv')
valid=trades.exit_bar.ge(0)
assert np.isclose(trades.loc[valid,'net_dollars'].sum(),focus.net_dollars)
assert np.allclose((trades.loc[valid,'exit_price']-trades.loc[valid,'entry_price'])*20-25,trades.loc[valid,'net_dollars'])
assert trades.loc[valid,'hold_minutes'].max()<=120*60
lines=['# Frozen NQ entries and the first extended-hold ML test','',
'The same100 entry definitions are locked. Original target/hour exits remain the reference; no ML exit has been approved as an upgrade. This is research, not live deployment.','',
'## What was tested','',
'Ridge regression, boosted regression trees and extra trees estimate remaining60-minute and240-minute price movement. Features use completed five-minute NQ and aligned ES observations across RTH and Globex. These are expected movement estimates, not calibrated probability scores. The continuation policy checks after the first60 minutes, then every five minutes; it holds while either estimated remaining move is positive. Both120-hour and five-trading-session caps were evaluated on2025. The selected exit was frozen before its separate2026 trade replay.','',
'## The candidate selected in the reviewed view','',
'| 2026 result | Original target/hour exit | First ML continuation exit |','|---|---:|---:|',
f'| Trades | {int(focus.trades_original)} | {int(focus.trades)} |',
f'| Net profit, one contract, assumed $25 round trip | ${focus.net_profit:,.0f} | ${focus.net_dollars:,.0f} |',
f'| Net win rate | {focus.net_win_rate:.1%} | {focus.win_rate:.1%} |',
f'| Trade profit factor | {focus.profit_factor_original:.2f} | {focus.profit_factor:.2f} |',
f'| Mean hold | Original target/hour window | {focus.mean_hold_hours:.2f} hours |',
f'| Longest ML hold | — | {focus.max_hold_hours:.1f} hours |','',
f'The ML candidate has ${focus.five_minute_mark_dd:,.0f} drawdown measured at observed five-minute marks. The original report has $9,955 minute-close drawdown; different marking frequencies prevent treating these as identical risk measurements. The extension increases observed profit but reduces win rate and trade PF.','',
'## Across the shortlist','',
f'{len(complete)} of100 selected ML candidate replays have no unresolved deadline trade. Of these,{int(complete.net_dollars.gt(complete.net_profit).sum())} have more observed profit than their original exit, and none has a higher net win rate. One remaining candidate has an unresolved trade. Correlated candidates are not independent strategies and their profits must not be summed as a portfolio.','',
'The2026 forward-movement forecasts do not improve RMSE over a zero-change forecast at either horizon. Increased historical profit alone therefore does not establish that the models reliably identify when the edge disappears. A rolling-horizon expected-return screen is not a solved dynamic-programming optimal-stopping policy.','',
'## Why this remains preliminary','',
'Some requested fixed-hold deadlines land in closures or absent quotes. Those trades remain unresolved, with complete net PNL unavailable; resolved-only profits are not fair full-strategy comparisons. Model-held paths also traverse quote gaps: the exchange calendar and all overnight/roll execution remain uncertified. The source export lacks verified volume-roll provenance.2026 was previously used in research. Constant$25 costs omit overnight spread changes and rollover trades; margin and financing are not modeled. Session frequency is still a separate unmet requirement.','',
'A proper extension needs matched-duration benchmarks with certified full-session quotes and contract-roll handling, followed by conditional continuation evaluation near the existing target/time exit. Keep the original entries and exits while that work is developed.','',
'[Optimal-stopping and statistical-learning research](https://arxiv.org/abs/math/0408276) supports comparing continuation value with immediate exit value; this pilot is a simpler screening model, not a replication of that paper.','',
'## Reproduction and checks','',
'Runner: work/nq_long_ml/extended_hold_pilot.py. Fixed seeds and chronological2022–2024 training/2025 validation; crossing training labels purged. All100 locked-entry rules use their original IS-fitted thresholds. Execution-boundary tests exclude pre-entry candles, enforce fixed hold deadlines and prevent a fabricated cross-period exit. Selected trade PNL independently reconciles to actual entry/exit quotes and$20-per-point NQ value.177 repository tests passed; baseline lint/type/format failures are unchanged.']
(D/'RESULTS.md').write_text('\n'.join(lines)+'\n')
files=[target,D/'PROTOCOL.json',D/'ENTRY_CATALOG.csv']
(D/'LOCK_MANIFEST.json').write_text(json.dumps({'locked_date_eastern':'2026-10-07','entries':100,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'research_stage':'Frozen entries; original exits retained; ML extension not promoted'},indent=2))
print(json.dumps(status,indent=2));print('REPORT',str((D/'RESULTS.md').resolve()))
