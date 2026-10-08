from pathlib import Path
import json
import numpy as np
import pandas as pd

out=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/COMPONENT_SENSITIVITY_20261007')
d=pd.read_csv(out/'ALL_METRICS.csv');fold=pd.read_csv(out/'FOLD_METRICS.csv')
assert len(d)==12000 and not d.duplicated(['duration','candidate','policy','stress']).any()
a=fold.groupby(['duration','candidate','policy','stress']).net_profit.sum().sort_index()
b=d.set_index(['duration','candidate','policy','stress']).net_profit.sort_index()
assert np.allclose(a,b)
for h in [5,15,60,240]:
    frame=pd.read_parquet(out/f'{h}m_inputs/opportunities.parquet')
    assert pd.DatetimeIndex(frame.planned_exit).max()<pd.Timestamp('2025-08-21',tz='UTC')
summary=pd.read_csv(out/'SUMMARY.csv');errors=pd.read_csv(out/'FORECAST_COMPARISON.csv')
text=['# Target benchmarks and MAE risk sensitivity','',
 '12,000 independent one-contract candidate-policy-cost replays. All four matching RTH charts, 100 entry families each, January–August13,2025 development only. Entry/rule/operator choices unchanged. No reserved outcomes or joint target-stop optimization.', '',
 'Hourly historical forecasts are competitive as exits:63–64of100families profitable versus61withthepreviouslearnedtarget. Learnedhourlyforecasts haveMAE33.99points versus37.76historicalmean, butRMSE70.41 versus70.72—large misses remain. Simple forecasting methods should remain benchmark components.', '',
 'Hourly learned target multipliers0.5–0.8 retained57–61profitablefamilies;0.9–1.0retained50–51. This supports studying a region of settings instead of claiming one optimal multiplier.', '',
 'Hourly NQ_000054683951 with originaltarget and MAEstopdistance×0.75:36trades,net$16,655,stress$15,570,PF3.92,win83.3%,minuteDD$2,407.50. Originaltarget/no-stop:net$10,215,DD$4,842.50. Removingbestday leaves$11,165. Q1$3,515/14trades,Q2$14,000/21trades,partialQ3−$860/1trade. This is an exploratory risk candidate, not a promoted configuration.', '',
 'The same hourly entry with historicalnormalizedMFEq70×0.7target andnostop:net$28,160,stress$27,005,PF2.62,win69.4%,DD$6,085. Existingfrozenlearnedtarget:net$26,360. The target and risk result are separate tests; their profits cannot be combined or added.', '',
 'Five-minute median results remain negative.15-minute results are mixed. Four-hour results are sensitive to sparse entry counts. Prior selection contamination, related entry families and repeated development tuning remain; no fresh significance or future-profit claim.', '',
 '## All method summaries','',summary.to_markdown(index=False),'','## Forward forecast errors','',errors.to_markdown(index=False)]
(out/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
v=json.loads((out/'VERIFICATION.json').read_text())
v.update(fold_pnl_reconciliation=True,repository_tests='191passed',baseline_static_failures='40rufferrors,89mypyerrorsin11files,6formatfiles;unchanged',configs_folder='cases',saved_research_configs=len(list((out/'cases').glob('*.json'))))
(out/'VERIFICATION.json').write_text(json.dumps(v,indent=2))
print('SENSITIVITY_RECONCILED',len(d),'CASES',v['saved_research_configs'])
