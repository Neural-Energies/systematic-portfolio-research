from pathlib import Path
import logging,re,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import quantstats as qs

O=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/FULL_VALIDATION_20261007/MINUTE_END_SCENARIO');Q=O/'quantstats';Q.mkdir(exist_ok=True)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
metrics=pd.read_csv(O/'all_periods_execution_metrics.csv');cid='NQ_000748828115'
for name,period,file in [('base','post_training_combined',f'{cid}_base_combined_daily.csv'),('combined_stress','post_training_combined',f'{cid}_combined_stress_combined_daily.csv'),('2026_base','final_confirmation',f'{cid}_final_confirmation_daily.csv')]:
 d=pd.read_csv(O/file,index_col=0,parse_dates=True);d.index=pd.to_datetime(d.index,utc=True).tz_convert('America/New_York').tz_localize(None).normalize();variant='base' if name=='2026_base' else name;r=metrics[(metrics.candidate==cid)&(metrics.period==period)&metrics.variant.eq(variant)].iloc[0]
 path=Q/f'{cid}_minute_end_{name}.html'
 qs.reports.html(d['return'],output=str(path),title=f'{cid} | minute-end assumption | {period} | {variant}',rf=0,periods_per_year=252,compounded=True)
 panel=f'''<section style="max-width:1000px;margin:24px auto;padding:20px;background:#eef4f9;border-radius:10px"><h2>Individual-trade simulation metrics</h2><p>Trades: {int(r.trades)} · Net profit: ${r.net_profit:,.0f} · Trade PF: {r.profit_factor:.2f} · Net-winning trades: {r.net_win_rate:.1%} · Target hits: {r.target_hit_rate:.1%} · Minute-close drawdown: ${r.minute_close_dd:,.0f}</p><p>Frozen NQ long rules; original cash-capped one-hour target/time exit; no stop. This is a historical minute-end sensitivity supported by 6849 exactly matched overlapping five-minute volumes and OHLC shapes. Human endpoint confirmation and underlying volume-roll provenance remain pending.2025/2026 were previously seen; no fresh holdout or live fills are claimed. The original ten-million-definition selection is not corrected by this report.</p><p>One contract, $100,000 reference account, zero risk-free rate,252-session annualization. Base assumes$25 round trip; combined stress uses$50 plus one-minute entry delay,one-tick market slippage and one-tick target penetration. Native QuantStats PF and win rate below describe DAILY returns, not individual trades. Position size does not compound. Kelly, risk-of-ruin and probability-of-Sharpe statistics below do not account for selection bias and are not validated sizing or safety probabilities.</p><p>This rule passes post-2024 gates but fails some discovery-period stresses:$100 costs yield−$5,980 and combined stress−$315. No rule passed every period/scenario gate.</p></section>'''
 text=path.read_text(encoding='utf-8');text=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+panel,text,count=1);text=text.replace('<th>Return</th>','<th>Daily return sum</th>').replace('<th>Cumulative</th>','<th>Account return</th>');path.write_text(text,encoding='utf-8')
 assert 'Individual-trade simulation metrics' in text
 print('QUANTSTATS',path,flush=True)
(Q/'VERIFICATION.json').write_text(json.dumps({'reports':3,'candidate':cid,'timestamp_case':'minute_end_sensitivity','scope':'Native daily return tear sheets with separate individual-trade panel','all_period_all_scenario_pass':False},indent=2))
