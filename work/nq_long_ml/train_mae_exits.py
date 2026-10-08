"""Fixed entries; historical and learned MAE stops/MFE targets; chronological selection."""
raise SystemExit(
    "Superseded: run separate_exit_components.py. Components must be evaluated "
    "independently and the reserved-year outcomes must be excluded."
)
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_pinball_loss
from systematic_research.mae_brackets import bracket_outcomes
from systematic_research.execution_stress import schedule_events,marked_drawdowns
from systematic_research.trade_report_metrics import trade_kpis

OUT=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/MAE_ML_EXITS_20261007');OUT.mkdir(exist_ok=True)
leaders={15:'NQ_000502850189',60:'NQ_001009785391',240:'NQ_000054683951'}
results=[];prediction_metrics=[];decisions=[];saved_configs=[]

def research(duration,m,f,b,features,signals):
    cid=leaders[duration];print('TRAIN_MODELS',duration,cid,flush=True)
    # Pooled opportunities provide enough observations to learn excursions;
    # entry rules remain unchanged and select which forecasts become trades.
    scale=pd.Series((b.high-b.low).ewm(span=14,adjust=False,min_periods=14).mean().to_numpy(),index=pd.DatetimeIndex(b.available_at)).reindex(f.index,method='ffill').clip(lower=.25)
    x=features.replace([np.inf,-np.inf],np.nan).copy();x['clock_minutes']=f.slot;x['prior_range_points']=scale
    x['open_gap_scaled']=(f.entry_open-b.close.set_axis(pd.DatetimeIndex(b.available_at)).reindex(f.index,method='ffill'))/scale
    labels={'MFE':f.mfe_points/scale,'MAE':f.mae_points/scale}
    train=f.split.eq('in_sample')&f.eligible&f.scorable&scale.notna()
    predictions={};models={}
    for label,y in labels.items():
        pool=train&y.notna();common={'random_state':20261007}
        definitions={'ridge':make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=20)),
                     'extra_trees':make_pipeline(SimpleImputer(add_indicator=True),ExtraTreesRegressor(n_estimators=80,max_depth=6,min_samples_leaf=30,n_jobs=2,**common))}
        for q in ([.5,.7] if label=='MFE' else [.75,.9]):
            definitions[f'quantile_{q}']=make_pipeline(SimpleImputer(add_indicator=True),HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5,early_stopping=False,**common))
        for name,model in definitions.items():
            log=name in ['ridge','extra_trees'];response=np.log1p(y[pool]) if log else y[pool]
            model.fit(x.loc[pool],response)
            raw=model.predict(x);normalized=np.expm1(np.clip(raw,0,10)) if log else np.maximum(raw,0)
            pred=np.ceil(np.maximum(normalized*scale.to_numpy(),.25)*4)/4
            key=(label,name);predictions[key]=pred;models[key]=model
            for period in ['validation','final_confirmation']:
                mask=f.split.eq(period)&f.scorable&f.eligible&scale.notna();actual=(y*scale)[mask].to_numpy();estimated=pred[mask]
                prediction_metrics.append({'duration':duration,'label':label,'model':name,'period':period,'samples':len(actual),'mae_points':mean_absolute_error(actual,estimated),'rmse_points':np.sqrt(mean_squared_error(actual,estimated)),
                                           'mse_points':mean_squared_error(actual,estimated),'actual_below_prediction':float(np.mean(actual<=estimated))})
    signal=signals[cid]&scale.notna().to_numpy();starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8;clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float)
    variants={'baseline_no_stop':(f.target_points.to_numpy(),np.full(len(f),np.inf),None,None)}
    # Simple MAE reference fit only on the pooled discovery opportunities.
    for q in [.75,.9]:
        stop=np.ceil(labels['MAE'][train].quantile(q)*scale.to_numpy()*4)/4
        variants[f'historical_MAE_{q}']=(f.target_points.to_numpy(),stop,None,None)
    for target_model in ['ridge','extra_trees','quantile_0.5','quantile_0.7']:
        for stop_model in ['ridge','extra_trees','quantile_0.75','quantile_0.9']:
            for factor in [.7,1.]:
                name=f'{target_model}_target{factor}_{stop_model}_stop'
                variants[name]=(np.ceil(predictions[('MFE',target_model)]*factor*4)/4,predictions[('MAE',stop_model)],target_model,stop_model)
    summaries=[]
    def score(name,values,period,stress=False,save=False):
        target,stop,_,_=values;cost=50. if stress else 25.;slip=.25 if stress else 0.;penetration=.25 if stress else 0.
        outcomes=bracket_outcomes(clock,ohlc,starts,ends,f.entry_open.to_numpy()+target,stop,0,cost,slip,slip,penetration)
        scope=f.split.eq(period).to_numpy();chosen=schedule_events(signal&scope&np.isfinite(target)&(target>0)&(stop>0),starts,outcomes);a=outcomes[chosen]
        t=f.iloc[chosen][['anchor','planned_exit']].copy();t['target_points']=target[chosen];t['stop_points']=stop[chosen];t['entry_time']=f.index[chosen]
        t['entry_price']=a[:,4];t['exit_price']=a[:,5];t['exit_time']=pd.to_datetime(a[:,0].astype(np.int64),utc=True);t['net_dollars']=a[:,1]
        t['status']=np.where(a[:,3]==0,'unresolved',np.where(a[:,11]==-1,'stop',np.where(a[:,2]==1,'target','timeout')))
        t['holding_minutes']=a[:,8];t['mae_upper_points']=a[:,6]
        sessions=pd.DatetimeIndex(f.loc[scope,'anchor'].unique()).sort_values();kp,curve=trade_kpis(t,sessions);dd,low=marked_drawdowns(clock,ohlc,starts,outcomes,chosen,0,cost)
        kp.update(duration=duration,candidate=cid,variant=name,period=period,stress=stress,minute_close_dd=dd,minute_low_dd_bound=low,stop_rate=float(t.status.eq('stop').mean()),same_minute_order='stop_first')
        if save:
            tag=f'{duration}m_{period}_{"stress" if stress else "base"}_{"baseline" if name=="baseline_no_stop" else "selected"}'
            t.to_csv(OUT/(tag+'_trades.csv'),index=False);curve.to_csv(OUT/(tag+'_daily.csv'))
        results.append(kp);return kp
    for name,values in variants.items():
        kp=score(name,values,'validation');summaries.append(kp)
    val=pd.DataFrame(summaries);base=val[val.variant.eq('baseline_no_stop')].iloc[0]
    # Profit is the primary objective. Avoid selecting a configuration from tiny
    # validation samples or unresolved executions; baseline participates too.
    candidates=val[(val.trades>=max(10,base.trades*.8))&val.unknown_trades.eq(0)].sort_values(['net_profit','minute_close_dd'],ascending=[False,True])
    winner=candidates.iloc[0].variant if len(candidates) else 'baseline_no_stop'
    selection={'duration':duration,'candidate':cid,'selected_variant':winner,'selection_period':'2025 validation only','selection_objective':'Maximum validation net profit; at least80%baseline trade count and10trades; lower drawdown tie-break','variants_tested':len(variants),'baseline_validation_profit':base.net_profit,'selected_validation_profit':float(val[val.variant.eq(winner)].net_profit.iloc[0])}
    decisions.append(selection);(OUT/f'{duration}m_FROZEN_SELECTION.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    # Record selection BEFORE final-period performance is calculated.
    for name in dict.fromkeys(['baseline_no_stop',winner]):
        for stage in ['in_sample','final_confirmation']:
            score(name,variants[name],stage,save=stage=='final_confirmation')
        score(name,variants[name],'final_confirmation',stress=True,save=True)
    selected=variants[winner]
    package={'selection':selection,'features':list(x.columns),'normalization':'Predictions in prior completed14barEWMA range units; converted toNQpoints and rounded up0.25; target anchored recordedopen, stop anchored actualfill',
             'source_clock':'Corrected minute-start clock','target_model':selected[2],'stop_model':selected[3],'entry_rule':'Unchanged chart-specific IS thresholds from FOUR_TIMEFRAMES_20261007','execution':'1contract; no trailing update; forecast target/MAE stop fixedatentry; timeexit atmatchingwindow/sessionclose; stop-first ambiguity, gap stopfillsworseopen',
             'models':{str(k):models[k] for k in models if k in [('MFE',selected[2]),('MAE',selected[3])]}}
    joblib.dump(package,OUT/f'{duration}m_SELECTED_MODELS.joblib',compress=3)
    pd.DataFrame(results).to_csv(OUT/'EXIT_RESULTS.csv',index=False);pd.DataFrame(prediction_metrics).to_csv(OUT/'PREDICTION_ERRORS.csv',index=False)
    print('SELECTED',selection,flush=True)

source=Path('work/nq_long_ml/run_four_timeframes.py').read_text()
prefix=source[:source.index('    for target_type,column')]
prefix=prefix.replace("O=L/'FOUR_TIMEFRAMES_20261007'","O=L/'MAE_ML_EXITS_20261007'")
prefix=prefix.replace('for duration in [5,15,60,240]:','for duration in [15,60,240]:')
prefix += '\n    research(duration,m,f,b,features,signals)\n'
exec(compile(prefix,'<frozen matching-timeframe preparation>','exec'),globals())
protocol={'entries_frozen':leaders,'target_and_stop_models':['Ridge','ExtraTrees','Histogramgradientboostingquantile'],
 'training':'Pooled eligible opportunities2022-2024, no post2024labels. Features median-imputed in training with missing indicators; linear model standardized. Labels normalized by prior completed14bar EWMA range. No outcome outliers dropped.',
 'selection':'35exitvariants per timeframe, maximum2025profit then lowerDDties, min10and80%baseline samplecount; baseline iseligible; selection written before2026 evaluation. No finalyearrescue tuning.',
 'stops':'Predict full-windowMAE; use as fixed at-entry adverse-distance stop. Actual futureMAE neverusedfor deciding same trade. Targets forecast full-windowMFE. Same-minutebothstopandtarget stopfirst; gap-stopfillsworseopen.',
 'scope':'Historical only; previouslyseen2025/2026; entry shortlistalreadyselectedonrecenthistory; resultsnotfreshunbiasedholdout. No promiseofmaximumachievableprofit. LastRTHwindowclippedclose; no overnight extension inthisstage.',
 'sources':['https://scikit-learn.org/1.5/auto_examples/ensemble/plot_gradient_boosting_quantile.html']}
(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
pd.DataFrame(decisions).to_csv(OUT/'SELECTIONS.csv',index=False)
d=pd.DataFrame(results);print(d[d.period.eq('final_confirmation')][['duration','variant','stress','trades','net_profit','profit_factor','net_win_rate','minute_close_dd','stop_rate']].to_string(index=False))
page='<h1>NQ: learned targets and MAE-based stops</h1><p>Entry rules unchanged; models trained through2024. Exit settings chosen for2025profit before2026replay. Baseline may win. Actual futureMAEis neveran input tothat trade. Stops andtargets fixedatentry; same-minute ambiguity resolvesstopfirst. Gapstops fillatworseopen.</p>'
page+='<p>Theseare previouslyseen historical data, notblindorlive results.35 variants/timeframe addselectionrisk; no overnight holdextensionorposition sizing. Basecost$25;stress$50plus1tickentry/stop/timeexitslippageandtargetpenetration.</p>'
page+='<h2>Selections</h2>'+pd.DataFrame(decisions).to_html(index=False)
page+='<h2>2026 replay</h2>'+d[d.period.eq('final_confirmation')][['duration','variant','stress','trades','net_profit','profit_factor','net_win_rate','sharpe','minute_close_dd','stop_rate']].to_html(index=False,float_format=lambda x:f'{x:,.3f}')
page+='<p><a href="EXIT_RESULTS.csv">All backtest outcomes</a> · <a href="PREDICTION_ERRORS.csv">MAE/MFE prediction errors</a> · <a href="PROTOCOL.json">Protocol</a></p>'
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NQ MAE exit research</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}th{background:#eef}</style>'+page,encoding='utf-8')
print('COMPLETE_MAE_EXIT_RESEARCH',flush=True)
