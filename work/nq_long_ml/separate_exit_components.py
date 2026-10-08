"""Independent target/MAE components; expanding development folds; lockout excluded."""
from pathlib import Path
import json
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
from systematic_research.research_partitions import guard_research_sample,purged_training_rows,require_before_lockout

OUT=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/SEPARATE_COMPONENTS_20261007');OUT.mkdir(exist_ok=True)
LOCKOUT=pd.Timestamp('2025-08-21',tz='America/New_York').tz_convert('UTC')
LABEL_END=pd.Timestamp('2025-08-14 09:30',tz='America/New_York').tz_convert('UTC')
LEADERS={5:'NQ_001049588361',15:'NQ_000502850189',60:'NQ_001009785391',240:'NQ_000054683951'}
FOLDS=[pd.Timestamp(t,tz='America/New_York').tz_convert('UTC') for t in ['2025-01-01','2025-04-01','2025-07-01','2025-08-14 09:30']]
metrics=[];errors=[];fold_audit=[];selections=[];input_quality=[]
protocol={'lockout_start':str(LOCKOUT),'last_permitted_outcome_end_exclusive':str(LABEL_END),'last_complete_development_cash_session':'2025-08-13',
 'reserved_period':'Original sealed year2025-08-21through2026-08-21; notopened inthis component experiment; previouslyconsumed byearlierresearch so notfresh. New2026Aug22onwardalsoreserved/unscored.',
 'components':'Entry unchanged; target tested with no stop; MAE exit tested against unchanged mean70target; separately frozen choices onlythencombined',
 'training':'Expanding all earlier eligible/scorable development rows; fivecashsession boundary embargo for eachfold. Three2025folds. Finalselectedmodels refit on ALLpermitted development rows endingbeforeAugust14 boundary. No latestyearlabels.',
 'models':'RidgeandExtraTrees log1p excursion/range regression; HistGradientBoosting conditional quantiles. Training-only median imputation andmissingindicator; linear standardized. ATRproxy prior completed14RTHbarEWMA high-low range.',
 'selection':'Maximum stitcheddevelopmentOOStestprofit; baseline participates; >=80%baseline events and10trades; drawdown tie-break. Target8model/factorvariants plusbaseline. Stop4MLmodels+2historicalquantiles+baseline. No jointgridsearch.',
 'costs':'1NQcontract$20/point;$25roundtripbase;stress$50plus.25entryandstop/timeexit slippage and.25targetpenetration. No sizingorovernighthold.',
 'execution':'Buy actual matchingRTHbaropen usingcompleted features andobserved openingup; fixedtargetandstop atentry; targetorMAEstoporwindow/sessionclose. Intraminutedualtouch stopfirst; observedtargetgapfirst; adversegapstopfillsworseopen.',
 'limitations':'Entryfamilies werealreadyselectedusingpost-cutoffhistory; cannotundo thatcontamination. Thesearedevelopmentcomponent tests,notindependentstrategyvalidation. Rollprovenanceunverified. New5mexport lacksminute/ES/fullGlobexinputs.',
 'sources':['https://scikit-learn.org/1.5/auto_examples/ensemble/plot_gradient_boosting_quantile.html']}
(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

def make_model(label,name):
    if name=='ridge':
        return make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=20))
    if name=='extra_trees':
        return make_pipeline(SimpleImputer(add_indicator=True),ExtraTreesRegressor(n_estimators=80,max_depth=6,min_samples_leaf=30,n_jobs=2,random_state=20261007))
    return make_pipeline(SimpleImputer(add_indicator=True),HistGradientBoostingRegressor(loss='quantile',quantile=float(name.split('_')[1]),max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=20261007))

def research(duration,m,f,b,features,signals):
    cid=LEADERS[duration];print('COMPONENT_START',duration,cid,flush=True)
    guard_research_sample(f.index,'selection')
    require_before_lockout(m.index,LOCKOUT);require_before_lockout(f.index,LOCKOUT)
    assert pd.DatetimeIndex(f.planned_exit).max()<=LABEL_END
    session_clock=pd.DatetimeIndex(schedule.open)
    scale=pd.Series((b.high-b.low).ewm(span=14,adjust=False,min_periods=14).mean().to_numpy(),index=pd.DatetimeIndex(b.available_at)).reindex(f.index,method='ffill').clip(lower=.25)
    x=features.replace([np.inf,-np.inf],np.nan).copy();x['clock_minutes']=f.slot;x['prior_range_points']=scale
    x['open_gap_scaled']=(f.entry_open-b.close.set_axis(pd.DatetimeIndex(b.available_at)).reindex(f.index,method='ffill'))/scale
    y={'MFE':f.mfe_points/scale,'MAE':f.mae_points/scale}
    model_names={'MFE':['ridge','extra_trees','q_0.5','q_0.7'],'MAE':['ridge','extra_trees','q_0.75','q_0.9']}
    pred={(label,name):np.full(len(f),np.nan) for label,names in model_names.items() for name in names}
    historical={q:np.full(len(f),np.nan) for q in [.75,.9]}
    eligible=f.eligible.to_numpy()&scale.notna().to_numpy();evaluation=np.zeros(len(f),dtype=bool)
    for start,end in zip(FOLDS[:-1],FOLDS[1:]):
        train=purged_training_rows(f.index,pd.DatetimeIndex(f.planned_exit),session_clock,start)&eligible&f.scorable.to_numpy()
        test=np.asarray((f.index>=start)&(f.index<end))&eligible;evaluation|=test
        assert not np.any(train&test);require_before_lockout(f.index[train|test],LOCKOUT)
        fold_audit.append({'duration':duration,'fold_start':str(start),'fold_end':str(end),'training_rows':int(train.sum()),'test_opportunities':int(test.sum()),'training_last_label':str(pd.DatetimeIndex(f.planned_exit)[train].max()),'lockout_rows':0})
        for q in historical:historical[q][test]=np.ceil(y['MAE'][train].quantile(q)*scale[test].to_numpy()*4)/4
        for label,names in model_names.items():
            for name in names:
                model=make_model(label,name);log=name in ['ridge','extra_trees'];values=y[label][train]
                guard_research_sample(f.index[train],'fit')
                model.fit(x.loc[train],np.log1p(values) if log else values)
                raw=model.predict(x.loc[test]);normalized=np.expm1(np.clip(raw,0,10)) if log else np.maximum(raw,0)
                points=np.ceil(np.maximum(normalized*scale[test].to_numpy(),.25)*4)/4;pred[(label,name)][test]=points
                valid=f.scorable.to_numpy()[test];actual=(y[label][test]*scale[test]).to_numpy()[valid];estimated=points[valid]
                errors.append({'duration':duration,'label':label,'model':name,'fold_start':str(start),'samples':len(actual),'MAE_points':mean_absolute_error(actual,estimated),'MSE_points':mean_squared_error(actual,estimated),'RMSE_points':np.sqrt(mean_squared_error(actual,estimated)),
                               'coverage_actual_below_forecast':float(np.mean(actual<=estimated)),'pinball_loss':mean_pinball_loss(actual,estimated,alpha=float(name.split('_')[1])) if name.startswith('q_') else np.nan})
        print('FOLD_DONE',duration,str(start),int(train.sum()),int(test.sum()),flush=True)
    input_quality.append({'duration':duration,'development_rows':len(f),'first':str(f.index.min()),'last':str(f.index.max()),'evaluation_opportunities':int(evaluation.sum()),'feature_columns':len(x.columns),'missing_feature_fraction':float(x.isna().to_numpy().mean()),'label_incomplete_rows':int((~f.scorable).sum()),'reserved_rows':0})
    clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float);starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8
    signal=signals[cid]&evaluation;base_target=f.target_points.to_numpy();no_stop=np.full(len(f),np.inf)
    settings={'baseline':(base_target,no_stop)}
    target_variants=[];stop_variants=[]
    for name in model_names['MFE']:
        for factor in [.7,1.]:
            key=f'target_{name}_x{factor}';settings[key]=(np.ceil(pred[('MFE',name)]*factor*4)/4,no_stop);target_variants.append(key)
    for name in model_names['MAE']:
        key=f'stop_{name}';settings[key]=(base_target,pred[('MAE',name)]);stop_variants.append(key)
    for q,values in historical.items():
        key=f'stop_historical_{q}';settings[key]=(base_target,values);stop_variants.append(key)
    def score(name,stress=False,save=False):
        target,stop=settings[name];cost=50. if stress else 25.;slip=.25 if stress else 0.;penetration=.25 if stress else 0.
        a=bracket_outcomes(clock,ohlc,starts,ends,f.entry_open.to_numpy()+target,stop,0,cost,slip,slip,penetration)
        chosen=schedule_events(signal&np.isfinite(target)&(target>0)&(stop>0),starts,a);selected=a[chosen]
        t=f.iloc[chosen][['anchor','planned_exit']].copy();t['entry_time']=f.index[chosen];t['entry_price']=selected[:,4];t['exit_price']=selected[:,5];t['target_points']=target[chosen];t['stop_points']=stop[chosen]
        t['exit_time']=pd.to_datetime(selected[:,0].astype(np.int64),utc=True);t['net_dollars']=selected[:,1];t['status']=np.where(selected[:,3]==0,'unresolved',np.where(selected[:,11]==-1,'stop',np.where(selected[:,2]==1,'target','timeout')))
        t['holding_minutes']=selected[:,8];t['mae_upper_points']=selected[:,6]
        sessions=pd.DatetimeIndex(f.loc[(f.index>=FOLDS[0])&(f.index<FOLDS[-1]),'anchor'].unique()).sort_values();kp,curve=trade_kpis(t,sessions);dd,low=marked_drawdowns(clock,ohlc,starts,a,chosen,0,cost)
        kp.update(duration=duration,candidate=cid,variant=name,stress=stress,minute_close_dd=dd,minute_low_dd_bound=low,stop_rate=float(t.status.eq('stop').mean()),evaluation='Stitched expanding2025development folds; no lockout')
        metrics.append(kp)
        if save:
            tag=f'{duration}m_{name}_{"stress" if stress else "base"}'
            t.to_csv(OUT/(tag+'_trades.csv'),index=False);curve.to_csv(OUT/(tag+'_daily.csv'))
        return kp
    comparison={'baseline':score('baseline',save=True)}
    for name in target_variants+stop_variants:comparison[name]=score(name)
    chosen={}
    for component,names in [('target',target_variants),('MAE_exit',stop_variants)]:
        table=pd.DataFrame([comparison[n] for n in ['baseline']+names]);base=table[table.variant.eq('baseline')].iloc[0]
        usable=table[(table.trades>=max(10,base.trades*.8))&table.unknown_trades.eq(0)].sort_values(['net_profit','minute_close_dd'],ascending=[False,True])
        winner=usable.iloc[0].variant if len(usable) else 'baseline';chosen[component]=winner
        decision={'duration':duration,'candidate':cid,'component':component,'selected':winner,'development_profit':comparison[winner]['net_profit'],'baseline_profit':base.net_profit,'development_dd':comparison[winner]['minute_close_dd'],'baseline_dd':base.minute_close_dd,'selection':'Eachcomponent selected separately; maximumdevelopmentOOSprofit,drawdowntie-break'}
        selections.append(decision)
        (OUT/f'{duration}m_{component}_FROZEN.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
        if winner!='baseline':score(winner,save=True);score(winner,stress=True,save=True)
    score('baseline',stress=True,save=True)
    # Choices already frozen separately; this is an interaction check only.
    settings['combined_frozen_components']=(settings[chosen['target']][0],settings[chosen['MAE_exit']][1])
    score('combined_frozen_components',save=True);score('combined_frozen_components',stress=True,save=True)
    # Input was already clipped at the original calendar's five-session
    # boundary. Do not embargo again using the shortened calendar.
    final_train=np.asarray((f.index<LABEL_END)&(pd.DatetimeIndex(f.planned_exit)<=LABEL_END))&eligible&f.scorable.to_numpy()
    require_before_lockout(f.index[final_train],LOCKOUT)
    package={'duration':duration,'candidate':cid,'choices':chosen,'features':list(x.columns),'training_rows':int(final_train.sum()),'last_training_label':str(pd.DatetimeIndex(f.planned_exit)[final_train].max()),'lockout_start':str(LOCKOUT),'models':{},'normalization':'Prior completed14barEWMA high-low range; points roundedupquartertick',
             'entry_thresholds':pd.read_csv('saved_strategies/NQ_RTH_100_LOCKED_20261007/FOUR_TIMEFRAMES_20261007/IS_ONLY_THRESHOLDS.csv').query('duration==@duration').to_dict('records')}
    for component,label in [('target','MFE'),('MAE_exit','MAE')]:
        winner=chosen[component]
        if winner=='baseline':continue
        if winner.startswith('stop_historical'):
            q=float(winner.split('_')[-1]);package['models'][component]={'historical_MAE_normalized_quantile':float(y['MAE'][final_train].quantile(q))};continue
        name=winner.removeprefix('target_').split('_x')[0] if component=='target' else winner.removeprefix('stop_')
        model=make_model(label,name);log=name in ['ridge','extra_trees'];values=y[label][final_train]
        guard_research_sample(f.index[final_train],'fit')
        model.fit(x.loc[final_train],np.log1p(values) if log else values)
        package['models'][component]={'model':model,'name':name,'log_label':log,'factor':float(winner.split('_x')[-1]) if component=='target' else 1.}
    joblib.dump(package,OUT/f'{duration}m_COMPONENT_MODELS.joblib',compress=3)
    pd.DataFrame(metrics).drop_duplicates(['duration','variant','stress']).to_csv(OUT/'BACKTEST_COMPARISONS.csv',index=False);pd.DataFrame(errors).to_csv(OUT/'FORECAST_ERRORS.csv',index=False);pd.DataFrame(fold_audit).to_csv(OUT/'FOLD_INPUT_AUDIT.csv',index=False)
    print('COMPONENTS_FROZEN',duration,chosen,flush=True)

source=Path('work/nq_long_ml/run_four_timeframes.py').read_text();prefix=source[:source.index('    for target_type,column')]
prefix=prefix.replace("O=L/'FOUR_TIMEFRAMES_20261007'","O=L/'SEPARATE_COMPONENTS_20261007'")
prefix=prefix.replace('del original\n','del original\nm=m.loc[m.index<LABEL_END].copy();schedule=schedule.loc[schedule.open<LABEL_END].copy()\n')
prefix=prefix.replace("(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')",'')
prefix+='\n    research(duration,m,f,b,features,signals)\n'
exec(compile(prefix,'<permitted development preparation>','exec'),globals())
pd.DataFrame(selections).to_csv(OUT/'COMPONENT_SELECTIONS.csv',index=False)
pd.DataFrame(input_quality).to_csv(OUT/'DATA_QUALITY.csv',index=False)
print(pd.DataFrame(selections).to_string(index=False));print('SEPARATE_COMPONENT_RESEARCH_COMPLETE',flush=True)
