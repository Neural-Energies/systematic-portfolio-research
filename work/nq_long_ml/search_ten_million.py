"""Resumable ten-million-definition entry search; candidate files in compressed shards."""
from pathlib import Path
from datetime import datetime,UTC
import argparse,hashlib,json,math,shutil,sqlite3,time,zipfile,msvcrt
import numpy as np
import pandas as pd
from numba import set_num_threads
from systematic_research.entry_discovery_search import screen_rules,unrank_triples,execution_metrics,hit_tier

P=Path('data/processed/nq_entry_discovery_202210')
ROOT=Path('work/nq_long_ml/ten_million_entry_search')
ROOT.mkdir(parents=True,exist_ok=True)
def packed(values):
    x=np.asarray(values,dtype=bool)
    return np.packbits(np.pad(x,(0,(-len(x))%64)),bitorder='little').copy().view('<u8')

def run(target=10_000_000,batch_size=2000):
    lease=(ROOT/'worker.lock').open('a+b')
    if lease.tell()==0:lease.write(b'0');lease.flush()
    lease.seek(0)
    try:msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another worker owns this search; do not launch a second writer')
    set_num_threads(4)
    fingerprints={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['import_manifest.json','feature_manifest.json','atom_bits.npy','atoms.csv','opportunities.parquet']}
    frozen=ROOT/'input_fingerprints.json'
    if frozen.exists() and json.loads(frozen.read_text())!=fingerprints:raise ValueError('Frozen input fingerprint mismatch; use a new research run folder')
    frozen.write_text(json.dumps(fingerprints,indent=2))
    f=pd.read_parquet(P/'opportunities.parquet')
    atoms=np.load(P/'atom_bits.npy',mmap_mode='r')
    registry=pd.read_csv(P/'atoms.csv')
    groups=pd.factorize(registry.feature)[0].astype(np.int64)
    masks=np.stack([packed(f.split.eq(s)&f.eligible) for s in ['in_sample','validation']])
    hits=np.stack([packed(f[name].eq(1)) for name in ['mean70_hit','percentile70_hit','full_range70_hit','mean50_hit','mean90_hit','mean100_hit']])
    stage=np.where(f.split.eq('in_sample'),0,np.where(f.split.eq('validation'),1,-1)).astype(np.int64)
    starts=f.index.asi8.copy()
    ends=np.where(f.exit_available_ns.gt(0),f.exit_available_ns,pd.DatetimeIndex(f.planned_exit).asi8).astype(np.int64)
    pnl=f.hypothetical_net_dollars.to_numpy(float);mae=f.mae_points.to_numpy(float);mfe=f.mfe_points.to_numpy(float)
    target_hits=f.mean70_hit.eq(1).to_numpy(bool)
    baselines=[float(f.loc[f.eligible&f.split.eq(s),'mean70_hit'].fillna(0).mean()) for s in ['in_sample','validation']]
    days=pd.factorize(f.anchor)[0].astype(np.int64)
    candidate_days=[f.loc[f.split.eq(s)&f.eligible,'anchor'].nunique() for s in ['in_sample','validation']]
    total=math.comb(len(atoms),3);step=1000003
    while math.gcd(step,total)!=1:step+=2
    offset=20261006%total
    db=sqlite3.connect(ROOT/'registry.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS signatures (hash BLOB PRIMARY KEY,a INTEGER,b INTEGER,c INTEGER) WITHOUT ROWID')
    db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY,payload TEXT)')
    prior=db.execute('SELECT payload FROM state WHERE id=1').fetchone()
    state=json.loads(prior[0]) if prior else {'cursor':0,'definitions_tested':0,'saved_unique_signals':0,'duplicate_is_streams':0,'failed_save_cutoff':0,'started_utc':datetime.now(UTC).isoformat()}
    protocol={'target_unique_canonical_definitions':target,'rule_grammar':'AND of three different-feature predicates, current opening_up required; affine permutation of canonical triples without repeated definitions','unique_method_vs_behavior':'Definitions are unique; same IS event streams are deduplicated among saved candidates by digest plus exact verification. Ten million definitions are not ten million independent economic ideas.','save':'IS conservative raw-event mean70 hit rate strictly >53%, any sample size; individual JSON file per unique passing signal in a ZIP shard, not a live recommendation','tiers':'50-53 watch, >53 saved, >=70 tier1, >=90 tier2, exactly100 tier3; each period labeled separately. Tiny samples retain insufficient_evidence flags.','quantitative_metrics':'raw overlapping-event counts and three target hit rates; each unique saved candidate also receives nonoverlapping single-position hypothetical $25-cost PNL, closed-trade DD, full-hour MFE/MAE sums and session coverage. Detailed minute paths can be reconstructed from definition and frozen data.','splits':'IS 2022-2024, validation 2025, final 2026; five-session boundary embargo. Final confirmation is not scored by search. Validation does not alter thresholds or definitions. No assertion that globally reused dates are historically untouched.','references':'mean70 primary, P70 comparison, 70% mean high-minus-low comparison; history past20 same-clock qualifying sessions, IQR past only.','economic_validation':'Hit rate != profitability. No stops. Finite sample, interval-censored target-minute drawdown and unverified export volume roll prevent institutional-grade validation claim.','research_sources':['https://www.nber.org/papers/w7613','https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf'],'resource_limits':'four native compute threads, bounded batches, restartable SQLite checkpoint, stop safely below 4GiB free disk; candidate files compressed to avoid millions of filesystem entries','final_confirmation':'Only a frozen, adequately supported shortlist should be replayed on 2026 after discovery; this script never scans final confirmation labels or returns in metrics.'}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    began=time.monotonic();initial=state['definitions_tested']
    while state['definitions_tested']<target and state['cursor']<total:
        if (ROOT/'STOP_REQUESTED').exists():state['status']='paused_by_stop_file';break
        if shutil.disk_usage(ROOT).free<4*1024**3:state['status']='resource_blocked_low_disk';break
        cursor=state['cursor'];size=min(batch_size,total-cursor)
        ranks=((cursor+np.arange(size,dtype=np.int64))*step+offset)%total
        definitions=unrank_triples(ranks,len(atoms))
        scores=screen_rules(atoms,definitions,masks,hits,groups)
        valid=np.flatnonzero(scores[:,14])
        remaining=target-state['definitions_tested']
        if len(valid)>remaining:valid=valid[:remaining];size=int(valid[-1])+1
        records=[];saved=0;dup=0;failed=0
        shard=ROOT/f'batch_{cursor:012d}'
        archive=zipfile.ZipFile(str(shard)+'.zip.tmp','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)
        db.execute('BEGIN')
        for j in valid:
            a,b,c=map(int,definitions[j]);counts=scores[j]
            n=int(counts[0]);rate=float(counts[1]/n) if n else np.nan
            decision='watch_50_to_53' if n and rate>=.5 else 'failed'
            record={'rule_rank':int(ranks[j]),'a':a,'b':b,'c':c,'is_events':n,'is_hits':int(counts[1]),'validation_events':int(counts[7]),'validation_hits':int(counts[8]),'is_p70_hits':int(counts[2]),'validation_p70_hits':int(counts[9]),'is_range70_hits':int(counts[3]),'validation_range70_hits':int(counts[10]),'is_mean50_hits':int(counts[4]),'validation_mean50_hits':int(counts[11]),'is_mean90_hits':int(counts[5]),'validation_mean90_hits':int(counts[12]),'is_mean100_hits':int(counts[6]),'validation_mean100_hits':int(counts[13])}
            if n and rate>.53:
                signal=atoms[a]&atoms[b]&atoms[c]
                signature=(signal&masks[0]).tobytes()
                digest=hashlib.blake2b(signature,digest_size=16).digest()
                existing=db.execute('SELECT a,b,c FROM signatures WHERE hash=?',(digest,)).fetchone()
                if existing:
                    representative=atoms[existing[0]]&atoms[existing[1]]&atoms[existing[2]]&masks[0]
                    if representative.tobytes()!=signature:raise RuntimeError('Hash collision requires expanded key')
                    decision='duplicate_is_stream';dup+=1
                else:
                    db.execute('INSERT INTO signatures VALUES (?,?,?,?)',(digest,a,b,c))
                    metric=execution_metrics(signal,stage,starts,ends,pnl,mae,mfe,days,target_hits)
                    oo=int(counts[7]);oosrate=float(counts[8]/oo) if oo else None
                    candidate={'id':f'NQ_{int(ranks[j]):012d}','definition':[registry.iloc[k].to_dict() for k in [a,b,c]],'is':{'events':n,'hits':int(counts[1]),'accuracy':rate,'tier':hit_tier(rate),'percentile70_hits':int(counts[2]),'full_range70_hits':int(counts[3]),'mean50_hits':int(counts[4]),'mean90_hits':int(counts[5]),'mean100_hits':int(counts[6])},'validation':{'events':oo,'hits':int(counts[8]),'accuracy':oosrate,'tier':hit_tier(oosrate) if oosrate is not None else 'no_events','percentile70_hits':int(counts[9]),'full_range70_hits':int(counts[10]),'mean50_hits':int(counts[11]),'mean90_hits':int(counts[12]),'mean100_hits':int(counts[13])},'evidence':'adequate_raw_support' if n>=100 and oo>=50 else 'insufficient_sample_support','execution_metrics':{},'confirmation':'not_evaluated','roll_provenance':'unverified'}
                    for k,label in enumerate(['is','validation']):
                        x=metric[k];candidate['execution_metrics'][label]={'selected_entries':int(x[0]),'unresolved':int(x[1]),'observed_net_dollars':float(x[2]),'max_closed_trade_drawdown_dollars':float(x[3]),'gross_positive_net_dollars':float(x[4]),'gross_negative_net_dollars':float(x[5]),'full_hour_mae_sum_points':float(x[6]),'full_hour_mfe_sum_points':float(x[7]),'sessions_with_entry':int(x[8]),'candidate_sessions':int(candidate_days[k]),'mae_observations':int(x[9]),'mfe_observations':int(x[10]),'mean_full_hour_mae_points':float(x[6]/x[9]) if x[9] else None,'mean_full_hour_mfe_points':float(x[7]/x[10]) if x[10] else None,'profit_factor':float(x[4]/-x[5]) if x[5]<0 else None,'net_win_rate':float(x[11]/(x[0]-x[1])) if x[0]>x[1] else None,'mean_net_dollars':float(x[2]/(x[0]-x[1])) if x[0]>x[1] else None,'net_dollars_std':float(max(0,x[12]/(x[0]-x[1])-(x[2]/(x[0]-x[1]))**2)**.5) if x[0]>x[1] else None,'worst_trade_dollars':float(x[13]),'best_trade_dollars':float(x[14]),'maximum_full_hour_mae_points':float(x[17]),'maximum_full_hour_mfe_points':float(x[18]),'mean_holding_minutes':float(x[19]/(x[0]-x[1])) if x[0]>x[1] else None,'nonoverlapping_target_hit_rate':float(x[20]/(x[0]-x[1])) if x[0]>x[1] else None}
                    candidate['evidence']='adequate_raw_support' if n>=100 and oo>=50 and metric[0,8]>=30 and metric[1,8]>=20 else 'insufficient_sample_or_session_support'
                    candidate['is_unmatched_uplift']=rate-baselines[0]
                    candidate['validation_unmatched_uplift']=oosrate-baselines[1] if oosrate is not None else None
                    candidate['ordinary_is_accuracy']=baselines[0]
                    candidate['ordinary_validation_accuracy']=baselines[1]
                    folder='passed_validation' if oosrate is not None and oosrate>.53 and oo>=50 else ('validation_failed' if oosrate is not None and oo>=50 else 'insufficient_validation')
                    candidate['validation_status']=folder
                    archive.writestr(folder+'/'+candidate['id']+'.json',json.dumps(candidate,separators=(',',':'),allow_nan=False))
                    decision='saved_'+candidate['is']['tier'];saved+=1
            else:failed+=1
            record['decision']=decision;records.append(record)
        archive.close()
        Path(str(shard)+'.zip.tmp').replace(Path(str(shard)+'.zip'))
        table=pd.DataFrame(records)
        table.to_parquet(str(shard)+'.parquet',index=False)
        rejected=ROOT/'rejected';rejected.mkdir(exist_ok=True)
        table.loc[~table.decision.str.startswith('saved_')].to_parquet(rejected/(shard.name+'.parquet'),index=False)
        state.update({'cursor':cursor+size,'definitions_tested':state['definitions_tested']+len(valid),'saved_unique_signals':state['saved_unique_signals']+saved,'duplicate_is_streams':state['duplicate_is_streams']+dup,'failed_save_cutoff':state['failed_save_cutoff']+failed,'target_definitions':target,'updated_utc':datetime.now(UTC).isoformat(),'status':'running','threads':4})
        db.execute('INSERT OR REPLACE INTO state VALUES (1,?)',(json.dumps(state),));db.commit()
        elapsed=time.monotonic()-began;state['definitions_per_second_this_run']=(state['definitions_tested']-initial)/max(elapsed,.001)
        (ROOT/'progress.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state),flush=True)
    if state['definitions_tested']>=target:state['status']='discovery_complete_final_confirmation_pending'
    elif state['cursor']>=total:state['status']='canonical_search_space_exhausted'
    (ROOT/'progress.json').write_text(json.dumps(state,indent=2));db.close();lease.close()
    print('STOP',json.dumps(state),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=int,default=10_000_000);parser.add_argument('--batch-size',type=int,default=2000)
    args=parser.parse_args();run(args.target,args.batch_size)
