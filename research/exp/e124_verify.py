#!/usr/bin/env python3
"""Independent E124 data/forecast checks; no fitting and no production writes."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('output',type=Path);ap.add_argument('source',type=Path)
ap.add_argument('--phase',choices=['data','results','all'],default='all')
a=ap.parse_args();P=a.output;S=a.source
d=pd.read_parquet(P/'landmarks.parquet')
paths=np.load(S/'paths.npz')
checks={}
def check(name,value):
    if not bool(value):raise AssertionError(name)
    checks[name]=True
def close(x,y):return np.allclose(x,y,atol=2e-7,rtol=2e-6,equal_nan=True)


if a.phase in ['data','all']:
    features=json.loads((P/'feature_contract.json').read_text())
    check('same20_scheduled_landmarks_each_trade',d.groupby('trade').size().eq(20).all())
    check('original20_session_deadline',d.elapsed.add(d.remaining).eq(20).all())
    check('unique_trade_landmarks',not d.duplicated(['trade','elapsed']).any())
    check('no_future_labels_in_features',not any(c.startswith(('y','future','gain','comp','risk','arrived')) for c in features['daily_state']))
    check('same_entry_features_in_both_arms',set(features['static_age']).issubset(features['daily_state']))
    selected=d[d.strict&d.signal_date.between('2025-10-08','2026-10-05')].trade.unique()
    # Real prices, independent prefix and suffix calculations for every strict trade.
    z=d[d.trade.isin(selected)].set_index(['trade','elapsed'])
    for trade in selected:
        h=paths['h'][trade];l=paths['l'][trade];c=paths['c'][trade];o=paths['o'][trade]
        for t in (0,1,3,7,12,19):
            r=z.loc[(trade,t)]
            spot=0 if t==0 else c[t-1]
            peak=0 if t==0 else max(h[:t]);low=0 if t==0 else min(l[:t])
            check(f'{trade}_{t}_prefix_price',close(r.spot_return,spot) and close(r.known_peak,peak) and close(r.known_low,low))
            check(f'{trade}_{t}_asof_session',r.asof_date==pd.Timestamp(paths['dates'][trade,max(t-1,0)]))
            check(f'{trade}_{t}_future_high',close(r.future_max_from_spot,(1+max(h[t:]))/(1+spot)-1))
            for target in (10,15):
                level=target/100
                arrived=bool(t and max(h[:t])>=level-1e-12)
                check(f'{trade}_{t}_{target}_fixed_target',r[f'arrived{target}']==arrived and r[f'y{target}']==bool(max(h[t:])>=level-1e-12))
                check(f'{trade}_{t}_{target}_close_target',r[f'y_close{target}']==bool(max(c[t:])>=level-1e-12))
                outcome=0;held=c[-1]
                for day in range(t,20):
                    tp=h[day]>=level-1e-12;sl=l[day]<=-.05+1e-12
                    if tp or sl:
                        outcome=1 if tp and (not sl or o[day]>=level-1e-12) else 2
                        break
                check(f'{trade}_{t}_{target}_competing_order',r[f'comp{target}']==outcome)
                for day in range(t,20):
                    if h[day]>=level-1e-12:held=level;break
                check(f'{trade}_{t}_{target}_gain_next_open',close(r[f'gain{target}'],held-min(o[t],level)))
    # Technical state0 must be exactly the known entry information, not day1 close.
    initial=d[d.elapsed.eq(0)]
    for col in [c for c in features['static_age'] if c.startswith('entry_tech_')]:
        now=col.replace('entry_tech_','now_tech_',1)
        check('open_known_'+col,close(initial[col],initial[now]))
    result={'all_passed':True,'checks':len(checks),'strict_trades_independently_checked':len(selected),
            'landmarks_checked_per_trade':6,'source_input_sha256':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['paths.npz','signal_paths.parquet']}}
    (P/'data_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if a.phase in ['results','all']:
    checks={}
    pred=pd.read_parquet(P/'outer_predictions.parquet')
    frozen=json.loads((P/'selection_frozen.json').read_text())
    check('frozen_protocol_identity',frozen['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest())
    check('unique_forecasts',not pred.duplicated(['trade','elapsed']).any())
    check('all20_forecast_slots',pred.groupby('trade').size().eq(20).all())
    for arm in ['static_age','daily_state']:
        p10=pred['hit10_'+arm];p15=pred['hit15_'+arm]
        check(arm+'_probability_range',p10.between(0,1).all() and p15.between(0,1).all())
        check(arm+'_nested_targets',p10.ge(p15-1e-12).all())
        for target in [10,15]:check(arm+str(target)+'_arrived_absorbing',pred.loc[pred[f'arrived{target}'],f'hit{target}_'+arm].eq(1).all())
    for r in pd.read_csv(P/'outer_maturity_checks.csv').itertuples(index=False):
        cutoff=pd.Timestamp(r.cutoff)
        tr=d[d.mature_date.lt(cutoff)]
        te=pred[pred.asof_date.dt.to_period('M').astype(str).eq(r.month)]
        check(r.month+'_train_matured',tr.mature_date.max()==pd.Timestamp(r.last_train_label) and tr.mature_date.max()<cutoff)
        check(r.month+'_no_trade_overlap',set(tr.trade).isdisjoint(te.trade))
    metrics=pd.read_csv(P/'probability_metrics.csv')
    for r in metrics.itertuples(index=False):
        z=pred if r.cohort=='broad' else pred[pred.strict]
        z=z[z[f'risk{r.target}']&z.elapsed.ge(1)]
        col=f'hit{r.target}_{r.arm}' if r.arm in ['static_age','daily_state'] else f'{r.arm}{r.target}'
        prob=z[col].to_numpy();y=z[f'y{r.target}'].to_numpy(float)
        check(f'metric_{r.cohort}_{r.target}_{r.arm}',close(r.brier,np.mean((prob-y)**2)) and r.trades==z.trade.nunique() and r.landmarks==len(z))
    # Independent execution loop: yesterday's decision, today's open, then high.
    strict=pred[pred.strict]
    for target in [10,15]:
        level=target/100
        for arm in ['static_age','daily_state']:
            cut=frozen['selections'][f'gain{target}_{arm}']['exit_cutoff']
            trades=pd.read_csv(P/f'exit_trades_{target}_{arm}.csv').set_index('trade')
            for trade,g in strict.groupby('trade'):
                gains=g.set_index('elapsed')[f'gain{target}_{arm}']
                for day in range(20):
                    if day and gains.loc[day]<cut:
                        value=min(paths['o'][trade,day],level);reason='daily_exit';break
                    if paths['h'][trade,day]>=level-1e-12:
                        value=level;reason='target';break
                    if day==19:
                        value=paths['c'][trade,day];reason='deadline';break
                check(f'exit_{target}_{arm}_{trade}',close(trades.at[trade,'gross'],value) and trades.at[trade,'exit_day']==day+1 and trades.at[trade,'reason']==reason)
    tracker=pd.read_csv(P/'daily_tracker.csv')
    check('tracker_has_no_future_actual_labels',not set(tracker.columns)&{'y10','y15','y_close10','y_close15','gain10','gain15','future_max_from_spot'})
    check('tracker21_slots_including_deadline',tracker.groupby('trade').size().eq(21).all())
    for q in [.25,.5,.75]:
        check('quantile_to_purchase_'+str(q),close(tracker[f'future_high_q{q}_from_entry'],(1+tracker.spot_return)*(1+tracker[f'future_high_q{q}_from_spot'])-1))
    check('quantiles_ordered',(pred['quantile0.25_daily_state']<=pred['quantile0.5_daily_state']).all() and (pred['quantile0.5_daily_state']<=pred['quantile0.75_daily_state']).all())
    repo=Path(__file__).resolve().parents[2]
    files=['research/daily_arrival.py','research/exp/e124_daily_arrival.py','research/exp/e124_verify.py','tests/test_daily_arrival.py']
    if (repo/'research/exp/e124_report.py').exists():files.append('research/exp/e124_report.py')
    result={'all_passed':True,'checks':len(checks),'strict_trades':strict.trade.nunique(),'strict_landmarks':len(strict),
            'code_sha256':{n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in files}}
    (P/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
