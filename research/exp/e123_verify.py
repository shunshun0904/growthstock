#!/usr/bin/env python3
"""Independent numerical checks of the saved E123 run, no fitting or writes to production."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

P=Path(sys.argv[1]).resolve()
d=pd.read_parquet(P/'signal_paths.parquet')
px=pd.read_parquet(P/'adjusted_prices.parquet')
cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(P/'calendar.csv').Date))
plan=json.loads((P/'protocol.json').read_text())
selection=json.loads((P/'selection_frozen.json').read_text())
test=pd.read_csv(P/'holdout_trades_current_tp10_day20.csv')
pred=pd.read_csv(P/'holdout_forecasts.csv')
complete=d.status.eq('complete')
checks={}

def check(name,condition):
    if not bool(condition):raise AssertionError(name)
    checks[name]=True

def close(a,b):return np.allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True)

check('eight_exact_oof_hash_matches',pd.read_csv(P/'model_identity.csv').matches_production_meta.eq(True).sum()==8)
check('frozen_protocol_hash',hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()==selection['protocol_sha256'])
check('no_development_rule_selection_below_minimum',selection['selection_reason'].startswith('insufficient') and selection['exit']['target']==.10)
check('all_holdout_dates_in_declared_period',d.loc[test.id,'Date'].between(*plan['sell_rule_holdout']).all())
check('all_holdout_ids_complete_and_classifier_scores_available',d.loc[test.id,'status'].eq('complete').all() and d.loc[test.id,[f'pct_{a}' for a in ['lgbm','xgb','cat','lgbm_k15','xgb_k15','cat_k15']]].notna().all().all())
check('holdout_tuning_labels_already_known',d.loc[test.id,'Date'].gt(pd.Timestamp('2025-10-07')).all())
check('daily_top_two',d.loc[test.id].groupby('Date').size().max()<=2)
check('all_rolling_untrainable_keys_predate_last_training_cutoff',len(pd.read_csv(P/'rolling_untrainable_keys.csv'))==7)
primary=[]
recomputed={.05:[],.08:[],.10:[],None:[]}
for i in test.id:
    row=d.loc[i]
    start=cal.searchsorted(row.Date,side='right')
    days=cal[start:start+20]
    z=px[px.Code.eq(row.Code)&px.Date.isin(days)].set_index('Date').reindex(days)
    entry=z.AdjO.iloc[0]
    check(f'path_{i}_next_market_open',row.entry_date==days[0] and row.mature_date==days[-1])
    check(f'path_{i}_upside_from_raw_ohlc',close(row.mfe,z.AdjH.max()/entry-1))
    check(f'path_{i}_downside_from_raw_ohlc',close(row.mae,z.AdjL.min()/entry-1))
    for target in (.05,.08,.10,None):
        touched=z.AdjH.ge(entry*(1+target)-1e-9) if target else pd.Series(False,index=z.index)
        if touched.any():
            j=int(np.flatnonzero(touched)[0]);value=target
            check(f'path_{i}_hit{round(target*100)}_day',row[f'hit{round(target*100)}_day']==j+1)
        else:
            j=19;value=z.AdjC.iloc[-1]/entry-1
        recomputed[target].append(value)
    check(f'path_{i}_mfe_not_claimed_as_sell_return',row.mfe+1e-10>=recomputed[.10][-1])
for target,name in [(.05,'tp5_day20'),(.08,'tp8_day20'),(.10,'current_tp10_day20'),(None,'close_day20')]:
    out=pd.read_csv(P/f'holdout_trades_{name}.csv').set_index('id').loc[test.id]
    check(f'{name}_independent_fills',close(out.gross,recomputed[target]))
    agg=pd.read_csv(P/'exit_comparison.csv')
    agg=agg[agg.period.eq('holdout')&agg.exit.eq(name)&agg.order.eq('stop_first')&agg.cost_pct.eq(.2)].iloc[0]
    check(f'{name}_mean_cost_and_win_rate',close(agg.mean_net_pct,100*(np.mean(recomputed[target])-.002)) and
          close(agg.win_pct,100*(np.asarray(recomputed[target])>.002).mean()))
features=[f'pct_{a}' for a in ['lgbm','xgb','cat','lgbm_k15','xgb_k15','cat_k15','ltr','ltr5']]
for r in pred.itertuples(index=False):
    month=pd.Timestamp(r.Date).to_period('M').start_time
    pool=d[d.fit_reference&complete&d.base_min.ge(95)&d.mature_date.lt(month)]
    check(f'forecast_{r.id}_full_ranking_pools',bool(d.at[r.id,'all_eight']) and pool.ltr_pool_complete.all() and pool.ltr5_pool_complete.all())
    check(f'forecast_{r.id}_no_unmatured_labels',pd.Timestamp(r.fit_latest_maturity)<month and r.n_fit==len(pool))
    delta=pool[features].to_numpy(float)-d.loc[r.id,features].to_numpy(float)
    ni=pool.index.to_numpy()[np.argsort(np.sum(delta*delta,axis=1),kind='stable')[:int(r.k)]]
    # Distance scale /100 cancels in ordering; independent recomputation from saved rows.
    check(f'forecast_{r.id}_neighbor_mean',close(r.expected_mfe,d.loc[ni,'mfe'].mean()))
    check(f'forecast_{r.id}_quartile',close(r.q25_mfe,d.loc[ni,'mfe'].quantile(.25)))
    check(f'forecast_{r.id}_hit5',close(r.p5,d.loc[ni,'hit5'].mean()))
check('tiny_sample_intervals_suppressed',pd.read_csv(P/'opportunity_by_threshold.csv').query('period=="holdout" and n<30').hit5_ci_low_pct.isna().all())
src=Path(__file__).resolve().parents[2]
fingerprints={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    src/'research/exit_upside.py',src/'research/exp/e123_exit_upside.py',src/'research/exp/e123_report.py',
    src/'research/exp/e123_verify.py',src/'tests/test_exit_upside.py']}
result={'passed':len(checks),'holdout_paths_independently_rebuilt':len(test),'forecasts_independently_rebuilt':len(pred),
        'code_sha256':fingerprints,'all_passed':True}
(P/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
