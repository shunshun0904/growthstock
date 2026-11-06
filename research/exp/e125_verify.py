#!/usr/bin/env python3
"""Independent real-data verification of E125; never fits or changes forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('source',type=Path,help='E124 results')
    ap.add_argument('prices',type=Path,help='E123 results')
    a = ap.parse_args(); p = a.output
    q = pd.read_parquet(p/'queries.parquet')
    d = pd.read_parquet(p/'daily_tracker.parquet')
    context = np.load(p/'contexts.npy')
    audit = json.loads((p/'data_audit.json').read_text())
    inference = json.loads((p/'inference_audit.json').read_text())
    plan = json.loads((p/'protocol.json').read_text())
    cal = pd.DatetimeIndex(pd.read_csv(a.source/'calendar.csv',parse_dates=['Date']).Date)
    checks = {}
    def check(name, condition):
        if not bool(condition): raise AssertionError(name)
        checks[name] = True
    def close(x,y): return np.allclose(x,y,atol=3e-7,rtol=2e-6,equal_nan=True)
    check('common_cohort_60_trades_1140_states',q.trade.nunique()==60 and len(q)==1140)
    check('nineteen_states_per_trade',q.groupby('trade').size().eq(19).all())
    check('fixed_twenty_session_deadline',(q.elapsed+q.remaining).eq(20).all())
    check('no_duplicate_queries',not q.duplicated(['trade','elapsed']).any())
    check('all_methods_common_states',d.groupby('method').qid.nunique().eq(len(q)).all() and len(d)==3*len(q))
    check('original_weights_predate_all_signals',q.signal_date.min()>pd.Timestamp(plan['tsfm']['weight_publication']).tz_localize(None))
    check('frozen_protocol',hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest()==audit['protocol_sha256'])
    for name in ['queries.parquet','contexts.npy','protocol.json']:
        check('inference_input_hash_'+name,hashlib.sha256((p/name).read_bytes()).hexdigest()==inference[name])
    check('no_cross_learning_or_fine_tune',inference['cross_learning'] is False and inference['fine_tuned'] is False)
    check('individual_batch_isolation',inference['isolation_max_abs_log_error']<1e-4)
    prices = pd.read_parquet(a.prices/'adjusted_prices.parquet',filters=[('Code','in',q.Code.unique().tolist())])
    for code, group in q.groupby('Code'):
        frame = prices[prices.Code.eq(code)].set_index('Date').reindex(cal)
        h = frame.AdjH.to_numpy(float); c = frame.AdjC.to_numpy(float)
        for row in group.itertuples():
            j = cal.get_loc(row.asof_date); r = row.remaining
            # Independent NumPy windows over the past prefix, never future prices.
            past = h[:j+1]
            maxima = np.lib.stride_tricks.sliding_window_view(past,r).max(axis=1)
            expected = np.log(maxima[-256:]/c[j]).astype(np.float32)
            check(f'past_only_context_{row.qid}',close(expected,context[row.qid]))
            check(f'original_deadline_{row.qid}',cal[j+r]==row.mature_date)
            future_high = max(h[j+1:j+r+1])
            check(f'future_max_target_{row.qid}',close(row.actual_max_spot,future_high/c[j]) and close(row.actual_max_buy,future_high/row.buy))
            # All return volatility uses known daily bars only.
            gap = np.log(frame.AdjO / frame.AdjC.shift(1)).iloc[max(0,j-59):j+1]
            park = (np.log(frame.AdjH/frame.AdjL)**2/(4*np.log(2))).iloc[max(0,j-59):j+1]
            check(f'past_only_volatility_{row.qid}',close(row.sigma_gap,np.nanstd(gap,ddof=1)) and close(row.sigma_intraday,np.sqrt(np.nanmean(park))))
    source = pd.read_parquet(a.source/'landmarks.parquet',columns=['trade','elapsed','strict','mature_date','signal_date','risk10','risk15','y10','y15'])
    prior = source[source.strict & source.elapsed.ge(1) & source.mature_date.lt(plan['outer_signal_dates'][0])]
    check('prior_only_215_selected_matured_trades',prior.trade.nunique()==215 and set(prior.trade).isdisjoint(q.trade))
    check('prior_matured_before_every_forecast',prior.mature_date.max()<q.asof_date.min())
    for row in pd.read_csv(p/'age_prior.csv').itertuples():
        z = prior[prior.elapsed.eq(row.elapsed)&prior[f'risk{row.target}']]
        check(f'age_prior_{row.elapsed}_{row.target}',row.n==len(z) and row.hits==z[f'y{row.target}'].sum() and close(row.p,(row.hits+1)/(row.n+2)))
    for method in ['chronos2','volatility']:
        z = d[d.method.eq(method)]
        check(method+'_nested_targets',z.p10.ge(z.p15-1e-12).all())
        check(method+'_ordered_quantiles',(np.diff(z[['q10','q25','q50','q75','q90']].to_numpy(),axis=1)>=0).all())
    with np.load(p/'chronos_quantiles.npz') as model:
        raw = model['raw']; levels = model['levels']
        check('every_forecast_cached',model['done'].all() and np.isfinite(raw).all())
    tsfm = d[d.method.eq('chronos2')].set_index('qid')
    # Independent numerical integration of the complete approximate quantile function.
    u = np.linspace(0,1,20001)
    for i in range(len(q)):
        values = np.sort(np.exp(raw[i]))
        grid = np.interp(u,levels,values)
        mean = np.trapezoid(grid,u); sd = np.sqrt(max(np.trapezoid(grid*grid,u)-mean*mean,0))
        check(f'quantile_moments_{i}',abs(mean-tsfm.at[i,'max_mean'])<1e-6 and abs(sd-tsfm.at[i,'max_sd'])<1e-5)
        for t in (10,15):
            threshold = q.iloc[i].buy/q.iloc[i].close*(1+t/100)
            approximation = (grid>=threshold).mean()
            check(f'quantile_arrival_{i}_{t}',abs(approximation-tsfm.at[i,f'p{t}'])<1e-4)
    for row in pd.read_csv(p/'scores.csv').itertuples():
        z = d[d.method.eq(row.method)&d[f'risk{row.target}']]
        errors = (z[f'p{row.target}']-z[f'y{row.target}'])**2
        check(f'scores_{row.method}_{row.target}',close(errors.mean(),row.brier) and row.trades==z.trade.nunique() and row.states==len(z))
    paths = np.load(p/'selected_paths.npz')
    path_id = {v:i for i,v in enumerate(paths['trade'])}
    exits = pd.read_parquet(p/'exits.parquet')
    check('all_policy_purchases_including_buy_day_winners',exits.groupby(['target','profit_gap','method','rule']).size().eq(60).all())
    for row in exits.itertuples():
        i = path_id[row.trade]; level = row.target/100
        state = d[d.trade.eq(row.trade)&d.method.eq(row.method)].sort_values('elapsed')
        ps = state[f'p{row.target}'].to_numpy()
        for k in range(20):
            trigger = False
            if row.method!='fixed' and k>=1:
                trigger = ps[k-1]<.2 if row.rule=='absolute' else k>=2 and ps[k-1]<ps[0]/2
            if trigger:
                value = paths['o'][i,k]
                if row.profit_gap=='limit': value=min(value,level)
                reason = 'probability_exit'; break
            if paths['h'][i,k]>=level-1e-12:
                value = level if row.profit_gap=='limit' else max(level,paths['o'][i,k])
                reason = 'target'; break
            if k==19: value=paths['c'][i,k]; reason='deadline'; break
        check(f'next_open_exit_{row.Index}',close(row.gross,value) and row.exit_day==k+1 and row.reason==reason and close(row.net,value-.002))
    for row in pd.read_csv(p/'policies.csv').itertuples():
        z = exits[exits.target.eq(row.target)&exits.profit_gap.eq(row.profit_gap)&exits.method.eq(row.method)&exits.rule.eq(row.rule)]
        check(f'policy_summary_{row.target}_{row.profit_gap}_{row.method}_{row.rule}',close(z.net.mean(),row.net) and len(z)==row.trades)
    result = dict(all_passed=True,checks=len(checks),trades=60,states=1140,exit_cases=len(exits),
                  methods=3,tsfm_input='past high rolling maxima only; no labels, scores or future covariates',
                  independence_unit='trade and 21-session signal-date blocks, never simulated paths or daily state count')
    (p/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
