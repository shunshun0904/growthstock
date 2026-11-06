#!/usr/bin/env python3
"""Independent, real-data timing, prefix, fit-isolation and fill checks for E126."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
parser.add_argument('--source',type=Path,default=Path('/workspace/scratch/daily-arrival-20261105/results'))
parser.add_argument('--prices',type=Path,default=Path('/workspace/scratch/exit-upside-20261105/results'))
args=parser.parse_args();P=args.output;SOURCE=args.source;PRICES=args.prices
RESEARCH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RESEARCH));import optimal_exit as O

def main():
    ep=pd.read_parquet(P/'episodes.parquet');q=pd.read_parquet(P/'outer_states.parquet')
    outer=pd.read_parquet(P/'outer_episodes.parquet');pred=pd.read_parquet(P/'predictions.parquet')
    exits=pd.read_parquet(P/'exits.parquet');assign=pd.read_parquet(P/'prior_assignments.parquet')
    plan=json.loads((P/'protocol.json').read_text());sel=json.loads((P/'selected.json').read_text())
    audit=json.loads((P/'data_audit.json').read_text())
    fits=json.loads((P/'outer_fits.json').read_text())
    cal=pd.DatetimeIndex(pd.read_csv(SOURCE/'calendar.csv',parse_dates=['Date']).Date)
    with np.load(P/'paths.npz') as d:paths={k:d[k] for k in d.files}
    checks={}
    def check(name,condition):
        if not bool(condition):raise AssertionError(name)
        checks[name]=True
    def close(x,y):return np.allclose(x,y,atol=1e-9,rtol=1e-7,equal_nan=True)
    check('broad_common_cohort',len(ep)==15506 and len(outer)==3265 and len(q)==62035)
    check('nineteen_daily_states',q.groupby('trade').size().eq(19).all() and not q.duplicated(['trade','elapsed']).any())
    check('original_twenty_session_deadline',(q.elapsed+q.remaining).eq(20).all())
    check('no_whole_trade_exclusions_in_outer',np.array_equal(outer.trade,ep.loc[ep.signal_date.between(*plan['outer_signal_dates']),'trade']))
    check('frozen_protocol',hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()==audit['protocol_sha256']==sel['protocol_sha256'])
    check('zero_costs',all(x==0 for x in plan['costs'].values()))
    for row in pd.read_csv(P/'model_identity.csv').itertuples():
        f=RESEARCH/'model'/('meta.json' if row.model=='lgbm' else f'models/{row.model}/meta.json')
        check('current_model_identity_'+row.model,json.loads(f.read_text())['artifactHashes']['oof.parquet']==row.sha256)
    index=ep.set_index('trade').pidx
    for trade,z in q.groupby('trade',sort=False):
        i=int(index.loc[trade]);k=z.elapsed.to_numpy()
        c=paths['c'][i];h=paths['h'][i];days=paths['dates'][i]
        pos=cal.searchsorted(z.iloc[0].signal_date,side='right')
        check(f'next_official_session_and_deadline_{trade}',np.array_equal(days,cal[pos:pos+20].to_numpy()))
        check(f'known_state_{trade}',close(z.spot_return,c[:19]) and close(z.known_peak,np.maximum.accumulate(h)[:19]))
        check(f'asof_{trade}',np.array_equal(z.asof_date.to_numpy(),days[:19]))
        future=np.array([max(h[j:]) for j in k])
        check(f'future_max_{trade}',close(z.actual_max_spot,(1+future)/(1+c[:19])))
        for t in [10,15]:
            risk=np.maximum.accumulate(h)[:19]<t/100-1e-12
            check(f'arrival_labels_{trade}_{t}',np.array_equal(z[f'risk{t}'],risk) and np.array_equal(z[f'y{t}'],future>=t/100-1e-12))
    # Independent prefix windows, one check per code; all 62,035 outer states.
    stock=pd.read_parquet(PRICES/'adjusted_prices.parquet',columns=['Code','Date','AdjC'])
    groups=q.groupby('Code',sort=False).groups;vol_error=0.
    for code,frame in stock.groupby('Code',sort=False):
        if code not in groups:continue
        z=q.loc[groups[code]]
        price=frame.set_index('Date').AdjC.reindex(cal).to_numpy(float)
        returns=np.diff(np.log(price),prepend=np.nan)
        good=True
        for w,minimum in [(20,16),(60,40)]:
            expected=[]
            for date in z.asof_date:
                j=cal.get_loc(date);past=returns[max(0,j-w+1):j+1];valid=past[np.isfinite(past)]
                expected.append(np.std(valid,ddof=1) if len(valid)>=minimum else np.nan)
            actual=z[f'sigma{w}'].to_numpy();expected=np.array(expected)
            good &=close(actual,expected)
            if np.isfinite(actual).any():vol_error=max(vol_error,float(np.nanmax(abs(actual-expected))))
        check('known_volatility_'+str(code),good)
    scorecols=['score_lgbm','score_xgb','score_cat']
    # Independently reconstruct every fit, including winsorization, date weights,
    # causal score calibration and band shrinkage; no runner fitting functions.
    records={}
    for rec in sel['fit_audits']+fits:
        cutoff=pd.Timestamp(rec['cutoff']);train=ep[ep.mature_date.lt(cutoff)].copy()
        check('purged_fit_'+rec['cutoff'],len(train)==rec['train_n'] and str(train.mature_date.max())==rec['last_mature'])
        counts=train.groupby('signal_date').size();weight=train.signal_date.map(1/counts).to_numpy();weight/=weight.mean()
        low,high=np.percentile(train.outcome_delta,[1,99]);target=np.clip(train.outcome_delta,low,high)
        mean=np.sum(weight*target)/np.sum(weight)
        check('pooled_prior_'+rec['cutoff'],close(mean,rec['pooled']))
        cdfs=np.column_stack([100*np.searchsorted(np.sort(train[c]),train[c],side='left')/len(train) for c in scorecols]).min(axis=1)
        bands=np.array(plan['estimation']['score_bands']);group=np.searchsorted(bands,cdfs,side='right')-1
        means=[]
        for band in range(len(bands)-1):
            which=group==band
            means.append((np.sum(weight[which]*target.to_numpy()[which])+100*mean)/(np.sum(weight[which])+100))
        check('band_prior_'+rec['cutoff'],close(means,rec['bands']))
        records[rec['cutoff']]=(train,rec)
    for cutoff,z in assign.groupby('prior_cutoff',sort=False):
        train,rec=records[str(cutoff.date())]
        entries=outer.set_index('trade').loc[z.trade]
        pct=np.column_stack([100*np.searchsorted(np.sort(train[c]),entries[c],side='left')/len(train) for c in scorecols]).min(axis=1)
        check('causal_entry_cdf_'+str(cutoff.date()),close(pct,z.causal_percentile) and entries.signal_date.ge(cutoff).all())
        for method,rows in z.groupby('method',sort=False):
            config=sel['selected'][method];band=np.searchsorted(plan['estimation']['score_bands'],rows.causal_percentile,side='right')-1
            expected=np.full(len(rows),rec['pooled']) if config['prior_type']=='pooled' else np.array(rec['bands'])[band]
            check('prior_assignment_'+str(cutoff.date())+method,close(rows.prior_delta,expected))
    fdict={}
    for method,f in pred.groupby('method',sort=False):
        f=f.sort_values('qid').reset_index(drop=True);fdict[method]=f
        config=sel['selected'][method];k=config['strength']
        expected=f.prior_delta if k=='infinite' else (k*f.prior_delta+q.log_gain)/(k+q.elapsed)
        check(method+'_daily_drift',close(expected,f.delta))
        check(method+'_entry_prior_frozen',f.groupby('trade').prior_delta.nunique().eq(1).all())
        check(method+'_no_future_fit',pd.to_datetime(f.prior_cutoff).le(q.signal_date).all())
        check(method+'_nested_arrival',f.p10.ge(f.p15-1e-12).all())
        check(method+'_ordered_quantiles',f.q10.le(f.q50).all() and f.q50.le(f.q90).all())
        check(method+'_positive_volatility',f.sigma.gt(0).all() and np.isfinite(f.sigma).all())
        check(method+'_finite_forecasts',np.isfinite(f[['p10','p15','max_mean','q10','q50','q90']]).all().all())
        decision=f.delta.le(0) if method=='drift_gate' else q.log_drawdown.ge(f.boundary)
        check(method+'_stopping_decision',np.array_equal(decision,f.decision))
        # Independent integration of the analytic survival curve on fixed probes.
        for j in np.linspace(0,len(f)-1,30,dtype=int):
            row=f.iloc[j];r=int(q.iloc[j].remaining);d=row.delta;s=row.sigma
            survival=lambda x:1-(ndtr((x-d*r)/(s*np.sqrt(r)))-np.exp(2*d*x/(s*s))*ndtr((-x-d*r)/(s*np.sqrt(r))))
            bound=max(d*r,0)+12*s*np.sqrt(r)+.1
            mean=1+quad(lambda x:np.exp(x)*survival(x),0,bound,epsabs=1e-8)[0]
            check(f'{method}_max_mean_integral_{j}',abs(mean-row.max_mean)<1e-6)
    # Sequential independent fills, including day1 targets and next-open gaps.
    eps=json.loads((P/'numerics.json').read_text())['log_boundary_error_bound']
    lookup={m:f.set_index(['trade','elapsed']) for m,f in fdict.items()}
    for row in exits.itertuples():
        i=int(index.loc[row.trade]);o,h,c=paths['o'][i],paths['h'][i],paths['c'][i]
        method=row.method;target=None;decision=np.zeros(19,bool)
        if method=='first_possible_next_open':decision[:]=True
        elif method.startswith('fixed_tp'):target=int(method.split('_')[1][2:])/100
        elif method=='tuned_fixed_trailing':
            trailing=sel['selected']['fixed_trailing']['threshold']
            decision=(1+c[:19])/(1+np.maximum.accumulate(h)[:19])<=1-trailing
        elif method.startswith('tuned_') or method.startswith('boundary_'):
            family=method[6:] if method.startswith('tuned_') else 'drawdown_boundary'
            state=lookup[family].loc[row.trade]
            if method.startswith('boundary_'):
                offset=eps if method.endswith('later') else -eps
                b=state.boundary.to_numpy();alpha=state.alpha.to_numpy()
                b=np.where((alpha>0)&(alpha<1),np.maximum(b+offset,0),b)
                decision=np.log((1+np.maximum.accumulate(h)[:19])/(1+c[:19]))>=b
            else:decision=state.decision.to_numpy()
        for day in range(20):
            if day>0 and decision[day-1]:value=o[day];reason='next_open';break
            if target is not None and h[day]>=target-1e-12:
                value=target if method.endswith('limit') else max(target,o[day]);reason='target';break
            if day==19:value=c[day];reason='deadline';break
        check(f'fill_{row.Index}',close(value,row.gross) and row.exit_day==day+1 and row.reason==reason)
    check('common_exit_cohort',exits.groupby('method').trade.nunique().eq(3265).all())
    check('tiny_cohort_no_intervals',pd.read_csv(P/'paired_returns.csv').query("cohort=='base99'")[['ci_low','ci_high']].isna().all().all())
    # Fixed-parameter Brownian-bridge experiment: verify V1, not profit optimality.
    rng=np.random.default_rng(126);n=100000;s=.05;alpha=.5
    step=s*rng.standard_normal((n,20));level=np.column_stack([np.zeros(n),np.cumsum(step,axis=1)])
    high=level[:,:-1]+(step+np.sqrt(step*step-2*s*s*np.log(rng.random((n,20)))))/2
    peak=np.maximum.accumulate(high,axis=1);boundary,_=O.solve_boundary(s,alpha,hold=20,points=1601)
    trigger=peak[:,:19]-level[:,1:20]>=boundary[np.arange(19,0,-1)]
    first=np.where(trigger.any(axis=1),trigger.argmax(axis=1)+1,20)
    sale=level[np.arange(n),first];maximum=high.max(axis=1);ratio=np.exp(maximum-sale)
    synthetic=[]
    for name,control in [('hold20',np.exp(maximum-level[:,-1])),('immediate',np.exp(maximum))]:
        difference=ratio-control;se=difference.std(ddof=1)/np.sqrt(n)
        check('known_GBM_V1_improvement_'+name,difference.mean()+5*se<0)
        synthetic.append(dict(control=name,n=n,sigma=s,alpha=alpha,policy_ratio=ratio.mean(),
                control_ratio=control.mean(),paired_difference=difference.mean(),se=se))
    pd.DataFrame(synthetic).to_csv(P/'synthetic_gbm_check.csv',index=False)
    result=dict(all_passed=True,checks=len(checks),episodes=len(outer),daily_states=len(q),exit_cases=len(exits),
                methods=exits.method.nunique(),max_volatility_error=vol_error,
                independence_unit='Episodes and 21-market-session signal-date blocks; daily states and synthetic paths are not trade sample size')
    (P/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
