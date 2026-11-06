#!/usr/bin/env python3
"""E126: broad-universe, tuned optimal-stopping exit research. No production writes.

Run prepare, grid, tune, outer, evaluate. Row-level vendor data stay in --output.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

RESEARCH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RESEARCH))
import optimal_exit as O
from exit_upside import historical_percentile, block_interval

PROTOCOL=RESEARCH.parent/'docs/experiments/optimal-stopping-2026-11-06/protocol.json'
SCORES=['score_lgbm','score_xgb','score_cat']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):
    def convert(v):
        if isinstance(v,(pd.Timestamp,np.datetime64)):return str(pd.Timestamp(v))
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=convert)+'\n')


def prepare(a,plan):
    source=pd.read_parquet(a.source/'source_signals.parquet')
    ep=source[source.fit_reference & source.status.eq('complete')].copy()
    ep['trade']=ep.index
    ep=ep[['trade','Code','Date','entry_date','mature_date','entry_adjusted','base_min']+SCORES]
    ep=ep.rename(columns={'Date':'signal_date','entry_adjusted':'buy'}).sort_values('trade').reset_index(drop=True)
    assert len(ep)==plan['expected_episodes']
    ep['pidx']=np.arange(len(ep))
    with np.load(a.prices/'paths.npz') as data:
        paths={k:data[k][ep.trade.to_numpy()].copy() for k in ['o','h','l','c','dates']}
    assert all(np.isfinite(paths[k]).all() for k in ['o','h','l','c'])
    ep['outcome_delta']=np.log1p(paths['c'][:,-1])/20
    ep['episode_sigma']=np.std(np.diff(np.column_stack([np.zeros(len(ep)),np.log1p(paths['c'])]),axis=1),axis=1,ddof=1)
    ep['max20']=paths['h'].max(axis=1)
    cols=['trade','Code','signal_date','entry_date','mature_date','asof_date','elapsed','remaining',
          'spot_return','known_peak','future_max_from_spot','risk10','risk15','y10','y15']
    q=pd.read_parquet(a.source/'landmarks.parquet',columns=cols)
    q=q[q.elapsed.ge(1)].sort_values(['trade','elapsed']).reset_index(drop=True)
    assert len(q)==19*len(ep) and np.array_equal(q.trade.to_numpy(),np.repeat(ep.trade.to_numpy(),19))
    q=q.merge(ep[['trade','pidx','buy','base_min']],on='trade',validate='many_to_one',sort=False)
    q['elapsed']=q.elapsed.astype(int);q['remaining']=q.remaining.astype(int)
    # Replace float32 inherited state values by exact split-adjusted path ratios.
    q['spot_return']=paths['c'][:,:19].ravel()
    q['known_peak']=np.maximum.accumulate(paths['h'],axis=1)[:,:19].ravel()
    q['close']=q.buy*(1+q.spot_return)
    q['log_gain']=np.log1p(q.spot_return)
    q['log_drawdown']=np.log1p(q.known_peak)-q.log_gain
    q['actual_max_spot']=1+q.future_max_from_spot
    cal=pd.DatetimeIndex(pd.read_csv(a.source/'calendar.csv',parse_dates=['Date']).Date)
    stock=pd.read_parquet(a.prices/'adjusted_prices.parquet',columns=['Code','Date','AdjC'])
    q['sigma20']=np.nan;q['sigma60']=np.nan
    qgroups=q.groupby('Code',sort=False).groups
    max_spot_error=0.;n=0
    for code,frame in stock.groupby('Code',sort=False):
        if code not in qgroups:continue
        ix=qgroups[code]
        prices=frame.sort_values('Date').set_index('Date').AdjC.reindex(cal)
        ret=np.log(prices).diff()
        query_dates=pd.DatetimeIndex(q.loc[ix,'asof_date'])
        for w,minimum in [(20,16),(60,40)]:
            q.loc[ix,f'sigma{w}']=ret.rolling(w,min_periods=minimum).std(ddof=1).reindex(query_dates).to_numpy()
        err=np.max(abs(prices.reindex(query_dates).to_numpy()/q.loc[ix,'buy'].to_numpy()-1-q.loc[ix,'spot_return'].to_numpy()))
        max_spot_error=max(max_spot_error,float(err));n+=1
    assert n==ep.Code.nunique() and max_spot_error<1e-10
    assert np.array_equal(paths['dates'][:,:19].ravel(),q.asof_date.to_numpy())
    outer=ep[ep.signal_date.between(*plan['outer_signal_dates'])]
    prior=ep[ep.mature_date.lt(plan['outer_signal_dates'][0])]
    assert len(outer)==plan['expected_outer_episodes'] and len(prior)==plan['initial_matured_reference']
    ep.to_parquet(a.output/'episodes.parquet',index=False)
    q.to_parquet(a.output/'states.parquet',index=False)
    np.savez_compressed(a.output/'paths.npz',trade=ep.trade.to_numpy(),**paths)
    identity=pd.read_csv(a.source/'model_identity.csv')
    for row in identity.itertuples():
        file=RESEARCH/'model'/('meta.json' if row.model=='lgbm' else f'models/{row.model}/meta.json')
        assert json.loads(file.read_text())['artifactHashes']['oof.parquet']==row.sha256
    identity.to_csv(a.output/'model_identity.csv',index=False)
    (a.output/'protocol.json').write_bytes(PROTOCOL.read_bytes())
    counts=[]
    for name,mask in [('all',np.ones(len(ep),bool))]+[(f'base{x}',ep.base_min.ge(x)) for x in [95,97,99]]:
        for period,sel in [('history',np.ones(len(ep),bool)),('initial_fit',ep.index.isin(prior.index)),('outer',ep.index.isin(outer.index))]:
            z=ep[mask & sel]
            counts.append(dict(cohort=name,period=period,episodes=len(z),codes=z.Code.nunique(),signal_dates=z.signal_date.nunique()))
    pd.DataFrame(counts).to_csv(a.output/'cohort_counts.csv',index=False)
    audit=dict(protocol_sha256=sha(PROTOCOL),source_rows=len(source),episodes=len(ep),states=len(q),
               outer_episodes=len(outer),initial_fit=len(prior),outer_states=len(outer)*19,
               max_spot_error=max_spot_error,price_last=stock.Date.max(),
               sigma20_missing=int(q.sigma20.isna().sum()),sigma60_missing=int(q.sigma60.isna().sum()),
               source_hashes={str(f):sha(f) for f in [a.source/'source_signals.parquet',a.source/'landmarks.parquet',
                            a.source/'calendar.csv',a.source/'model_identity.csv',a.prices/'paths.npz',a.prices/'adjusted_prices.parquet']})
    dump(a.output/'data_audit.json',audit)
    print('PREPARED',json.dumps(audit,default=str),flush=True)


def grid(a,plan):
    n=plan['theory']['numerics'];sg=np.array(n['sigma_grid']);ag=np.array(n['alpha_grid'])
    start=time.time();out=np.empty((len(sg),len(ag),20));audits=[]
    for i,s in enumerate(sg):
        for j,alpha in enumerate(ag):
            b,audit=O.solve_boundary(s,alpha,points=n['grid_points'],ymax=n['normalized_drawdown_max'])
            assert np.isfinite(b).all(),(s,alpha,'censored boundary')
            out[i,j]=b
            audits.append(dict(sigma=s,alpha=alpha,non_upper_stop_nodes=audit['non_upper_stop_nodes'],b19=b[-1],
                               monotonic_remaining=bool((np.diff(b)>=-1e-8).all())))
        print('GRID',i+1,'/',len(sg),'seconds',round(time.time()-start,1),flush=True)
    convergence=[]
    # Representative fixed points include both interior and sensitive endpoints.
    for s,alpha in [(0.0025,.001),(.01,.1),(.03,.5),(.08,.9),(.2,.99),(.03,.999)]:
        coarse,_=O.solve_boundary(s,alpha,points=801,ymax=40)
        fine,_=O.solve_boundary(s,alpha,points=1601,ymax=40)
        wide,_=O.solve_boundary(s,alpha,points=2401,ymax=60)
        convergence.append(dict(sigma=s,alpha=alpha,mesh_max_log_error=float(np.max(abs(fine-coarse))),
                                domain_max_log_error=float(np.max(abs(wide-fine)))))
    # Interpolation is checked away from table knots before examining returns.
    interpolation=[]
    interp=RegularGridInterpolator((sg,ag),out)
    for s,alpha in [(.00625,.075),(.0125,.45),(.025,.65),(.05,.85),(.10,.985),(.16,.995),(.07,.925),(.14,.99375),(.18,.99825)]:
        direct,_=O.solve_boundary(s,alpha,points=1601,ymax=40)
        estimate=interp([[s,alpha]])[0]
        interpolation.append(dict(sigma=s,alpha=alpha,max_log_error=float(np.max(abs(estimate-direct)))))
    error=max(x['mesh_max_log_error']+x['domain_max_log_error'] for x in convergence)
    error=max(error,max(x['max_log_error'] for x in interpolation))
    assert np.isfinite(error) and error<.01,('Numerical uncertainty too large',error)
    np.savez_compressed(a.output/'boundary_grid.npz',sigma=sg,alpha=ag,boundary=out)
    pd.DataFrame(audits).to_csv(a.output/'grid_audit.csv',index=False)
    pd.DataFrame(convergence).to_csv(a.output/'grid_convergence.csv',index=False)
    pd.DataFrame(interpolation).to_csv(a.output/'grid_interpolation.csv',index=False)
    dump(a.output/'numerics.json',dict(log_boundary_error_bound=error,grids=len(audits),
         finite_boundaries=bool(np.isfinite(out).all()),max_non_upper_nodes=max(x['non_upper_stop_nodes'] for x in audits),
         monotonic_failures=sum(not x['monotonic_remaining'] for x in audits),
         bound_interpretation='Maximum observed mesh/domain/interpolation discrepancy on prespecified probes; not a rigorous global error bound',
         seconds=time.time()-start,protocol_sha256=sha(PROTOCOL)))
    print('NUMERICS',error,flush=True)


class Boundary:
    def __init__(self,path):
        with np.load(path) as d:self.s=d['sigma'];self.a=d['alpha'];self.b=d['boundary']
        self.interp=RegularGridInterpolator((self.s,self.a,np.arange(20)),self.b,bounds_error=True)

    def lookup(self,sigma,alpha,remaining):
        s,alpha,r=np.broadcast_arrays(sigma,alpha,remaining)
        result=np.full(s.shape,np.inf);result[alpha<=0]=0
        interior=(alpha>0)&(alpha<1)
        result[interior]=self.interp(np.column_stack([np.clip(s[interior],self.s[0],self.s[-1]),
                           np.clip(alpha[interior],self.a[0],self.a[-1]),r[interior]]))
        clipped=interior & ((s<self.s[0])|(s>self.s[-1])|(alpha<self.a[0])|(alpha>self.a[-1]))
        return result,clipped


def fit_prior(ep,cutoff,plan):
    train=ep[ep.mature_date.lt(cutoff)].copy()
    assert len(train)>0 and train.mature_date.max()<pd.Timestamp(cutoff)
    refs={col:np.sort(train[col].to_numpy()) for col in SCORES}
    pct=np.min(np.column_stack([historical_percentile(train[c],refs[c]) for c in SCORES]),axis=1)
    bands=np.array(plan['estimation']['score_bands'])
    group=np.clip(np.searchsorted(bands,pct,side='right')-1,0,len(bands)-2)
    w=1/train.groupby('signal_date').trade.transform('size').to_numpy();w/=w.mean()
    lo,hi=np.quantile(train.outcome_delta,[.01,.99]);y=np.clip(train.outcome_delta.to_numpy(),lo,hi)
    pooled=float(np.average(y,weights=w));shrink=plan['estimation']['band_shrink_episodes']
    means=[];sizes=[]
    for i in range(len(bands)-1):
        mask=group==i
        means.append(float((np.sum(w[mask]*y[mask])+shrink*pooled)/(np.sum(w[mask])+shrink)))
        sizes.append(int(mask.sum()))
    sigma=float(train.episode_sigma[train.episode_sigma.gt(1e-8)].median())
    assert np.isfinite(sigma) and sigma>0
    record=dict(cutoff=str(pd.Timestamp(cutoff).date()),train_n=len(train),train_codes=train.Code.nunique(),
                train_dates=train.signal_date.nunique(),last_mature=train.mature_date.max(),pooled=pooled,
                bands=means,band_n=sizes,fallback_sigma=sigma,winsor_low=lo,winsor_high=hi,
                train_trade_sha256=hashlib.sha256(train.trade.to_numpy(dtype='<i8').tobytes()).hexdigest())
    return dict(refs=refs,bands=bands,record=record)


def assign_prior(ep,fit,kind):
    pct=np.min(np.column_stack([historical_percentile(ep[c],fit['refs'][c]) for c in SCORES]),axis=1)
    groups=np.clip(np.searchsorted(fit['bands'],pct,side='right')-1,0,len(fit['bands'])-2)
    rec=fit['record']
    prior=np.full(len(ep),rec['pooled']) if kind=='pooled' else np.array(rec['bands'])[groups]
    return prior,pct


def candidates(family):
    if family=='fixed_trailing':return [dict(family=family,threshold=t) for t in [.03,.05,.08,.12]]
    vols=[60] if family=='drift_gate' else [20,60]
    return [dict(family=family,prior_type=p,strength=k,vol_window=w)
            for p,k,w in itertools.product(['pooled','score_band'],[20,60,120,'infinite'],vols)]


def forecast_state(q,prior,fallback,config,boundary,offset=0):
    if config['family']=='fixed_trailing':
        return dict(decision=q.log_drawdown.to_numpy()>=-np.log1p(-config['threshold']))
    strength=np.inf if config['strength']=='infinite' else config['strength']
    delta=O.posterior_drift(prior,q.log_gain.to_numpy(),q.elapsed.to_numpy(),strength)
    window=config['vol_window'];other=60 if window==20 else 20
    sigma=q[f'sigma{window}'].to_numpy().copy()
    missing=~np.isfinite(sigma)|(sigma<=1e-8)
    sigma[missing]=q.loc[missing,f'sigma{other}'].to_numpy()
    robust=~np.isfinite(sigma)|(sigma<=1e-8)
    sigma[robust]=np.broadcast_to(fallback,sigma.shape)[robust]
    alpha=delta/sigma**2+.5
    if config['family']=='drift_gate':
        bound=np.full(len(q),np.nan);clipped=np.zeros(len(q),bool);decision=delta<=0
    else:
        bound,clipped=boundary.lookup(sigma,alpha,q.remaining.to_numpy())
        # Sensitivity only perturbs the nontrivial (0<alpha<1) boundary.
        bound=np.where((alpha>0)&(alpha<1),np.maximum(bound+offset,0),bound)
        decision=q.log_drawdown.to_numpy()>=bound
    return dict(delta=delta,sigma=sigma,alpha=alpha,boundary=bound,decision=decision,
                vol_fallback=missing,robust_fallback=robust,grid_clipped=clipped)


def simulate(ep,paths,decision=None,target=None,gap='open'):
    ix=ep.pidx.to_numpy();o=paths['o'][ix];h=paths['h'][ix];c=paths['c'][ix]
    n=len(ep);ret=c[:,-1].copy();day=np.full(n,20);reason=np.full(n,'deadline',dtype=object)
    if decision is not None:
        matrix=np.asarray(decision,bool).reshape(n,19)
        active=matrix.any(axis=1);first=matrix.argmax(axis=1)+1
        ret[active]=o[np.arange(n)[active],first[active]]
        day[active]=first[active]+1;reason[active]='next_open'
    if target is not None:
        assert decision is None
        hit=h>=target-1e-12;active=hit.any(axis=1);first=hit.argmax(axis=1)
        price=np.maximum(o[np.arange(n),first],target) if gap=='open' else np.full(n,target)
        ret[active]=price[active];day[active]=first[active]+1;reason[active]='target'
    return pd.DataFrame(dict(trade=ep.trade.to_numpy(),gross=ret,exit_day=day,reason=reason))


def criterion(ep,out,pct):
    data=ep[['trade','signal_date']].merge(out,on='trade',validate='one_to_one')
    scores=[];fallback=False;stats={}
    for name,mask in [('broad',np.ones(len(ep),bool)),('tail',pct>=95)]:
        z=data[mask];stats[f'{name}_n']=len(z);stats[f'{name}_dates']=z.signal_date.nunique()
        if name=='tail' and (len(z)<10 or z.signal_date.nunique()<5):
            fallback=True;scores.append(scores[0]);continue
        date_means=z.groupby('signal_date').gross.mean()
        value=z.gross.mean()-date_means.std(ddof=1)/np.sqrt(len(date_means))
        stats[f'{name}_mean']=z.gross.mean();stats[f'{name}_score']=value;scores.append(value)
    return float(np.mean(scores)),dict(tail_fallback=fallback,**stats)


def load(a):
    ep=pd.read_parquet(a.output/'episodes.parquet');q=pd.read_parquet(a.output/'states.parquet')
    with np.load(a.output/'paths.npz') as d:paths={k:d[k] for k in ['o','h','c','dates']}
    return ep,q,paths,Boundary(a.output/'boundary_grid.npz')


def tune(a,plan):
    ep,q,paths,boundary=load(a);rows=[];fits=[];configs={}
    for family in ['fixed_trailing','drift_gate','drawdown_boundary']:
        for j,config in enumerate(candidates(family)):configs[f'{family}_{j:02d}']=config
    for fold,(start,end) in enumerate(plan['tuning']['inner_cv']):
        fit=fit_prior(ep,start,plan);fits.append(fit['record'])
        val=ep[ep.signal_date.between(start,end)].copy().reset_index(drop=True)
        states=q[q.trade.isin(val.trade)].copy().reset_index(drop=True)
        assert len(states)==len(val)*19 and set(val.trade).isdisjoint(ep.loc[ep.mature_date.lt(start),'trade'])
        assigned={kind:assign_prior(val,fit,kind) for kind in ['pooled','score_band']}
        for cid,config in configs.items():
            kind=config.get('prior_type','pooled');prior,pct=assigned[kind]
            state=forecast_state(states,np.repeat(prior,19),fit['record']['fallback_sigma'],config,boundary)
            out=simulate(val,paths,decision=state['decision'])
            score,stats=criterion(val,out,pct)
            rows.append(dict(candidate=cid,fold=fold,family=config['family'],score=score,**stats))
        print('TUNED fold',fold,'fit',len(ep[ep.mature_date.lt(start)]),'validation',len(val),flush=True)
    trials=pd.DataFrame(rows);trials.to_csv(a.output/'tuning_folds.csv',index=False)
    summary=trials.groupby(['family','candidate'],sort=False).agg(score=('score','mean'),folds=('fold','nunique')).reset_index()
    summary.to_csv(a.output/'tuning_summary.csv',index=False)
    chosen={}
    for family,z in summary.groupby('family',sort=False):
        best=z.sort_values(['score','candidate'],ascending=[False,True],kind='stable').iloc[0]
        chosen[family]=dict(candidate=best.candidate,cv_score=best.score,**configs[best.candidate])
    dump(a.output/'selected.json',dict(protocol_sha256=sha(PROTOCOL),selected=chosen,configs=configs,fit_audits=fits))
    print('SELECTED',json.dumps(chosen),flush=True)


def outer(a,plan):
    ep,q,paths,boundary=load(a)
    selection=json.loads((a.output/'selected.json').read_text())
    assert selection['protocol_sha256']==sha(PROTOCOL)
    chosen=selection['selected']
    outer_ep=ep[ep.signal_date.between(*plan['outer_signal_dates'])].copy().reset_index(drop=True)
    query=q[q.trade.isin(outer_ep.trade)].copy().reset_index(drop=True)
    query['qid']=np.arange(len(query))
    outer_ep['prior_cutoff']=outer_ep.signal_date.dt.to_period('M').dt.start_time
    first=pd.Timestamp(plan['outer_signal_dates'][0]);outer_ep.loc[outer_ep.prior_cutoff.lt(first),'prior_cutoff']=first
    fits=[];assignments=[];pred=[]
    for cutoff,z in outer_ep.groupby('prior_cutoff',sort=True):
        fit=fit_prior(ep,cutoff,plan);fits.append(fit['record'])
        states=query[query.trade.isin(z.trade)].copy()
        train_states=q[q.mature_date.lt(cutoff)]
        age={}
        for t in [10,15]:
            risk=train_states[train_states[f'risk{t}']]
            grouped=risk.groupby('elapsed')[f'y{t}'].agg(['sum','count'])
            age[t]=(grouped['sum']+1)/(grouped['count']+2)
        for family in ['drift_gate','drawdown_boundary']:
            config=chosen[family];prior,pct=assign_prior(z,fit,config['prior_type'])
            priormap=pd.Series(prior,index=z.trade)
            state=forecast_state(states,states.trade.map(priormap).to_numpy(),fit['record']['fallback_sigma'],config,boundary)
            f=states[['qid','trade','elapsed']].copy();f['method']=family
            for name,values in state.items():f[name]=values
            f['prior_delta']=states.trade.map(priormap).to_numpy();f['prior_cutoff']=cutoff
            f['max_mean']=O.stopping_payoff(states.remaining.to_numpy(),0,f.sigma.to_numpy(),f.alpha.to_numpy())
            for quantile in [10,50,90]:
                f[f'q{quantile}']=O.maximum_quantile(quantile/100,f.delta.to_numpy(),f.sigma.to_numpy(),states.remaining.to_numpy())
            for t in [10,15]:
                threshold=np.maximum(np.log((1+t/100)/(1+states.spot_return.to_numpy())),0)
                f[f'p{t}']=1-O.maximum_cdf(threshold,f.delta.to_numpy(),f.sigma.to_numpy(),states.remaining.to_numpy())
                f[f'age_p{t}']=states.elapsed.map(age[t]).to_numpy()
            pred.append(f)
            assignments.extend(dict(trade=int(trade),method=family,prior_cutoff=cutoff,prior_delta=float(p),causal_percentile=float(cdf))
                               for trade,p,cdf in zip(z.trade,prior,pct))
        print('OUTER',str(cutoff.date()),'fit',fit['record']['train_n'],'episodes',len(z),flush=True)
    pred=pd.concat(pred).sort_values(['method','trade','elapsed']).reset_index(drop=True)
    assert len(pred)==2*len(query) and np.isfinite(pred[['sigma','delta','max_mean','p10','p15','q10','q50','q90']]).all().all()
    exits=[]
    def save(name,result):
        result['method']=name;exits.append(result)
    save('hold20',simulate(outer_ep,paths))
    save('first_possible_next_open',simulate(outer_ep,paths,decision=np.ones(len(query),bool)))
    for t in [10,15]:
        save(f'fixed_tp{t}',simulate(outer_ep,paths,target=t/100))
        save(f'fixed_tp{t}_limit',simulate(outer_ep,paths,target=t/100,gap='limit'))
    config=chosen['fixed_trailing'];state=forecast_state(query,None,None,config,boundary)
    save('tuned_fixed_trailing',simulate(outer_ep,paths,decision=state['decision']))
    for method,f in pred.groupby('method',sort=False):
        assert np.array_equal(f.qid.to_numpy(),query.qid.to_numpy())
        save(f'tuned_{method}',simulate(outer_ep,paths,decision=f.decision.to_numpy()))
        if method=='drawdown_boundary':
            eps=json.loads((a.output/'numerics.json').read_text())['log_boundary_error_bound']
            for label,offset in [('earlier',-eps),('later',eps)]:
                b=f.boundary.to_numpy();alpha=f.alpha.to_numpy()
                b=np.where((alpha>0)&(alpha<1),np.maximum(b+offset,0),b)
                save(f'boundary_{label}',simulate(outer_ep,paths,decision=query.log_drawdown.to_numpy()>=b))
    outer_ep.to_parquet(a.output/'outer_episodes.parquet',index=False)
    query.to_parquet(a.output/'outer_states.parquet',index=False)
    pred.to_parquet(a.output/'predictions.parquet',index=False)
    pd.concat(exits).to_parquet(a.output/'exits.parquet',index=False)
    pd.DataFrame(assignments).to_parquet(a.output/'prior_assignments.parquet',index=False)
    dump(a.output/'outer_fits.json',fits)


def masks(ep):
    return [('all',np.ones(len(ep),bool))]+[(f'base{t}',ep.base_min.ge(t).to_numpy()) for t in [95,97,99]]


def evaluate(a,plan):
    ep=pd.read_parquet(a.output/'outer_episodes.parquet');q=pd.read_parquet(a.output/'outer_states.parquet')
    pred=pd.read_parquet(a.output/'predictions.parquet');exits=pd.read_parquet(a.output/'exits.parquet')
    data=exits.merge(ep,on='trade',validate='many_to_one')
    tracker=pred.merge(q,on=['qid','trade','elapsed'],validate='many_to_one')
    tracker.to_parquet(a.output/'daily_tracker.parquet',index=False)
    cal=pd.DatetimeIndex(pd.read_csv(a.source/'calendar.csv',parse_dates=['Date']).Date)
    summaries=[];paired=[];scores=[];calibration=[];buckets=[];maxima=[];paired_brier=[];stateaudit=[]
    for cohort,mask in masks(ep):
        sub=ep[mask];ids=sub.trade;wide=data[data.trade.isin(ids)].pivot(index='trade',columns='method',values='gross').reindex(ids)
        for method,z in data[data.trade.isin(ids)].groupby('method',sort=False):
            summaries.append(dict(cohort=cohort,method=method,n=len(z),codes=z.Code.nunique(),dates=z.signal_date.nunique(),
                    mean=z.gross.mean(),median=z.gross.median(),win=(z.gross>0).mean(),tail5=z.gross.nsmallest(max(1,int(np.ceil(len(z)*.05)))).mean(),
                    mean_holding=z.exit_day.mean(),inverse_ratio=((1+z.max20)/(1+z.gross)).mean(),
                    capture_ratio=((1+z.gross)/(1+z.max20)).mean(),deadline_fraction=z.reason.eq('deadline').mean()))
            if method.startswith('tuned_'):
                for control in ['hold20','fixed_tp10','fixed_tp15','tuned_fixed_trailing']:
                    if method==control:continue
                    diff=wide[method]-wide[control]
                    ci=block_interval(diff.to_numpy(),sub.signal_date,cal) if len(sub)>=20 and sub.signal_date.nunique()>=10 else [None,None]
                    paired.append(dict(cohort=cohort,method=method,control=control,n=len(sub),delta=diff.mean(),ci_low=ci[0],ci_high=ci[1]))
        for method,z in tracker[tracker.trade.isin(ids)].groupby('method',sort=False):
            stateaudit.append(dict(cohort=cohort,method=method,states=len(z),vol_fallback=z.vol_fallback.mean(),
               robust_fallback=z.robust_fallback.mean(),grid_clipped=z.grid_clipped.mean(),alpha_le0=z.alpha.le(0).mean(),
               alpha_ge1=z.alpha.ge(1).mean(),decision_fraction=z.decision.mean(),median_sigma=z.sigma.median(),median_delta=z.delta.median()))
            err=z.max_mean-z.actual_max_spot
            maxima.append(dict(cohort=cohort,method=method,states=len(z),trades=z.trade.nunique(),mae=abs(err).mean(),
                 bias=err.mean(),rmse=np.sqrt((err**2).mean()),median_mae=abs(z.q50-z.actual_max_spot).mean(),
                 q10_q90_coverage=z.actual_max_spot.between(z.q10,z.q90).mean(),actual_below_current=z.actual_max_spot.lt(1).mean()))
            for target in [10,15]:
                risk=z[z[f'risk{target}']].copy();y=risk[f'y{target}'].to_numpy(float)
                for name,column in [(method,f'p{target}'),('age_prior',f'age_p{target}')]:
                    if name=='age_prior' and method=='drift_gate':continue
                    prob=risk[column].to_numpy();risk['loss']=(prob-y)**2
                    scores.append(dict(cohort=cohort,method=name,target=target,n=risk.trade.nunique(),states=len(risk),
                          brier=risk.loss.mean(),trade_equal_brier=risk.groupby('trade').loss.mean().mean(),
                          predicted=prob.mean(),actual=y.mean(),logloss=np.mean(-y*np.log(np.clip(prob,1e-6,1-1e-6))-(1-y)*np.log(np.clip(1-prob,1e-6,1-1e-6)))))
                    for lo,hi in [(1,5),(6,10),(11,15),(16,19)]:
                        b=risk[risk.elapsed.between(lo,hi)]
                        buckets.append(dict(cohort=cohort,method=name,target=target,start=lo,end=hi,states=len(b),
                                 brier=b.loss.mean(),predicted=b[column].mean(),actual=b[f'y{target}'].mean()))
                    for group,b in risk.groupby(np.minimum((risk[column]*5).astype(int),4)):
                        calibration.append(dict(cohort=cohort,method=name,target=target,bin=group,states=len(b),
                                    predicted=b[column].mean(),actual=b[f'y{target}'].mean()))
                risk['delta_brier']=(risk[f'p{target}']-y)**2-(risk[f'age_p{target}']-y)**2
                ag=risk.groupby('trade',sort=False).agg(delta=('delta_brier','mean'),date=('signal_date','first'))
                ci=block_interval(ag.delta,ag.date,cal) if len(ag)>=20 and ag.date.nunique()>=10 else [None,None]
                paired_brier.append(dict(cohort=cohort,method=method,target=target,n=len(ag),
                                   trade_equal_delta=ag.delta.mean(),ci_low=ci[0],ci_high=ci[1]))
    for name,records in [('policies',summaries),('paired_returns',paired),('scores',scores),('calibration',calibration),
                         ('day_buckets',buckets),('maxima',maxima),('paired_brier',paired_brier),('state_audit',stateaudit)]:
        pd.DataFrame(records).to_csv(a.output/f'{name}.csv',index=False)
    # A forecast-only export excludes future outcomes and retains every query.
    forecast=q[['qid','trade','Code','signal_date','entry_date','asof_date','elapsed','remaining','buy','close','spot_return','base_min']].copy()
    for t in [10,15]:forecast[f'status{t}']=q[f'risk{t}'].map({True:'unarrived',False:'arrived'})
    for method,f in pred.groupby('method',sort=False):
        f=f.set_index('qid').reindex(forecast.qid)
        for t in [10,15]:
            forecast[f'{method}_p{t}_remaining']=f[f'p{t}'].to_numpy()
            forecast[f'{method}_p{t}_ever']=np.where(q[f'risk{t}'],f[f'p{t}'],1)
        for col in ['max_mean','q10','q50','q90']:
            forecast[f'{method}_{col}_pct_from_buy']=100*(f[col].to_numpy()*(1+q.spot_return.to_numpy())-1)
        for col in ['decision','delta','sigma','alpha','boundary','prior_delta','prior_cutoff']:
            forecast[f'{method}_{col}']=f[col].to_numpy()
    forecast.to_csv(a.output/'daily_forecasts.csv',index=False)
    data.to_csv(a.output/'trade_outcomes.csv',index=False)
    print(pd.DataFrame(summaries).query("cohort in ['all','base95']")[[ 'cohort','method','n','mean','mean_holding']].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','grid','tune','outer','evaluate'])
    p.add_argument('--source',type=Path,default=Path('/workspace/scratch/daily-arrival-20261105/results'))
    p.add_argument('--prices',type=Path,default=Path('/workspace/scratch/exit-upside-20261105/results'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);plan=json.loads(PROTOCOL.read_text())
    if a.phase not in ['prepare','grid']:
        assert (a.output/'protocol.json').read_bytes()==PROTOCOL.read_bytes(),'Protocol changed after preparation'
    globals()[a.phase](a,plan)


if __name__=='__main__':main()
