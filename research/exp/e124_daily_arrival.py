#!/usr/bin/env python3
"""E124 daily fixed-entry +10/+15 arrival; research outputs only.

python research/exp/e124_daily_arrival.py --source E123_RESULTS --raw RELEASE_PRICES
  --market RAW_TOPIX --output OUTPUT --phase prepare|tune|outer|all
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.metrics import log_loss,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.insert(0,str(ROOT))
import daily_arrival as D
import exit_upside as E
warnings.filterwarnings('ignore',message='X does not have valid feature names')
PROTOCOL=REPO/'docs/experiments/daily-arrival-2026-11-05/protocol.json'
STATIC=('elapsed','remaining','entry_gap')+tuple('entry_score_'+x for x in E.CLASSIFIERS)+('entry_rank_ltr','entry_rank_ltr5')
PREFIX=tuple(n for n in D.PREFIX_FEATURES if n not in ('elapsed','remaining'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    def default(v):
        if isinstance(v,(pd.Timestamp,np.datetime64)):return str(pd.Timestamp(v))
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default)+'\n')


def prepare(a,plan):
    p=a.output
    source=pd.read_parquet(a.source/'signal_paths.parquet')
    paths=np.load(a.source/'paths.npz')
    ids=source.index[source.fit_reference&source.status.eq('complete')].to_numpy()
    sig=source.loc[ids].copy()
    chosen=E.selected_ids(source,{'base':95})
    source['selected_top2']=source.index.isin(chosen)
    source.to_parquet(p/'source_signals.parquet',index=False)
    identity=pd.read_csv(a.source/'model_identity.csv')
    names={'lgbm':'fixed_return_v1_oof.parquet','xgb':'fixed_return_v1_xgb_oof.parquet','cat':'fixed_return_v1_cat_oof.parquet',
           **{x:f'fixed_return_v1_{x}_oof.parquet' for x in E.CLASSIFIERS[3:]},'ltr':'ltr_oof.parquet','ltr5':'ltr5_oof.parquet'}
    for model,file in names.items():
        meta=json.loads((ROOT/'model'/('meta.json' if model=='lgbm' else f'models/{model}/meta.json')).read_text())
        actual=sha(a.source.parent/'oof'/file)
        assert actual==meta['artifactHashes']['oof.parquet']==identity.loc[identity.model.eq(model),'sha256'].iloc[0]
    identity.to_csv(p/'model_identity.csv',index=False)
    trade_dates=sig.Date.to_numpy(dtype='datetime64[ns]')
    lookup=np.column_stack([trade_dates,paths['dates'][ids,:-1]])
    stock=pd.read_parquet(a.source/'adjusted_prices.parquet')
    stock=stock[stock.Date.ge(sig.Date.min()-pd.Timedelta(days=50))&stock.Code.isin(sig.Code.unique())].copy()
    pieces=[]
    volhash={}
    for file in sorted(a.raw.glob('bars_*.parquet')):
        if int(file.stem[-4:])<sig.Date.min().year:continue
        raw=pd.read_parquet(file,columns=['Code','Date','C','Vo'])
        pieces.append(raw[raw.Code.isin(sig.Code.unique())&raw.Date.ge(stock.Date.min())])
        volhash[file.name]=sha(file)
    volumes=pd.concat(pieces,ignore_index=True).rename(columns={'C':'raw_close'})
    stock=stock.merge(volumes,on=['Code','Date'],how='left',validate='one_to_one')
    stock['volume']=stock.Vo*stock.raw_close/stock.AdjC
    del volumes,pieces
    topix_files=sorted(a.market.glob('topix_*.parquet'))
    topix=pd.concat([pd.read_parquet(f) for f in topix_files]).sort_values('Date').drop_duplicates('Date').set_index('Date').topix
    bm=pd.DataFrame({f'market_ret{lag}':topix/topix.shift(lag)-1 for lag in (1,5,20)})
    tnames=list(D.TECH_FEATURES)+list(bm.columns)
    technical=np.full((len(ids),20,len(tnames)),np.nan,dtype=np.float32)
    entry_gap=np.full(len(ids),np.nan)
    local=sig.reset_index(drop=False).rename(columns={'index':'trade'})
    grouped=local.groupby('Code').groups
    for code,g in stock.groupby('Code',sort=False):
        if code not in grouped:continue
        ix=np.asarray(grouped[code])
        g=g.sort_values('Date').set_index('Date')
        t=D.daily_technicals(g,g.volume).join(bm,how='left')
        technical[ix]=t.reindex(lookup[ix].ravel()).to_numpy().reshape(len(ix),20,-1)
        previous=g.AdjC.reindex(local.loc[ix,'Date']).to_numpy()
        entry_gap[ix]=local.loc[ix,'entry_adjusted'].to_numpy()/previous-1
    print(f'prepared technical prefixes for {len(ids):,} trades',flush=True)
    states=D.landmark_arrays(*(paths[k][ids] for k in ('o','h','l','c')))
    n=len(ids)
    rows=pd.DataFrame({'trade':np.repeat(ids,20),'Code':np.repeat(sig.Code.to_numpy(),20),
                       'signal_date':np.repeat(trade_dates,20),'entry_date':np.repeat(sig.entry_date.to_numpy(),20),
                       'mature_date':np.repeat(sig.mature_date.to_numpy(),20),
                       'asof_date':np.column_stack([sig.entry_date.to_numpy(),paths['dates'][ids,:-1]]).ravel(),
                       'strict':np.repeat(source.loc[ids,'selected_top2'].to_numpy(),20)})
    for name,values in states.items():rows[name]=values.ravel()
    rows['entry_gap']=np.repeat(entry_gap,20)
    for algo in E.CLASSIFIERS:rows['entry_score_'+algo]=np.repeat(sig['score_'+algo].to_numpy(),20)
    for algo in ('ltr','ltr5'):rows['entry_rank_'+algo]=np.repeat(sig['pct_'+algo].to_numpy()/100,20)
    for j,name in enumerate(tnames):
        rows['entry_tech_'+name]=np.repeat(technical[:,0,j],20)
        rows['now_tech_'+name]=technical[:,:,j].ravel()
    mkt0=topix.reindex(trade_dates).to_numpy()
    mkt=topix.reindex(lookup.ravel()).to_numpy().reshape(n,20)/mkt0[:,None]-1
    rows['market_since_signal']=mkt.ravel()
    rows['relative_to_market']=(states['spot_return']-mkt).ravel()
    static=list(STATIC)+['entry_tech_'+name for name in tnames]
    dynamic=static+list(PREFIX)+['now_tech_'+name for name in tnames]+['market_since_signal','relative_to_market']
    for col in static+dynamic:
        rows[col]=rows[col].replace([np.inf,-np.inf],np.nan).astype(np.float32)
    rows.to_parquet(p/'landmarks.parquet',index=False)
    dump(p/'feature_contract.json',{'static_age':static,'daily_state':dynamic,'prefix_feature_count':len(dynamic)-len(static)})
    source_stats={}
    for name,window in [('calibration',plan['calibration_period']),('outer',plan['outer_entry_dates'])]:
        z=source[source.Date.between(*window)&source.fold_safe&source.selected_top2]
        source_stats[name]={'selected_before_quality':len(z),'complete_rank_and_price':int(z.index.isin(ids).sum()),
                            'excluded':z.loc[~z.index.isin(ids),['Code','Date','status','all_eight']].to_dict('records')}
    audit={'source_inputs':{name:sha(a.source/name) for name in ['signal_paths.parquet','paths.npz','adjusted_prices.parquet','model_identity.csv']},
           'volume_inputs':volhash,'topix_inputs':{f.name:sha(f) for f in topix_files},
           'protocol_sha256':sha(PROTOCOL),'trades':len(ids),'landmarks':len(rows),'price_through':stock.Date.max(),
           'topix_through':topix.index.max(),'technical_missing_pct':float(100*rows[dynamic].isna().mean().mean()),
           'strict_counts':source_stats,'targets':[10,15],'holding_sessions':20}
    dump(p/'data_audit.json',audit)
    print('DATA '+json.dumps({k:v for k,v in audit.items() if k not in ['source_inputs','volume_inputs','topix_inputs','strict_counts']},default=str),flush=True)


def task_rows(d,kind,target=None,quantile=None):
    if kind=='hit':return d[d[f'risk{target}']]
    if kind=='gain':return d[d[f'risk{target}']&d.elapsed.ge(1)]
    if kind=='comp':return d[d[f'comp_risk{target}']]
    if kind=='quantile':return d[d.elapsed.ge(1)]
    raise ValueError(kind)


def target_values(rows,kind,target=None):
    if kind=='hit':return rows[f'y{target}'].to_numpy(int)
    if kind=='gain':return rows[f'gain{target}'].to_numpy(float)
    if kind=='comp':return rows[f'comp{target}'].to_numpy(int)
    return rows.future_max_from_spot.to_numpy(float)


def fit(rows,cols,candidate,kind,target=None,quantile=None):
    common={'n_jobs':2,'verbosity':-1,'random_state':42,'subsample':.8,'colsample_bytree':.8,'subsample_freq':1,
            'deterministic':True,'force_col_wise':True,**candidate}
    if kind=='hit':model=lgb.LGBMClassifier(objective='binary',**common)
    elif kind=='comp':model=lgb.LGBMClassifier(objective='multiclass',num_class=3,**common)
    elif kind=='quantile':model=lgb.LGBMRegressor(objective='quantile',alpha=quantile,**common)
    else:model=lgb.LGBMRegressor(objective='regression',**common)
    model.fit(rows[cols].to_numpy(float),target_values(rows,kind,target))
    return model


def predict(model,rows,cols,kind):
    x=rows[cols].to_numpy(float)
    return model.predict_proba(x)[:,1] if kind=='hit' else model.predict_proba(x) if kind=='comp' else model.predict(x)


def loss(y,pred,kind,quantile=None):
    if kind=='comp':return np.sum((np.eye(3)[y.astype(int)]-pred)**2,axis=1)
    if kind=='quantile':
        delta=y-pred;return np.maximum(quantile*delta,(quantile-1)*delta)
    return (y-pred)**2


def tasks():
    out=[]
    for kind in ('hit','gain'):
        for target in D.TARGETS:
            for arm in ('static_age','daily_state'):out.append((f'{kind}{target}_{arm}',kind,target,None,arm))
    for target in D.TARGETS:out.append((f'comp{target}_daily_state','comp',target,None,'daily_state'))
    for q in (.25,.5,.75):out.append((f'quantile{q}_daily_state','quantile',None,q,'daily_state'))
    return out


def tune(p,plan,d,features,paths):
    selected={};all_trials=[];cv_policy=[]
    for key,kind,target,q,arm in tasks():
        part=task_rows(d,kind,target,q)
        splits=[]
        for start,end in plan['inner_cv']:
            tr=part[D.matured_mask(part,start)]
            va=part[part.signal_date.between(start,end)&part.elapsed.ge(1)]
            assert set(tr.trade).isdisjoint(va.trade)
            assert tr.mature_date.max()<pd.Timestamp(start)
            splits.append((tr,va,start,end))
        records=[]
        for j,candidate in enumerate(plan['tuning_candidates']):
            t0=time.time();scores=[]
            for tr,va,_,_ in splits:
                model=fit(tr,features[arm],candidate,kind,target,q)
                prob=predict(model,va,features[arm],kind)
                scores.append(float(loss(target_values(va,kind,target),prob,kind,q).mean()))
            rec={'task':key,'candidate':j,'cv_mean':float(np.mean(scores)),'fold_scores':scores,'seconds':time.time()-t0}
            records.append(rec);all_trials.append(rec)
        best=min(records,key=lambda x:(x['cv_mean'],x['candidate']))
        selected[key]={'candidate':best['candidate'],'params':plan['tuning_candidates'][best['candidate']],
                       'cv_loss':best['cv_mean'],'folds':[{'start':s,'end':e,'n_train_trades':tr.trade.nunique(),'n_valid_trades':va.trade.nunique(),'last_train_mature':tr.mature_date.max()} for tr,va,s,e in splits]}
        print(f'TUNED {key} candidate{best["candidate"]} CV={best["cv_mean"]:.6f}',flush=True)
        pd.DataFrame(all_trials).to_json(p/'tuning_trials.json',orient='records',indent=2)
        if kind=='gain':
            pieces=[]
            for tr,va,start,end in splits:
                v=va[va.strict].copy()
                model=fit(tr,features[arm],selected[key]['params'],kind,target,q)
                v['predicted_gain']=predict(model,v,features[arm],kind)
                pieces.append(v[['trade','elapsed','predicted_gain']])
                # Buy-day target winners have no unarrived evening row. They
                # must still be included in policy selection's purchase cohort.
                sentinel=d[d.strict&d.signal_date.between(start,end)&d.elapsed.eq(0)][['trade','elapsed']].copy()
                sentinel['predicted_gain']=np.nan
                pieces.append(sentinel)
            predictions=pd.concat(pieces)
            scores=[]
            for cutoff in plan['exit_cutoffs']:
                trades=policy_trades(d,predictions,paths,target,cutoff)
                dates=trades.groupby('signal_date').gross.mean()
                objective=trades.gross.mean()-plan['cost_ratio']-dates.std()/np.sqrt(len(dates))
                scores.append({'task':key,'cutoff':cutoff,'n':len(trades),'n_dates':len(dates),
                               'mean_net_pct':100*(trades.gross.mean()-plan['cost_ratio']),'objective':objective})
            eligible=[s for s in scores if s['n']>=30 and s['n_dates']>=10]
            policy=min(eligible,key=lambda x:(-x['objective'],x['cutoff'])) if eligible else {'cutoff':0,'reason':'insufficient trade count'}
            selected[key]['exit_cutoff']=policy['cutoff']
            cv_policy.extend(scores)
    dump(p/'selection_frozen.json',{'protocol_sha256':sha(PROTOCOL),'selections':selected,'outer_metrics_computed':False})
    pd.DataFrame(cv_policy).to_csv(p/'exit_threshold_development.csv',index=False)
    print('All hyperparameters and daily exit cutoffs frozen before outer evaluation',flush=True)
    return selected


def policy_trades(d,predictions,paths,target,cutoff,profit_gap='limit'):
    eligible=set(predictions.trade.unique())
    records=[]
    indexed=predictions.set_index(['trade','elapsed']).predicted_gain
    meta=d[d.trade.isin(eligible)].drop_duplicates('trade').set_index('trade')
    # Include early target hits too: selected trade universe is supplied separately
    # by assigning a sentinel row when it has no unarrived daily landmarks.
    for trade in sorted(eligible):
        gains=np.full(20,np.nan)
        for elapsed in range(1,20):gains[elapsed]=indexed.get((trade,elapsed),np.nan)
        r=D.simulate_daily_exit(*(paths[k][trade] for k in ('o','h','l','c')),gains,target=target/100,cutoff=cutoff,profit_gap=profit_gap)
        records.append({'trade':int(trade),'Code':meta.at[trade,'Code'],'signal_date':meta.at[trade,'signal_date'],**r})
    return pd.DataFrame(records)


def platt_fit(y,prob):
    z=np.log(np.clip(prob,1e-6,1-1e-6)/np.clip(1-prob,1e-6,1))
    def objective(ab):
        eta=ab[0]*z+ab[1]
        return float(np.mean(np.logaddexp(0,eta)-y*eta))
    res=minimize(objective,[1,0],method='L-BFGS-B',bounds=[(0,20),(-20,20)])
    if not res.success:raise ValueError('Calibration optimizer did not converge')
    return {'slope':float(res.x[0]),'intercept':float(res.x[1]),'n':len(y)}


def calibrate(prob,params):
    z=np.log(np.clip(prob,1e-6,1-1e-6)/np.clip(1-prob,1e-6,1))
    eta=np.clip(params['slope']*z+params['intercept'],-35,35)
    return 1/(1+np.exp(-eta))


def outer(p,plan,d,features,paths,selections):
    calibrators={}
    for target in D.TARGETS:
        for arm in ('static_age','daily_state'):
            key=f'hit{target}_{arm}';part=task_rows(d,'hit',target)
            tr=part[D.matured_mask(part,plan['calibration_period'][0])]
            va=part[part.signal_date.between(*plan['calibration_period'])&part.elapsed.ge(1)]
            m=fit(tr,features[arm],selections[key]['params'],'hit',target)
            calibrators[key]=platt_fit(target_values(va,'hit',target),predict(m,va,features[arm],'hit'))
    dump(p/'calibration_frozen.json',calibrators)
    test=d[d.signal_date.between(*plan['outer_entry_dates'])].copy()
    records=[];folds=[];importances=[]
    for month,ix in test.groupby(test.asof_date.dt.to_period('M')).groups.items():
        cutoff=max(month.start_time,pd.Timestamp(plan['outer_entry_dates'][0]))
        tr=d[D.matured_mask(d,cutoff)]
        te=test.loc[ix].copy()
        assert set(tr.trade).isdisjoint(te.trade)
        assert tr.mature_date.max()<cutoff
        folds.append({'month':str(month),'cutoff':cutoff,'last_train_label':tr.mature_date.max(),
                      'train_trades':tr.trade.nunique(),'test_trades':te.trade.nunique(),'test_rows':len(te)})
        pred=te[['trade','Code','signal_date','asof_date','entry_date','mature_date','elapsed','remaining','strict',
                 'spot_return','known_peak','known_low','future_max_from_spot']+
                [name+str(t) for t in D.TARGETS for name in ['y','y_close','risk','arrived','comp','comp_risk','gain']]].copy()
        for key,kind,target,q,arm in tasks():
            train=task_rows(tr,kind,target,q)
            m=fit(train,features[arm],selections[key]['params'],kind,target,q)
            raw=predict(m,te,features[arm],kind)
            if kind=='hit':
                gain=m.booster_.feature_importance(importance_type='gain')
                for name,weight in zip(features[arm],gain):importances.append({'month':str(month),'task':key,'feature':name,'gain':weight})
            modeldir=p/'fitted_last_month';modeldir.mkdir(exist_ok=True)
            m.booster_.save_model(str(modeldir/(key+'.txt')))
            if kind=='hit':
                pred[key+'_raw']=raw
                pred[key]=calibrate(raw,calibrators[key])
            elif kind=='comp':
                for j,label in enumerate(['expiry','target_first','stop_first']):pred[key+'_'+label]=raw[:,j]
            else:pred[key]=raw
        for arm in ('static_age','daily_state'):
            a10,a15=D.coherent_probabilities(pred['hit10_'+arm],pred['hit15_'+arm],pred.arrived10,pred.arrived15)
            pred['hit10_'+arm]=a10;pred['hit15_'+arm]=a15
        qcols=[f'quantile{q}_daily_state' for q in (.25,.5,.75)]
        pred[qcols]=np.sort(pred[qcols].to_numpy(),axis=1)
        # The empirical reference uses strict past trades at the same elapsed age.
        for target in D.TARGETS:
            r=tr[tr.strict&tr[f'risk{target}']].groupby('elapsed')[f'y{target}'].agg(['sum','count'])
            prior=(r['sum']+1)/(r['count']+2)
            pred[f'age_prior{target}']=pred.elapsed.map(prior).fillna(.5)
        records.append(pred)
        print(f'OUTER {month} train{tr.trade.nunique():,} test{len(te):,}',flush=True)
        pd.DataFrame(folds).to_csv(p/'outer_maturity_checks.csv',index=False)
    predictions=pd.concat(records).sort_values(['trade','elapsed']).reset_index(drop=True)
    # Initial probability is genuinely the prediction available at purchase,
    # including for positions that cross a model-refit boundary later.
    for target in D.TARGETS:
        initial=predictions[predictions.elapsed.eq(0)].set_index('trade')[f'hit{target}_static_age']
        predictions[f'purchase_only{target}']=predictions.trade.map(initial)
    predictions.to_parquet(p/'outer_predictions.parquet',index=False)
    pd.DataFrame(importances).to_csv(p/'feature_importance.csv',index=False)
    summarize(p,plan,d,predictions,paths,selections)


def summarize(p,plan,d,pred,paths,selected):
    rows=[];paired=[];cal=[];buckets=[];comp=[];quants=[];returns=[];diffs=[];gaps=[];gainmetrics=[];incremental=[]
    marketcal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(a_source_calendar(p)).Date))
    for cohort,whole in [('strict',pred[pred.strict]),('broad',pred)]:
        for target in D.TARGETS:
            z=whole[whole[f'risk{target}']&whole.elapsed.ge(1)].copy()
            y=z[f'y{target}'].to_numpy(float)
            forecasts={'purchase_only':z[f'purchase_only{target}'].to_numpy(),
                       'age_prior':z[f'age_prior{target}'].to_numpy(),
                       **{arm:z[f'hit{target}_{arm}'].to_numpy() for arm in ('static_age','daily_state')}}
            for arm,prob in forecasts.items():
                e=(y-prob)**2
                rows.append({'cohort':cohort,'target':target,'arm':arm,'trades':z.trade.nunique(),'landmarks':len(z),
                             'brier':e.mean(),'trade_equal_brier':D.trade_mean(e,z),
                             'logloss':log_loss(y,np.clip(prob,1e-6,1-1e-6),labels=[0,1]),
                             'auc':roc_auc_score(y,prob) if len(np.unique(y))>1 else np.nan,
                             'predicted_pct':100*prob.mean(),'actual_pct':100*y.mean(),
                             'close_arrival_actual_pct':100*z[f'y_close{target}'].mean()})
                if arm in ('static_age','daily_state'):
                    for low,high in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.00001)]:
                        mask=(prob>=low)&(prob<high)
                        if mask.any():cal.append({'cohort':cohort,'target':target,'arm':arm,'bin':f'{low:.0%}–{min(high,1):.0%}',
                                                 'trades':z.loc[mask,'trade'].nunique(),'n':int(mask.sum()),
                                                 'predicted_pct':100*prob[mask].mean(),'actual_pct':100*y[mask].mean()})
                    for name,lo,hi in [('1–3',1,3),('4–7',4,7),('8–12',8,12),('13–19',13,19)]:
                        mask=z.elapsed.between(lo,hi)
                        if mask.any():buckets.append({'cohort':cohort,'target':target,'arm':arm,'days':name,
                                                      'trades':z.loc[mask,'trade'].nunique(),'n':int(mask.sum()),
                                                      'brier':e[mask].mean(),'predicted_pct':100*prob[mask].mean(),'actual_pct':100*y[mask].mean()})
            change=(y-forecasts['daily_state'])**2-(y-forecasts['static_age'])**2
            ci=E.block_interval(change,z.signal_date,marketcal) if cohort=='strict' else [None,None]
            paired.append({'cohort':cohort,'target':target,'trades':z.trade.nunique(),'landmarks':len(z),
                           'dynamic_minus_static_brier':change.mean(),'ci_low':ci[0],'ci_high':ci[1]})
            for arm in ('static_age','daily_state'):
                error=z[f'gain{target}_{arm}'].to_numpy()-z[f'gain{target}'].to_numpy()
                gainmetrics.append({'cohort':cohort,'target':target,'arm':arm,'trades':z.trade.nunique(),'landmarks':len(z),
                                    'rmse_pt':100*np.sqrt(np.mean(error**2)),'mae_pt':100*np.mean(np.abs(error)),
                                    'predicted_mean_pt':100*z[f'gain{target}_{arm}'].mean(),'actual_mean_pt':100*z[f'gain{target}'].mean()})
            cz=whole[whole[f'comp_risk{target}']&whole.elapsed.ge(1)]
            probs=cz[[f'comp{target}_daily_state_'+s for s in ('expiry','target_first','stop_first')]].to_numpy()
            yy=cz[f'comp{target}'].to_numpy(int)
            comp.append({'cohort':cohort,'target':target,'trades':cz.trade.nunique(),'landmarks':len(cz),
                         'multiclass_brier':loss(yy,probs,'comp').mean(),
                         'target_first_above_unconditional_pct':100*(probs[:,1]>cz[f'hit{target}_daily_state'].to_numpy()+1e-12).mean(),
                         'target_first_pred_pct':100*probs[:,1].mean(),'target_first_actual_pct':100*(yy==1).mean(),
                         'stop_first_pred_pct':100*probs[:,2].mean(),'stop_first_actual_pct':100*(yy==2).mean()})
        qz=whole[whole.elapsed.ge(1)]
        coverage=(qz.future_max_from_spot>=qz['quantile0.25_daily_state'])&(qz.future_max_from_spot<=qz['quantile0.75_daily_state'])
        for q in (.25,.5,.75):
            yq=qz.future_max_from_spot.to_numpy();pr=qz[f'quantile{q}_daily_state'].to_numpy()
            quants.append({'cohort':cohort,'quantile':q,'trades':qz.trade.nunique(),'landmarks':len(qz),
                           'pinball':loss(yq,pr,'quantile',q).mean(),'predicted_pct':100*pr.mean(),
                           'actual_below_pct':100*(yq<=pr).mean(),'q25_q75_coverage_pct':100*coverage.mean()})
    # Exit rules on the same strict purchase universe, including buy-day target hits.
    strict=pred[pred.strict]
    for target in D.TARGETS:
        base_predictions=strict[['trade','elapsed']].assign(predicted_gain=np.nan)
        benchmark=policy_trades(d,base_predictions,paths,target,0)
        for arm in ('static_age','daily_state'):
            pp=strict[['trade','elapsed',f'gain{target}_{arm}']].rename(columns={f'gain{target}_{arm}':'predicted_gain'})
            chosen=selected[f'gain{target}_{arm}']['exit_cutoff']
            trades=policy_trades(d,pp,paths,target,chosen)
            trades['arm']=arm;trades['target']=target
            trades.to_csv(p/f'exit_trades_{target}_{arm}.csv',index=False)
            pair=trades.merge(benchmark[['trade','gross']],on='trade',suffixes=('','_fixed'),validate='one_to_one')
            ci=E.block_interval(pair.gross-pair.gross_fixed,pair.signal_date,marketcal)
            diffs.append({'target':target,'arm':arm,'trades':len(pair),'gain_vs_fixed_pct':100*(pair.gross-pair.gross_fixed).mean(),
                          'ci_low_pct':100*ci[0],'ci_high_pct':100*ci[1],'cutoff':chosen})
            for cost in plan['cost_sensitivity']:
                returns.append({'target':target,'arm':arm,'cost_pct':100*cost,'trades':len(trades),
                                'mean_net_pct':100*(trades.gross.mean()-cost),'win_pct':100*(trades.gross>cost).mean(),
                                'worst_net_pct':100*(trades.gross.min()-cost),'avg_days':trades.exit_day.mean(),
                                'early_exits':int(trades.reason.eq('daily_exit').sum())})
            optimistic=policy_trades(d,pp,paths,target,chosen,profit_gap='open')
            gaps.append({'target':target,'arm':arm,'trades':len(optimistic),'profit_gap':'opening_price',
                         'mean_net_pct':100*(optimistic.gross.mean()-plan['cost_ratio'])})
        for cost in plan['cost_sensitivity']:
            returns.append({'target':target,'arm':'fixed_target','cost_pct':100*cost,'trades':len(benchmark),
                            'mean_net_pct':100*(benchmark.gross.mean()-cost),'win_pct':100*(benchmark.gross>cost).mean(),
                            'worst_net_pct':100*(benchmark.gross.min()-cost),'avg_days':benchmark.exit_day.mean(),'early_exits':0})
        optimistic=policy_trades(d,base_predictions,paths,target,0,profit_gap='open')
        gaps.append({'target':target,'arm':'fixed_target','trades':len(optimistic),'profit_gap':'opening_price',
                     'mean_net_pct':100*(optimistic.gross.mean()-plan['cost_ratio'])})
        aa=pd.read_csv(p/f'exit_trades_{target}_static_age.csv')
        bb=pd.read_csv(p/f'exit_trades_{target}_daily_state.csv')
        pair=bb.merge(aa[['trade','gross']],on='trade',suffixes=('','_static'),validate='one_to_one')
        delta=pair.gross-pair.gross_static
        ci=E.block_interval(delta,pair.signal_date,marketcal)
        incremental.append({'target':target,'trades':len(pair),'daily_minus_static_net_pt':100*delta.mean(),
                            'ci_low_pt':100*ci[0],'ci_high_pt':100*ci[1]})
    for name,data in [('probability_metrics',rows),('paired_probability_comparison',paired),('probability_calibration',cal),
                      ('metrics_by_elapsed',buckets),('competing_risk_metrics',comp),('quantile_metrics',quants),
                      ('exit_policy_metrics',returns),('paired_exit_comparison',diffs)]:
        pd.DataFrame(data).to_csv(p/(name+'.csv'),index=False)
    strict.to_csv(p/'strict_daily_history.csv',index=False)
    pd.DataFrame(gaps).to_csv(p/'gap_fill_sensitivity.csv',index=False)
    pd.DataFrame(gainmetrics).to_csv(p/'continuation_gain_metrics.csv',index=False)
    pd.DataFrame(incremental).to_csv(p/'paired_state_exit_comparison.csv',index=False)
    tracker_cols=['trade','Code','signal_date','asof_date','elapsed','remaining','spot_return','known_peak','known_low']
    tracker=strict[tracker_cols].copy()
    for target in D.TARGETS:
        tracker[f'p{target}_remaining']=strict[f'hit{target}_daily_state']
        tracker[f'p{target}_entry_age']=strict[f'hit{target}_static_age']
        tracker[f'status{target}']=np.where(strict[f'arrived{target}'],'到達済み','未到達')
        tracker[f'continuation_gain{target}']=strict[f'gain{target}_daily_state'].where(strict[f'risk{target}']&strict.elapsed.ge(1))
        risk=strict[f'comp_risk{target}']
        for label in ('target_first','stop_first'):
            tracker[f'comp{target}_{label}']=strict[f'comp{target}_daily_state_{label}'].where(risk)
    tracker['competing_model_status']='未校正・参考用（主モデルとの確率不整合あり）'
    for q in (.25,.5,.75):
        tracker[f'future_high_q{q}_from_spot']=strict[f'quantile{q}_daily_state'].where(strict.elapsed.ge(1))
        tracker[f'future_high_q{q}_from_entry']=((1+strict.spot_return)*(1+strict[f'quantile{q}_daily_state'])-1).where(strict.elapsed.ge(1))
    terminal=[]
    for trade,g in tracker.groupby('trade'):
        row={k:np.nan for k in tracker.columns}
        row.update(g.iloc[-1][['trade','Code','signal_date']].to_dict())
        row.update(asof_date=paths['dates'][trade,-1],elapsed=20,remaining=0,spot_return=paths['c'][trade,-1],
                   known_peak=np.max(paths['h'][trade]),known_low=np.min(paths['l'][trade]))
        for target in D.TARGETS:
            hit=bool(np.max(paths['h'][trade])>=target/100-1e-12)
            row[f'p{target}_remaining']=float(hit);row[f'p{target}_entry_age']=float(hit)
            row[f'status{target}']='到達済み' if hit else '期限終了・未到達'
        terminal.append(row)
    tracker=pd.concat([tracker,pd.DataFrame(terminal)],ignore_index=True).sort_values(['trade','elapsed'])
    tracker.to_csv(p/'daily_tracker.csv',index=False)
    print('Outer metrics saved',flush=True)


def a_source_calendar(p):return p/'calendar.csv'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','raw','market','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','tune','outer','all'],default='all')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PROTOCOL.read_text());(a.output/'protocol.json').write_bytes(PROTOCOL.read_bytes())
    if a.phase in ('prepare','all'):
        prepare(a,plan)
        import shutil
        shutil.copy2(a.source/'calendar.csv',a.output/'calendar.csv')
    d=pd.read_parquet(a.output/'landmarks.parquet')
    features=json.loads((a.output/'feature_contract.json').read_text())
    paths=np.load(a.source/'paths.npz')
    if a.phase in ('tune','all'):tune(a.output,plan,d,features,paths)
    if a.phase in ('outer','all'):
        selected=json.loads((a.output/'selection_frozen.json').read_text())
        assert selected['protocol_sha256']==sha(PROTOCOL)
        outer(a.output,plan,d,features,paths,selected['selections'])


if __name__=='__main__':main()
