#!/usr/bin/env python3
"""Matched retuning of A(existing), C(inputs), B(theoretical value).

Same candidate bank, training rows, 3 forward CV windows and 30 candidates per
condition. Select separately by CV only. Freeze ALL selections before reading
holdout predictions. The holdout has been observed in E120, so is exploratory.
No production files, parameters, or models are written.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ltr
import ltr5
import tuning
import e119_capital_efficiency as E
import e120_theoretical_value as V

TASKS = ("class10", "class15", "ltr", "ltr5")
ARMS = ("A", "C", "B")
SEEDS = (42, 43, 44)
SEARCH_KEYS = ("learning_rate", "num_leaves", "min_child_samples", "subsample",
               "colsample_bytree", "reg_alpha", "reg_lambda")


def atomic_json(path, value):
    path=Path(path)
    temp=path.with_suffix('.tmp')
    E.write_json(temp,value)
    temp.replace(path)


def candidate_bank(previous, n_trials=30, seed=42):
    """One previous setting + random settings, identical across feature arms.

    A fixed bank avoids adaptive-search luck giving arms different candidates.
    Each arm still chooses its own optimum from its own validation scores.
    """
    import optuna
    if n_trials < 2:
        raise ValueError('At least two candidates required for retuning')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed),direction='maximize')
    bank=[{k:previous[k] for k in SEARCH_KEYS}]
    for _ in range(n_trials-1):
        trial=study.ask()
        params=ltr.search_space(trial)
        bank.append({k:params[k] for k in SEARCH_KEYS})
        study.tell(trial,0.)  # No holdout or CV result is involved in bank generation.
    return bank


def forward_folds(train, kind, n_splits=3, months=6, min_train_months=36):
    """Expanding date groups with exact per-company label-maturity purging."""
    dates=pd.to_datetime(train.Date)
    end,first=dates.max(),dates.min()
    folds=[];checks=[]
    for k in range(n_splits,0,-1):
        lo=end-pd.DateOffset(months=months*k)
        hi=end-pd.DateOffset(months=months*(k-1))
        vi=np.flatnonzero(((dates>lo)&(dates<=hi)).to_numpy())
        if not len(vi):
            raise ValueError('Empty inner validation window')
        start=dates.iloc[vi].min()
        ti=np.flatnonzero((dates.lt(start)&train.maturity_date.lt(start)).to_numpy())
        if not len(ti) or dates.iloc[ti].max()<first+pd.DateOffset(months=min_train_months):
            raise ValueError('Insufficient history for forward fold')
        if kind.startswith('class'):
            for ix in [ti,vi]:
                if train.iloc[ix]['y_'+kind].nunique()!=2:
                    raise ValueError('Both classes required in each CV side')
        elif (train.iloc[vi].groupby('Date').size()>=5).sum()==0:
            raise ValueError('No eligible ranking validation day')
        assert train.iloc[ti].maturity_date.max()<start
        assert not set(dates.iloc[ti])&set(dates.iloc[vi])
        folds.append((ti,vi))
        checks.append({'fold':len(folds),'train_rows':len(ti),'valid_rows':len(vi),
                       'train_from':dates.iloc[ti].min(),'train_to':dates.iloc[ti].max(),
                       'train_max_outcome':train.iloc[ti].maturity_date.max(),
                       'valid_from':start,'valid_to':dates.iloc[vi].max(),
                       'valid_max_outcome':train.iloc[vi].maturity_date.max()})
    return folds,checks


class Trainer:
    """Reuse training-only binning, without filtering features for one leaf size.

    CV and final fits use this same native LightGBM path. Production label_gain
    and class weighting are preserved. Dataset randomness is fixed across arms
    and model seeds. No validation rows participate in bin construction.
    """
    def __init__(self, frame, columns, kind, threads=2):
        import lightgbm as lgb
        if not frame.Date.is_monotonic_increasing:
            raise ValueError('Training rows must be sorted by date')
        self.kind,self.threads=kind,threads
        self.y=(frame['y_'+kind].to_numpy(dtype=int) if kind.startswith('class')
                else ltr.grades(frame.Date,frame[ltr.R20]))
        group=None if kind.startswith('class') else frame.groupby('Date',sort=True).size().to_numpy()
        self.dataset=lgb.Dataset(frame[columns].to_numpy(dtype=float),label=self.y,group=group,
            feature_name=list(columns),params={'feature_pre_filter':False,'data_random_seed':42,
                                               'num_threads':threads,'verbosity':-1},free_raw_data=True)
        self.dataset.construct()

    def fit(self, candidate, seed=42, trees=200):
        import lightgbm as lgb
        params=dict(candidate,boosting_type='gbdt',subsample_freq=1,feature_pre_filter=False,
                    data_random_seed=42,num_threads=self.threads,seed=seed,verbosity=-1,metric='None')
        if self.kind.startswith('class'):
            params.update(objective='binary',scale_pos_weight=tuning.scale_pos_weight(self.y))
        else:
            params.update(objective='lambdarank',label_gain=list(range(ltr.N_GRADES)))
        return lgb.train(params,self.dataset,num_boost_round=trees)


def primary(kind, frame, score):
    if kind.startswith('class'):
        return float(average_precision_score(frame['y_'+kind],score))
    return float(ltr.ndcg_at_k(frame.Date,score,ltr.grades(frame.Date,frame[ltr.R20])))


def select_candidate(records):
    """Only CV mean selects the winner; deterministic earlier-id tie break."""
    if not records or not all(np.isfinite(r['cv_mean']) for r in records):
        raise ValueError('Incomplete/nonfinite CV record')
    return max(records,key=lambda r:(r['cv_mean'],-r['candidate']))


def prepare(args):
    import lightgbm,sklearn,optuna
    files=[args.dataset,args.prices,Path(__file__),HERE/'exp/e120_theoretical_value.py',HERE/'exp/e119_capital_efficiency.py',
           HERE/'theoretical_value.py',HERE/'features.py',HERE/'label_specs.py',HERE/'ltr.py',HERE/'ltr5.py',
           HERE/'live_track.py',HERE/'train_model.py',HERE/'tuning.py']
    files += [HERE/n for n in ['lgbm_params.json','multi_params.json','ltr_params.json','ltr5_params.json']]
    source={'files':{str(p):E.digest(p) for p in files},
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                       'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'optuna':optuna.__version__}}
    data_key=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    p=args.output
    if (p/'prepared.json').exists() and json.loads((p/'prepared.json').read_text())['data_key']==data_key:
        return source,json.loads((p/'outer_split.json').read_text())
    print('Preparing fixed, identical data and exact outcome boundaries',flush=True)
    daily=pd.read_parquet(args.dataset)
    daily,pool,bars,checks=V.prepare_ranking(daily,args.prices)
    del bars
    train_end,test_start,test_end=V.T.holdout_bounds(daily.Date)
    last=E.parameter_outcome_end(daily,pool,train_end)
    test_start=max(test_start,last+pd.Timedelta(days=1))
    for name,labels in E.classification_labels(daily).items():daily[name]=labels
    np.testing.assert_array_equal(daily.y_class10.to_numpy(dtype=int),daily.label.to_numpy(dtype=int))
    split={'train_end':train_end,'test_start':test_start,'test_end':test_end,
           'last_training_outcome':last,'ranking_checks':checks,'tasks':[]}
    for kind in TASKS:
        d=pool if kind=='ltr5' else daily
        valid=d[ltr.R20].notna()&d.maturity_date.notna()
        if kind.startswith('class'):valid &= d['y_'+kind].notna()
        elif kind=='ltr5':valid &= ltr5.trainable(d)
        tr=d[valid&d.Date.le(train_end)&d.maturity_date.lt(test_start)].sort_values(['Date','Code']).reset_index(drop=True)
        te=d[valid&d.Date.between(test_start,test_end)].sort_values(['Date','Code']).reset_index(drop=True)
        assert tr.maturity_date.max()<te.Date.min()
        tr.to_parquet(p/(kind+'_train.parquet'),index=False)
        te.to_parquet(p/(kind+'_holdout.parquet'),index=False)
        split['tasks'].append({'task':kind,'train_rows':len(tr),'test_rows':len(te),
             'train_max_outcome':tr.maturity_date.max(),'test_min_date':te.Date.min(),'test_max_date':te.Date.max()})
    atomic_json(p/'outer_split.json',split)
    atomic_json(p/'prepared.json',{'data_key':data_key})
    return source,json.loads((p/'outer_split.json').read_text())


def run(args):
    p=args.output
    source,outer=prepare(args)
    source['experiment_sha256']=E.digest(__file__)
    atomic_json(p/'inputs.json',source)
    previous=E.parameters()
    banks={kind:candidate_bank(previous[kind],args.trials,args.search_seed) for kind in TASKS}
    fold_info={}
    for kind in TASKS:
        tr=pd.read_parquet(p/(kind+'_train.parquet'))
        _,fold_info[kind]=forward_folds(tr,kind,args.folds)
        assert max(pd.Timestamp(x['valid_max_outcome']) for x in fold_info[kind])<pd.Timestamp(outer['test_start'])
    plan={'arms':{'A':'existing 360 features (364 for rolling)','C':'A + 13 audited inputs/quality',
                  'B':'C + 10 theoretical-value statistics'},'tasks':TASKS,'trials_per_task_arm':args.trials,
          'inner_folds':args.folds,'inner_window_months':6,'inner_scheme':'expanding, exact company outcome purge',
          'search':'Optuna RandomSampler, matched candidate bank; previous setting + random settings',
          'search_seed':args.search_seed,'candidate_banks':banks,'trees':200,'final_model_seeds':SEEDS,
          'objectives':{'classification':'mean fold AP','ranking':'mean fold daily NDCG@5'},
          'cv_fit_seed':42,'binning_seed':42,'feature_pre_filter':False,'early_stopping':False,
          'per_arm_selection':'own maximum mean CV, never holdout','inner_fold_checks':fold_info,'outer_split':outer,
          'test_previously_observed':True,'production_changed':False,'tuning_seed_repeats':1,
          'expected_cv_fits':len(TASKS)*len(ARMS)*args.trials*args.folds,
          'expected_final_fits':len(TASKS)*len(ARMS)*len(SEEDS)}
    atomic_json(p/'experiment_plan.json',plan)
    fingerprint=E.digest(p/'inputs.json')+E.digest(p/'experiment_plan.json')
    selected={};total_trials=0
    for kind in TASKS:
        tr=pd.read_parquet(p/(kind+'_train.parquet'))
        folds,_=forward_folds(tr,kind,args.folds)
        for arm in ARMS:
            path=p/(kind+'_'+arm+'_trials.json')
            records=[]
            if path.exists():
                cache=json.loads(path.read_text())
                if cache['fingerprint']!=fingerprint:raise ValueError('Different experiment fingerprint; use a new output directory')
                records=cache['trials']
            cols=V.column_sets(kind=='ltr5')[arm]
            if len(records)<args.trials:
                cv=[]
                for ti,vi in folds:
                    valid=tr.iloc[vi]
                    cv.append((Trainer(tr.iloc[ti],cols,kind,args.threads),valid,valid[cols].to_numpy(dtype=float)))
                for j in range(len(records),args.trials):
                    start=time.monotonic();scores=[]
                    for trainer,valid,xv in cv:
                        model=trainer.fit(banks[kind][j])
                        score=model.predict(xv,num_threads=args.threads)
                        scores.append(primary(kind,valid,score))
                    records.append({'candidate':j,'params':banks[kind][j],'cv_folds':scores,
                                    'cv_mean':float(np.mean(scores)),'seconds':round(time.monotonic()-start,2)})
                    atomic_json(path,{'fingerprint':fingerprint,'trials':records})
                    print(json.dumps({'phase':'tune','task':kind,'arm':arm,'completed':j+1,'budget':args.trials,
                                      'seconds':records[-1]['seconds']},ensure_ascii=False),flush=True)
                del cv,model
                gc.collect()
            selected[kind+'_'+arm]=select_candidate(records)
            total_trials+=len(records)
        del tr
        gc.collect()
    # A global barrier: persist every chosen parameter BEFORE scoring any arm.
    atomic_json(p/'selection_frozen.json',{'fingerprint':fingerprint,'selections':selected,'trials':total_trials})
    freeze_sha=E.digest(p/'selection_frozen.json')
    results=[];seeds=[];quarters=[]
    for kind in TASKS:
        tr=pd.read_parquet(p/(kind+'_train.parquet'))
        te=pd.read_parquet(p/(kind+'_holdout.parquet'))
        outputs=te[['Code','Date',ltr.R20]].copy()
        for arm in ARMS:
            cols=V.column_sets(kind=='ltr5')[arm]
            trainer=Trainer(tr,cols,kind,args.threads)
            xt=te[cols].to_numpy(dtype=float);predictions=[]
            for seed in SEEDS:
                cache_path=p/(kind+'_'+arm+'_'+str(seed)+'_score.npy')
                if cache_path.exists() and (p/'score_fingerprint.json').exists() and json.loads((p/'score_fingerprint.json').read_text()).get('selection_sha256')==freeze_sha:
                    score=np.load(cache_path,allow_pickle=False)
                else:
                    model=trainer.fit(selected[kind+'_'+arm]['params'],seed=seed)
                    score=model.predict(xt,num_threads=args.threads)
                    np.save(cache_path,score,allow_pickle=False)
                if score.shape!=(len(te),) or not np.isfinite(score).all():raise ValueError('Invalid final scores')
                predictions.append(score)
                seeds.append({'task':kind,'arm':arm,'seed':seed,**E.evaluate(kind,te,score)})
            score=np.mean(predictions,axis=0);outputs[arm]=score
            for subset,mask in [('all',np.ones(len(te),bool)),('ri_available',te.tv_ri_mean_rel.notna().to_numpy()),('ddm_available',te.tv_ddm_mean_rel.notna().to_numpy())]:
                results.append({'task':kind,'arm':arm,'subset':subset,'features':len(cols),**E.evaluate(kind,te.loc[mask],score[mask])})
            for q in sorted(te.Date.dt.to_period('Q').unique()):
                mask=te.Date.dt.to_period('Q').eq(q).to_numpy()
                if mask.sum()>=100 and te.loc[mask,'Date'].nunique()>=20:
                    quarters.append({'task':kind,'arm':arm,'quarter':str(q),**E.evaluate(kind,te.loc[mask],score[mask])})
            pd.DataFrame(results).to_csv(p/'results.csv',index=False)
            pd.DataFrame(seeds).to_csv(p/'seed_results.csv',index=False)
            pd.DataFrame(quarters).to_csv(p/'quarter_results.csv',index=False)
            print(json.dumps({'phase':'evaluate','task':kind,'arm':arm,'seeds':len(SEEDS)}),flush=True)
            del trainer
            gc.collect()
        outputs.to_parquet(p/(kind+'_predictions.parquet'),index=False)
    atomic_json(p/'score_fingerprint.json',{'selection_sha256':freeze_sha})
    atomic_json(p/'completed.json',{'studies':len(selected),'candidates_per_study':args.trials,
        'cv_fits':total_trials*args.folds,'final_fits':len(seeds),'production_changed':False,
        'test_previously_observed':True,'selection_sha256':freeze_sha})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','prices','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--trials',type=int,default=30)
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--search-seed',type=int,default=42)
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    run(args)


if __name__=='__main__':main()
