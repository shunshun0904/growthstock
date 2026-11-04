#!/usr/bin/env python3
"""Predeclared exploratory CAPM / residual-income / dividend-value backtest.

Opt-in only. Frozen production parameters, identical rows, mature-outcome
embargo, 4 LightGBM tasks x 6 arms x 3 seeds. Raw financial data stay private.
Run --prepare-only for coverage without model fitting. Re-running resumes only
matching source/plan fingerprints. No production registration or retraining.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import theoretical_value as V
import features as F
import ltr
import ltr5
import train_model as T
import e119_capital_efficiency as E

SEEDS = (42, 43, 44)
ARMS = ("A", "C", "R", "D", "B", "P")
PERM_SEED = 20261103
QUALITY = ["missing_tv_ri", "missing_tv_ddm"]


def column_sets(rolling=False):
    base = F.columns("all_plus_prog_listing_vol_book_shape_dd_recent_tx") + (list(ltr5.MOVE_COLS) if rolling else [])
    controls = base + V.CONTROLS + QUALITY
    return {"A": base, "C": controls,
            "R": controls + [x for x in V.VALUES if x.startswith("tv_ri")],
            "D": controls + [x for x in V.VALUES if x.startswith("tv_ddm")],
            "B": controls + V.VALUES, "P": controls + V.VALUES}


def shuffled_values(df):
    """Joint shuffle within date AND availability pattern, preserving identities."""
    out = df.copy()
    rng = np.random.default_rng(PERM_SEED)
    for ix in out.groupby(["Date"] + QUALITY, sort=True).indices.values():
        out.loc[out.index[ix], V.VALUES] = df.iloc[rng.permutation(ix)][V.VALUES].to_numpy()
    np.testing.assert_array_equal(out[V.VALUES].isna(), df[V.VALUES].isna())
    return out


def input_fingerprint(args):
    import lightgbm, sklearn
    paths = [args.dataset, args.prices, args.yields, Path(__file__), HERE/'theoretical_value.py',
             HERE/'exp/e119_capital_efficiency.py', HERE/'features.py', HERE/'label_specs.py',
             HERE/'ltr.py', HERE/'ltr5.py', HERE/'live_track.py', HERE/'tuning.py', HERE/'train_model.py']
    paths += [HERE/x for x in ['lgbm_params.json','multi_params.json','ltr_params.json','ltr5_params.json']]
    paths += sorted(args.data_dir.glob('fins_*.parquet')) + sorted(args.data_dir.glob('topix_*.parquet'))
    return {"files": {str(p): E.digest(p) for p in paths},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__}}


def prepare(args):
    p = args.output
    source = input_fingerprint(args)
    if (p/'inputs.json').exists() and json.loads((p/'inputs.json').read_text()) == source and (p/'dataset.parquet').exists():
        return pd.read_parquet(p/'dataset.parquet')
    original = pd.read_parquet(args.dataset)
    original['Date'] = pd.to_datetime(original.Date)
    prices = pd.read_parquet(args.prices, columns=['Code','Date','close_raw','close'])
    prices = prices[prices.Code.isin(set(original.Code)) & prices.Date.le(original.Date.max())]
    yields = pd.read_csv(args.yields, skiprows=1, na_values=['-'])
    yields['Date'] = pd.to_datetime(yields.Date)
    market = pd.concat([pd.read_parquet(x) for x in sorted(args.data_dir.glob('topix_*.parquet'))], ignore_index=True)
    market['Date'] = pd.to_datetime(market.Date)
    print('Building trailing betas', flush=True)
    b = V.beta(original, prices, market, yields)
    rf = V.riskfree(original.Date, yields)
    cols = ['Code','DocType','DiscDate','DiscTime','CurPerType','CurFYSt','CurPerEn','NP','ShEq','ShOutFY','TrShFY','DivTotalAnn']
    fins = pd.concat([pd.read_parquet(x, columns=cols) for x in sorted(args.data_dir.glob('fins_*.parquet'))], ignore_index=True)
    print('Reconstructing disclosed financial states', flush=True)
    fin = V.financial_inputs(original, fins, prices)
    daily = pd.concat([original.reset_index(drop=True), b, rf, fin], axis=1)
    daily = V.valuations(daily)
    for name in ['ri','ddm']:
        daily['missing_tv_'+name] = daily['tv_'+name+'_mean_rel'].isna().astype(float)
    for c in original:
        pd.testing.assert_series_equal(original[c].reset_index(drop=True), daily[c])
    for c in ['tv_fin_source_date','tv_div_source_date','tv_rf_date']:
        if not (daily[c].isna() | daily[c].lt(daily.Date)).all():
            raise ValueError('Future/unreleased input: '+c)
    if np.isinf(daily[V.VALUES+V.CONTROLS].to_numpy(dtype=float)).any():
        raise ValueError('Non-finite feature')
    coverage = pd.DataFrame([{'feature': c, 'rows': int(daily[c].notna().sum()), 'pct': 100*daily[c].notna().mean()}
                            for c in V.VALUES+V.CONTROLS])
    coverage.to_csv(p/'coverage.csv', index=False)
    daily.to_parquet(p/'dataset.parquet', index=False)
    E.write_json(p/'data_checks.json', {'rows':len(daily), 'codes':int(daily.Code.nunique()),
        'date_min': daily.Date.min(), 'date_max': daily.Date.max(), 'existing_columns_identical':True,
        'financial_reference_strictly_before_candidate':True, 'rf_reference_strictly_before_candidate':True,
        'original_historical_api_versions_independently_verified':False,
        'yield_reference_last':yields.Date.max(), 'market_last':market.Date.max()})
    E.write_json(p/'inputs.json', source)
    return daily


def prepare_ranking(daily, prices):
    bars = pd.read_parquet(prices, columns=['Code','Date','open','high','low','close'])
    bars = bars[bars.Code.isin(set(daily.Code)) & bars.Date.ge(daily.Date.min()-pd.Timedelta(days=10))]
    bars = bars.rename(columns={'open':'AdjO','high':'AdjH','low':'AdjL','close':'AdjC'})
    daily = ltr.attach_r20(daily, bars)
    daily['maturity_date'] = E.forward_maturity(daily, bars).to_numpy()
    cal = ltr5.calendar(bars.Date.unique())
    pool = ltr5.pool_frame(daily, bars, cal, column_sets()['B'],
        decision_days=cal[(cal>=daily.Date.min()) & (cal<=daily.Date.max())], keep=('label','tv_fin_source_date','tv_rf_date'))
    pool = ltr5.attach_exit(pool, bars)
    pool['maturity_date'] = E.forward_maturity(pool,bars).to_numpy()
    move, same = ltr5.check_move_cols(pool,bars,cal), ltr5.check_same_day(pool,daily)
    if move['same'] != move['rows'] or same['mismatch']:
        raise ValueError('Pool differs from production definitions')
    for c in ['tv_fin_source_date','tv_rf_date']:
        if not (pool[c].isna() | pool[c].lt(pool.cand_date)).all():
            raise ValueError('Rolling candidate information was refreshed')
    return daily, pool, bars, {'rolling_rows':len(pool),'move_checks':move,'same_day_checks':same}


def forward_returns(keys, bars, hold):
    out = np.full(len(keys),np.nan)
    maturity = np.full(len(keys),np.datetime64('NaT'),dtype='datetime64[ns]')
    wanted = {c:g.index.to_numpy() for c,g in keys.reset_index(drop=True).groupby('Code',sort=False)}
    for code,g in bars.groupby('Code',sort=False):
        if code not in wanted:
            continue
        g=g.sort_values('Date'); dates=g.Date.to_numpy(dtype='datetime64[ns]')
        ix=wanted[code]; pos=np.searchsorted(dates,keys.iloc[ix].Date.to_numpy(dtype='datetime64[ns]'))
        ok=(pos+hold<len(g)) & (dates[np.minimum(pos,len(dates)-1)]==keys.iloc[ix].Date.to_numpy(dtype='datetime64[ns]'))
        j=pos[ok]; entry=g.AdjO.to_numpy()[j+1]; exit_=g.AdjC.to_numpy()[j+hold]
        good=(entry>0)&np.isfinite(entry)&np.isfinite(exit_)
        out[ix[ok][good]]=exit_[good]/entry[good]-1
        maturity[ix[ok][good]]=dates[j[good]+hold]
    return out, maturity


def value_sorts(daily, bars, test_start, test_end, output):
    d=daily[daily.Date.between(test_start,test_end)].copy().reset_index(drop=True)
    rows=[]; icrows=[]
    for h in [20,60,120]:
        ret,maturity=forward_returns(d,bars,h)
        for model in ['ri','ddm']:
            score='tv_'+model+'_median_rel'
            z=d[['Code','Date',score]].assign(ret=ret,maturity=pd.to_datetime(maturity)).dropna()
            z=z[z.groupby('Date')[score].transform('count').ge(5) & z.groupby('Date')[score].transform('nunique').ge(2)]
            if z.empty:
                continue
            # Average ranks keep equal values together; daily buckets may be uneven.
            z['bucket']=np.minimum(5,np.ceil(z.groupby('Date')[score].rank(pct=True,method='average')*5)).astype(int)
            for bucket,g in z.groupby('bucket'):
                byday=g.groupby('Date').ret.mean()
                rows.append({'model':model,'horizon':h,'bucket':int(bucket),'rows':len(g),'days':len(byday),
                             'daily_mean_return_pct':100*byday.mean(), 'median_sample_return_pct':100*g.ret.median(),
                             'positive_fraction':g.ret.gt(0).mean(), 'last_signal_date':g.Date.max(), 'last_exit_date':g.maturity.max()})
            cor=[]
            for date,g in z.groupby('Date'):
                if g.ret.nunique()>=2:
                    cor.append(g[score].corr(g.ret,method='spearman'))
            icrows.append({'model':model,'horizon':h,'rows':len(z),'days':len(cor),'mean_daily_spearman':np.mean(cor)})
    pd.DataFrame(rows).to_csv(output/'valuation_sorts.csv',index=False)
    pd.DataFrame(icrows).to_csv(output/'valuation_ic.csv',index=False)


def run(args):
    import lightgbm as lgb
    daily=prepare(args)
    if args.prepare_only:
        return
    daily,pool,bars,checks=prepare_ranking(daily,args.prices)
    train_end,test_start,test_end=T.holdout_bounds(daily.Date)
    latest=E.parameter_outcome_end(daily,pool,train_end)
    if pd.isna(latest):
        raise ValueError('Unknown training outcome boundary')
    test_start=max(test_start,latest+pd.Timedelta(days=1))
    for name,values in E.classification_labels(daily).items():
        daily[name]=values
    np.testing.assert_array_equal(daily.y_class10.to_numpy(dtype=int),daily.label.to_numpy(dtype=int))
    params=E.parameters()
    plan={'arms':{'A':'existing features','C':'A + audited financial/CAPM inputs, coverage and missingness',
                  'R':'C + RI five price-relative statistics','D':'C + DDM five price-relative statistics',
                  'B':'C + both families (10 values)','P':'C + jointly date/availability-shuffled ten values'},
          'seeds':SEEDS,'permutation_seed':PERM_SEED,'trees':200,'train_end':train_end,'test_start':test_start,'test_end':test_end,
          'latest_parameter_training_outcome':latest,'parameters':params,'ranking_checks':checks,
          'ERP':V.ERP,'horizons':V.HORIZONS,'payouts':V.PAYOUTS,'growth':V.GROWTH,'beta_window':504,'beta_min_obs':252,
          'beta_shrinkage':'2/3 raw + 1/3 (declared prior)','DDM_min_ke_minus_g':.01,'scenario_valid_fraction_min':.8,
          'primary_metrics':{'class10':'AP','class15':'AP','ltr':'daily NDCG@5','ltr5':'pool NDCG@5'},
          'rolling_feature_date':'candidate date, frozen','re_tuned':False,'adoption_test':False,
          'classifier_imbalance':'production tuning.scale_pos_weight(y_train)',
          'limitations':['previously examined single holdout','price-return beta excludes cash dividends',
                         'clean-surplus and fade are assumptions, not company forecasts','10Y JGB is a risk-free proxy, not zero curve',
                         'original historical API versions not independently verified','no costs/liquidity constraints/independent trade accounting',
                         '60/120-day sorts are right-censored, not same sample as 20-day']}
    E.write_json(args.output/'experiment_plan.json',plan)
    value_sorts(daily,bars,test_start,test_end,args.output)
    del bars
    cache=args.output/'score_cache';cache.mkdir(exist_ok=True)
    fingerprint=E.digest(args.output/'inputs.json')+E.digest(args.output/'experiment_plan.json')
    results,seeds,windows,splits=[],[],[],[]
    for kind in ['class10','class15','ltr','ltr5']:
        d=(pool if kind=='ltr5' else daily).copy()
        valid=d[ltr.R20].notna() & d.maturity_date.notna()
        if kind.startswith('class'):
            valid &= d['y_'+kind].notna()
        elif kind=='ltr5':
            valid &= ltr5.trainable(d)
        tr=d[valid & d.Date.le(train_end) & d.maturity_date.lt(test_start)].sort_values(['Date','Code'],kind='mergesort').reset_index(drop=True)
        te=d[valid & d.Date.between(test_start,test_end)].sort_values(['Date','Code'],kind='mergesort').reset_index(drop=True)
        assert tr.maturity_date.max()<te.Date.min()
        splits.append({'task':kind,'train_rows':len(tr),'test_rows':len(te),'train_max_date':tr.Date.max(),
                       'train_max_outcome':tr.maturity_date.max(),'test_min_date':te.Date.min(),'test_max_date':te.Date.max()})
        E.write_json(args.output/'split_checks.json',splits)
        ptr,pte=shuffled_values(tr),shuffled_values(te)
        outputs=te[['Code','Date',ltr.R20]+V.VALUES].copy()
        cols_of=column_sets(kind=='ltr5')
        for arm in ARMS:
            cols=cols_of[arm]; xtr,xte=(ptr,pte) if arm=='P' else (tr,te); scores=[]
            for seed in SEEDS:
                start=time.monotonic()
                key=hashlib.sha256((fingerprint+kind+arm+str(seed)).encode()).hexdigest()
                path=cache/(key+'.npy')
                if path.exists():
                    score=np.load(path,allow_pickle=False)
                    if score.shape!=(len(te),) or not np.isfinite(score).all():
                        raise ValueError('Invalid cached predictions')
                elif kind.startswith('class'):
                    y=tr['y_'+kind].to_numpy(dtype=int)
                    model=lgb.LGBMClassifier(**E.classifier_params(params[kind],y,seed,args.threads))
                    model.fit(xtr[cols].to_numpy(dtype=float),y)
                    score=model.predict_proba(xte[cols].to_numpy(dtype=float))[:,1]
                else:
                    p=dict(params[kind],n_jobs=args.threads,random_state=seed,n_estimators=200,verbose=-1)
                    model=ltr.fit(xtr,cols,p,seed=seed)
                    score=ltr.predict(model,xte[cols].to_numpy(dtype=float))
                if not path.exists():
                    with path.with_suffix('.tmp').open('wb') as f:
                        np.save(f,score,allow_pickle=False)
                    path.with_suffix('.tmp').replace(path)
                scores.append(score)
                seeds.append({'task':kind,'arm':arm,'seed':seed,**E.evaluate(kind,te,score)})
                print(json.dumps({'task':kind,'arm':arm,'seed':seed,'seconds':round(time.monotonic()-start,1)}),flush=True)
            score=np.mean(scores,axis=0); outputs[arm]=score
            for subset,mask in [('all',np.ones(len(te),bool)),('ri_available',te.tv_ri_mean_rel.notna().to_numpy()),('ddm_available',te.tv_ddm_mean_rel.notna().to_numpy())]:
                if mask.any():
                    results.append({'task':kind,'arm':arm,'subset':subset,'features':len(cols),**E.evaluate(kind,te.loc[mask],score[mask])})
            for period in sorted(te.Date.dt.to_period('Q').unique()):
                mask=te.Date.dt.to_period('Q').eq(period).to_numpy()
                if mask.sum()>=100 and te.loc[mask,'Date'].nunique()>=20:
                    windows.append({'task':kind,'arm':arm,'quarter':str(period),**E.evaluate(kind,te.loc[mask],score[mask])})
            pd.DataFrame(results).to_csv(args.output/'results.csv',index=False)
            pd.DataFrame(seeds).to_csv(args.output/'seed_results.csv',index=False)
            pd.DataFrame(windows).to_csv(args.output/'quarter_results.csv',index=False)
        outputs.to_parquet(args.output/(kind+'_predictions.parquet'),index=False)
    E.write_json(args.output/'completed.json',{'tasks':4,'arms':6,'seeds':len(SEEDS),'fits':len(seeds),
                                             'production_changed':False,'premium_changed':False})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','prices','data-dir','yields','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args(argv)
    if args.dataset.resolve()==(args.output/'dataset.parquet').resolve():
        raise ValueError('Cannot replace source dataset')
    args.output.mkdir(parents=True,exist_ok=True)
    run(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
