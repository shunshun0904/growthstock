#!/usr/bin/env python3
"""E125: purchase-only zero-shot future-maximum forecasting; no production writes.

Run prepare, baseline, tsfm, evaluate in that order. Model inference can use a
different Python environment from data preparation. All row-level outputs stay
in the explicitly supplied private --output directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
import tsfm_exit as T
from exit_upside import block_interval

PROTOCOL = RESEARCH.parent / 'docs/experiments/tsfm-exit-2026-11-06/protocol.json'
LEVELS = np.array([.01, .05, .10, .15, .20, .25, .30, .35, .40, .45,
                   .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, .99])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    def default(v):
        if isinstance(v, (pd.Timestamp, np.datetime64)): return str(pd.Timestamp(v))
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default) + '\n')


def prepare(a, plan):
    cols = ['trade', 'Code', 'signal_date', 'entry_date', 'mature_date', 'asof_date',
            'elapsed', 'remaining', 'strict', 'spot_return', 'future_max_from_spot',
            'risk10', 'risk15', 'y10', 'y15']
    d = pd.read_parquet(a.source / 'landmarks.parquet', columns=cols)
    d = d[d.strict & d.elapsed.ge(1)].copy()
    prior = d[d.mature_date.lt(plan['outer_signal_dates'][0])].copy()
    q = d[d.signal_date.between(*plan['outer_signal_dates'])].copy()
    q = q.sort_values(['trade', 'elapsed']).reset_index(drop=True)
    assert prior.trade.nunique() == 215
    assert q.trade.nunique() == plan['expected_outer_trades'] and len(q) == 61 * 19
    assert set(prior.trade).isdisjoint(q.trade)
    assert q.signal_date.min() > pd.Timestamp(plan['tsfm']['weight_publication']).tz_localize(None)
    s = pd.read_parquet(a.source / 'source_signals.parquet')
    q['buy'] = s.entry_adjusted.reindex(q.trade).to_numpy()
    cal = pd.DatetimeIndex(pd.read_csv(a.source / 'calendar.csv', parse_dates=['Date']).Date)
    stock = pd.read_parquet(a.prices / 'adjusted_prices.parquet',
                            filters=[('Code', 'in', q.Code.unique().tolist())])
    contexts = np.full((len(q), plan['tsfm']['context']), np.nan, np.float32)
    q['close'] = np.nan; q['sigma_gap'] = np.nan; q['sigma_intraday'] = np.nan
    q['finite_context'] = 0
    forward_errors = []; spot_errors = []
    for code, rows in q.groupby('Code', sort=False):
        frame = stock[stock.Code.eq(code)].sort_values('Date').set_index('Date').reindex(cal)
        high = frame.AdjH.to_numpy(float)
        sg, si = T.volatility_estimates(frame)
        for i, row in rows.iterrows():
            j = cal.get_loc(row.asof_date); r = int(row.remaining)
            current = frame.AdjC.iloc[j]
            context = T.context_at(high, j, r, plan['tsfm']['context'], current)
            contexts[i, -len(context):] = context
            q.loc[i, ['close', 'sigma_gap', 'sigma_intraday', 'finite_context']] = [
                current, sg[j], si[j], np.isfinite(context).sum()]
            assert cal[j + r] == row.mature_date
            forward_errors.append(abs(np.max(high[j + 1:j + r + 1]) / current - 1 - row.future_max_from_spot))
            spot_errors.append(abs(current / row.buy - 1 - row.spot_return))
    assert max(forward_errors) < 3e-7 and max(spot_errors) < 3e-7
    bad = (q.finite_context.lt(plan['tsfm']['minimum_finite_context']) |
           ~np.isfinite(q[['close', 'sigma_gap', 'sigma_intraday']]).all(axis=1))
    excluded = sorted(q.loc[bad, 'trade'].unique().tolist())
    # Exclusions are fixed before any forecasts; remove whole trades only.
    keep = ~q.trade.isin(excluded)
    contexts = contexts[keep]; q = q[keep].reset_index(drop=True)
    q['qid'] = np.arange(len(q)); q['actual_max_spot'] = 1 + q.future_max_from_spot
    q['actual_max_buy'] = q.actual_max_spot * q.close / q.buy
    q['elapsed'] = q.elapsed.astype(int); q['remaining'] = q.remaining.astype(int)
    q.to_parquet(a.output / 'queries.parquet', index=False)
    np.save(a.output / 'contexts.npy', contexts)
    counts = []
    for day in range(1, 20):
        for target in (10, 15):
            z = prior[prior.elapsed.eq(day) & prior[f'risk{target}']]
            counts.append(dict(elapsed=day, target=target, n=len(z), hits=int(z[f'y{target}'].sum()),
                               p=(z[f'y{target}'].sum() + 1) / (len(z) + 2)))
    pd.DataFrame(counts).to_csv(a.output / 'age_prior.csv', index=False)
    ids = q.trade.drop_duplicates().to_numpy()
    with np.load(a.prices / 'paths.npz') as paths:
        np.savez_compressed(a.output / 'selected_paths.npz', trade=ids,
                            **{k: paths[k][ids] for k in ('o', 'h', 'l', 'c', 'dates')})
    identity = pd.read_csv(a.source / 'model_identity.csv')
    for row in identity.itertuples():
        file = RESEARCH / 'model' / ('meta.json' if row.model == 'lgbm' else f'models/{row.model}/meta.json')
        assert json.loads(file.read_text())['artifactHashes']['oof.parquet'] == row.sha256
    identity.to_csv(a.output / 'model_identity.csv', index=False)
    (a.output / 'protocol.json').write_bytes(PROTOCOL.read_bytes())
    audit = dict(protocol_sha256=sha(PROTOCOL), prior_trades=prior.trade.nunique(),
                 prior_last_mature=prior.mature_date.max(), outer_trades=len(ids),
                 outer_codes=q.Code.nunique(), signal_dates=q.signal_date.nunique(),
                 queries=len(q), excluded_trades=excluded, min_finite_context=int(q.finite_context.min()),
                 signal_start=q.signal_date.min(), signal_end=q.signal_date.max(),
                 max_spot_error=max(spot_errors), max_forward_error=max(forward_errors),
                 risks={str(t): dict(trades=q.loc[q[f'risk{t}'], 'trade'].nunique(),
                                    states=int(q[f'risk{t}'].sum())) for t in (10, 15)},
                 source_hashes={str(f): sha(f) for f in [a.source / 'landmarks.parquet',
                     a.source / 'source_signals.parquet', a.prices / 'adjusted_prices.parquet',
                     a.prices / 'paths.npz', a.source / 'calendar.csv', a.source / 'model_identity.csv']})
    dump(a.output / 'data_audit.json', audit)
    print('PREPARED', json.dumps({k:v for k,v in audit.items() if k != 'source_hashes'}, default=str), flush=True)


def baseline(a, plan):
    q = pd.read_parquet(a.output / 'queries.parquet')
    age = pd.read_csv(a.output / 'age_prior.csv').set_index(['elapsed', 'target'])
    records = []
    for row in q.itertuples():
        r = T.volatility_paths(row.remaining, row.sigma_gap, row.sigma_intraday,
                               plan['baselines']['volatility_paths'], plan['baselines']['seed'] + row.qid)
        rec = dict(qid=row.qid, method='volatility', max_mean=float(r.mean()), max_sd=float(r.std()),
                   **{f'q{v}':float(np.quantile(r,v/100)) for v in (10,25,50,75,90)})
        for t in (10, 15): rec[f'p{t}'] = float(np.mean(r >= row.buy / row.close * (1 + t / 100)))
        records.append(rec)
        records.append(dict(qid=row.qid, method='age_prior',
                            **{f'p{t}':float(age.loc[(row.elapsed,t), 'p']) for t in (10,15)}))
    pd.DataFrame(records).to_parquet(a.output / 'baseline_predictions.parquet', index=False)
    print('BASELINES', len(q), 'queries', flush=True)


def tsfm(a, plan):
    import torch
    import chronos
    from chronos import Chronos2Pipeline
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert sha(a.model / 'model.safetensors') == plan['tsfm']['weight_sha256']
    config = json.loads((a.model / 'config.json').read_text())
    assert np.allclose(config['chronos_config']['quantiles'], LEVELS)
    pipe = Chronos2Pipeline.from_pretrained(str(a.model), device_map='cpu', dtype=torch.float32)
    pipe.model.eval()
    q = pd.read_parquet(a.output / 'queries.parquet')
    contexts = np.load(a.output / 'contexts.npy')
    binding = {f:sha(a.output / f) for f in ('queries.parquet','contexts.npy','protocol.json')}
    binding['model'] = plan['tsfm']['weight_sha256']
    cache = a.output / 'chronos_quantiles.npz'
    marker = a.output / 'cache_binding.json'
    if cache.exists():
        assert json.loads(marker.read_text()) == binding, 'Stale cache; use a new output directory'
        with np.load(cache) as old: raw = old['raw'].copy(); done = old['done'].copy()
    else:
        raw = np.full((len(q), len(LEVELS)), np.nan); done = np.zeros(len(q), bool)
        dump(marker, binding)
    def predict(ix, r):
        with torch.inference_mode():
            out = pipe.predict([contexts[i][None,:] for i in ix], prediction_length=int(r),
                               context_length=plan['tsfm']['context'], batch_size=a.batch_size,
                               cross_learning=False)
        arr = np.array([v[0,:,-1].float().cpu().numpy() for v in out])
        if arr.shape != (len(ix),len(LEVELS)) or not np.isfinite(arr).all():
            raise ValueError('Non-finite or unexpected forecast; do not selectively drop queries')
        return arr
    # Different query states cannot attend to each other; test it in this runtime.
    ix = q.index[q.remaining.eq(19)].to_numpy()[:2]
    together = predict(ix, 19)
    separate = np.concatenate([predict([i], 19) for i in ix])
    isolation_error = float(np.max(abs(together-separate)))
    assert isolation_error < 1e-4, 'Batch-dependent forecast would risk temporal leakage'
    dump(a.output / 'inference_audit.json', dict(isolation_max_abs_log_error=isolation_error,
         torch=torch.__version__, chronos=chronos.__version__, dtype='float32', device='cpu',
         cross_learning=False, fine_tuned=False, model_sha256=plan['tsfm']['weight_sha256'],
         model_revision=plan['tsfm']['revision'], batch_size=a.batch_size, **binding))
    start = time.time()
    for r in sorted(q.remaining.unique(), reverse=True):
        ids = q.index[q.remaining.eq(r) & ~done].to_numpy()
        for j in range(0,len(ids),a.batch_size):
            ix = ids[j:j+a.batch_size]
            raw[ix] = predict(ix,r); done[ix] = True
            tmp = a.output / 'chronos_quantiles.tmp.npz'
            np.savez_compressed(tmp, raw=raw, done=done, levels=LEVELS)
            tmp.replace(cache)
        print(f'CHRONOS r={r} done={done.sum()}/{len(q)} seconds={time.time()-start:.1f}', flush=True)
    assert done.all()
    records = []
    for row in q.itertuples():
        prices = np.exp(raw[row.qid])
        rec = dict(qid=row.qid, method='chronos2', quantile_crossings=int((np.diff(prices)<0).sum()),
                   **{f'q{v}':float(np.sort(prices)[np.where(np.isclose(LEVELS,v/100))[0][0]])
                      for v in (10,25,50,75,90)})
        for t in (10,15):
            threshold = row.buy / row.close * (1+t/100)
            prob, mean, sd = T.distribution_summary(prices, LEVELS, threshold)
            pl, ml, sl = T.distribution_summary(prices, LEVELS, threshold, 'linear')
            rec.update({f'p{t}':prob, f'p{t}_linear':pl, 'max_mean':mean, 'max_sd':sd,
                        'max_mean_linear':ml, 'max_sd_linear':sl,
                        f'outside{t}':bool(threshold < min(prices) or threshold > max(prices))})
        records.append(rec)
    pd.DataFrame(records).to_parquet(a.output / 'tsfm_predictions.parquet', index=False)


def evaluate(a, plan):
    q = pd.read_parquet(a.output / 'queries.parquet')
    pred = pd.concat([pd.read_parquet(a.output / f) for f in
                      ('baseline_predictions.parquet','tsfm_predictions.parquet')], ignore_index=True)
    assert not pred.duplicated(['qid','method']).any()
    assert pred.groupby('method').size().eq(len(q)).all()
    d = pred.merge(q, on='qid', validate='many_to_one')
    cal = pd.DatetimeIndex(pd.read_csv(a.source / 'calendar.csv',parse_dates=['Date']).Date)
    scores = []; buckets = []; calibration = []; paired = []; maxima = []
    loss_rows = []
    for method, z in d.groupby('method'):
        for t in (10,15):
            risk = z[z[f'risk{t}']].copy(); p = risk[f'p{t}'].to_numpy(); y = risk[f'y{t}'].to_numpy(float)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            risk['brier'] = (p-y)**2
            clipped = np.clip(p,1e-6,1-1e-6)
            risk['logloss'] = -y*np.log(clipped)-(1-y)*np.log(1-clipped)
            loss_rows.append(risk[['qid','trade','signal_date','method','brier','logloss']].assign(target=t))
            scores.append(dict(method=method,target=t,trades=risk.trade.nunique(),states=len(risk),
                               brier=risk.brier.mean(),logloss=risk.logloss.mean(),
                               trade_equal_brier=risk.groupby('trade').brier.mean().mean(),
                               predicted=p.mean(),actual=y.mean(),endpoint_fraction=np.mean((p==0)|(p==1))))
            for lo,hi in ((1,5),(6,10),(11,15),(16,19)):
                b = risk[risk.elapsed.between(lo,hi)]
                buckets.append(dict(method=method,target=t,day_start=lo,day_end=hi,states=len(b),
                                    trades=b.trade.nunique(),brier=b.brier.mean(),predicted=b[f'p{t}'].mean(),actual=b[f'y{t}'].mean()))
            risk['bin'] = np.minimum((risk[f'p{t}']*5).astype(int),4)
            for group,b in risk.groupby('bin'):
                calibration.append(dict(method=method,target=t,bin=int(group),states=len(b),trades=b.trade.nunique(),
                                        predicted=b[f'p{t}'].mean(),actual=b[f'y{t}'].mean()))
        if z.max_mean.notna().all():
            error = z.max_mean-z.actual_max_spot
            maxima.append(dict(method=method,states=len(z),trades=z.trade.nunique(),mean_mae=np.abs(error).mean(),
                 mean_rmse=np.sqrt(np.mean(error**2)),mean_bias=error.mean(),median_mae=np.abs(z.q50-z.actual_max_spot).mean(),
                 q10_q90_coverage=((z.actual_max_spot>=z.q10)&(z.actual_max_spot<=z.q90)).mean(),
                 **{f'below_q{v}':(z.actual_max_spot<=z[f'q{v}']).mean() for v in (10,25,50,75,90)}))
    losses = pd.concat(loss_rows,ignore_index=True)
    for t in (10,15):
        for other in ('age_prior','volatility'):
            x = losses[losses.target.eq(t)&losses.method.eq('chronos2')]
            x = x.merge(losses[losses.target.eq(t)&losses.method.eq(other)][['qid','brier']],on='qid',suffixes=('_tsfm','_base'))
            x['delta'] = x.brier_tsfm-x.brier_base
            # The ratio of all weighted state losses; bootstrap resamples signal-date blocks.
            interval = block_interval(x.delta.to_numpy(),x.signal_date,cal)
            paired.append(dict(target=t,comparison='chronos2 - '+other,delta_brier=x.delta.mean(),ci_low=interval[0],ci_high=interval[1]))
    paths = np.load(a.output / 'selected_paths.npz')
    path_index = {int(v):i for i,v in enumerate(paths['trade'])}
    trades = q.drop_duplicates('trade').set_index('trade')
    exits = []
    for t in (10,15):
        for gap in ('limit','open'):
            for trade,meta in trades.iterrows():
                i = path_index[int(trade)]
                for method,rule in [('fixed','fixed')]+[(m,r) for m in ('age_prior','volatility','chronos2') for r in ('absolute','half')]:
                    part = d[d.trade.eq(trade)&d.method.eq(method)].sort_values('elapsed')
                    ps = np.full(19,np.nan) if method=='fixed' else part[f'p{t}'].to_numpy()
                    assert len(ps)==19
                    result = T.policy_exit(paths['o'][i],paths['h'][i],paths['c'][i],ps,t/100,rule,profit_gap=gap)
                    exits.append(dict(trade=trade,signal_date=meta.signal_date,target=t,profit_gap=gap,
                                      method=method,rule=rule,**result,net=result['gross']-plan['round_trip_cost']))
    ex = pd.DataFrame(exits); ex.to_parquet(a.output / 'exits.parquet',index=False)
    policy = []
    for key,z in ex.groupby(['target','profit_gap','method','rule'],sort=False):
        t,gap,method,rule = key
        b = ex[ex.target.eq(t)&ex.profit_gap.eq(gap)&ex.method.eq('fixed')][['trade','gross']]
        j = z.merge(b,on='trade',suffixes=('','_fixed'))
        delta = j.gross-j.gross_fixed
        ci = block_interval(delta,j.signal_date,cal)
        policy.append(dict(target=t,profit_gap=gap,method=method,rule=rule,trades=len(j),gross=z.gross.mean(),
                      net=z.net.mean(),win_rate=(z.net>0).mean(),mean_holding=z.exit_day.mean(),
                      early_exit_fraction=z.reason.eq('probability_exit').mean(),delta_fixed=delta.mean(),
                      ci_low=ci[0],ci_high=ci[1],**{f'net_cost_{cost:g}':z.gross.mean()-cost for cost in plan['cost_sensitivity']}))
    tail = d[d.method.eq('chronos2')]
    tail_stats = dict(mean_abs_change_mean=float(np.mean(abs(tail.max_mean_linear-tail.max_mean))),
                      max_abs_change_mean=float(np.max(abs(tail.max_mean_linear-tail.max_mean))),
                      mean_abs_change_sd=float(np.mean(abs(tail.max_sd_linear-tail.max_sd))),
                      crossings=int(tail.quantile_crossings.sum()),
                      **{f'outside{t}_fraction':float(tail[f'outside{t}'].mean()) for t in (10,15)},
                      **{f'p{t}_max_abs_tail_change':float(np.max(abs(tail[f'p{t}_linear']-tail[f'p{t}']))) for t in (10,15)})
    for name,rows in [('scores',scores),('day_buckets',buckets),('calibration',calibration),
                      ('paired_brier',paired),('maxima',maxima),('policies',policy)]:
        pd.DataFrame(rows).to_csv(a.output / f'{name}.csv',index=False)
    d.to_parquet(a.output / 'daily_tracker.parquet',index=False)
    dump(a.output / 'tail_sensitivity.json',tail_stats)
    print('PROBABILITY\n'+pd.DataFrame(scores).to_string(index=False),flush=True)
    print('PAIRED\n'+pd.DataFrame(paired).to_string(index=False),flush=True)
    print('POLICIES\n'+pd.DataFrame(policy).query("profit_gap=='limit'")[['target','method','rule','net','delta_fixed','ci_low','ci_high']].to_string(index=False),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='E124 results')
    p.add_argument('--prices',type=Path,required=True,help='E123 results')
    p.add_argument('--model',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--phase',choices=['prepare','baseline','tsfm','evaluate','all'],required=True)
    a = p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    plan = json.loads(PROTOCOL.read_text())
    if a.phase not in ('prepare','all'):
        assert sha(a.output/'protocol.json') == sha(PROTOCOL), 'Protocol changed after preparation'
    for phase,func in [('prepare',prepare),('baseline',baseline),('tsfm',tsfm),('evaluate',evaluate)]:
        if a.phase in (phase,'all'): func(a,plan)


if __name__=='__main__': main()
