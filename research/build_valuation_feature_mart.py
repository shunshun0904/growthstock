#!/usr/bin/env python3
"""Generate past-only daily features for EVERY existing training security.

Daily histories are calculated before sampling, then only the original
security/date keys and derived research features are persisted. This avoids
publishing or storing thousands of full scenario matrices. Missing source
histories remain explicit missing rows, never a future-selected sub-universe.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_roic_wacc_series as B
import capital_efficiency as CE
import roic_wacc_valuation as R
import valuation_gap as G

FIELDS = list(dict.fromkeys(G.CONTROLS+G.GAPS+G.TIME+G.MARKET_CONTROLS+G.RIM_FIELDS))
_STATE = {}


def initialize_worker(raw,dataset):
    keys=pd.read_parquet(dataset,columns=['Code','Date'])
    keys['Code']=keys.Code.map(B.code5)
    keys['Date']=pd.to_datetime(keys.Date)
    keys=keys.drop_duplicates().sort_values(['Code','Date'])
    prices,fins,market,rates,fin,companies,_=B.load_inputs(raw,sorted(keys.Code.unique()))
    fins['Code']=fins.Code.map(B.code5)
    annual=R.reviewed_states(CE.annual_panel(fin,CE.periods_from_fins(fins),companies),market.Date)
    _STATE.update(pg={c:g for c,g in prices.groupby('Code',sort=False)},
                  fg={c:g for c,g in fins.groupby('Code',sort=False)},
                  ag={c:g for c,g in annual.groupby('Code',sort=False)},
                  kg={c:g for c,g in keys.groupby('Code',sort=False)},market=market,rates=rates,
                  empty_prices=prices.iloc[:0],empty_fins=fins.iloc[:0],empty_annual=annual.iloc[:0])


def build_chunk(offset,batch,output,fingerprint,resume,chunk_codes):
    path=output/f'features-{offset//chunk_codes:04d}.parquet'
    tag=path.with_suffix('.meta.json')
    if resume and path.exists() and tag.exists():
        old=json.loads(tag.read_text())
        if old.get('fingerprint')==fingerprint and old.get('sha256')==B.digest(path):
            return old['coverage']
    rows,records=[],[]
    for code in batch:
        wanted=_STATE['kg'][code]
        p=_STATE['pg'].get(code,_STATE['empty_prices'])
        p=p[p.Date.le(wanted.Date.max())]
        f=_STATE['fg'].get(code,_STATE['empty_fins'])
        if p.empty:
            history=pd.DataFrame()
        else:
            states=B.annual_dupont_inputs(_STATE['ag'].get(code,_STATE['empty_annual']),f)
            events=B.financial_events(f,p,_STATE['market'].Date)
            history=B.make_series(code,p,_STATE['market'],_STATE['rates'],states,events,include_market_comparison=False)
        sampled=sample_history(wanted,history)
        rows.append(sampled)
        records.append({'Code':code,'rows':len(wanted),'price_history':not p.empty,
                        'eva_gap_rows':int(sampled.valuation_gap.notna().sum()),
                        'rim_gap_rows':int(sampled.rim_valuation_gap.notna().sum()),
                        'common_gap_rows':int((sampled.valuation_gap.notna()&sampled.rim_valuation_gap.notna()).sum())})
    frame=pd.concat(rows,ignore_index=True)
    for field in FIELDS:
        frame[field]=pd.to_numeric(frame[field],errors='coerce').astype(float)
    tmp=path.with_suffix('.tmp')
    frame.to_parquet(tmp,index=False,compression='zstd',compression_level=9)
    tmp.replace(path)
    B.json_write(tag,{'fingerprint':fingerprint,'sha256':B.digest(path),'coverage':records})
    return records


def sample_history(keys, history):
    """Keep original keys, including all-missing dates and delisted securities."""
    if history.empty:
        out=keys.copy()
        for field in FIELDS:
            out[field]=np.nan
        return out
    controls=G.market_control_features(history)
    history=pd.concat([history,controls],axis=1)
    values=history.reindex(columns=['Code','Date',*FIELDS])
    return keys.merge(values,on=['Code','Date'],how='left',validate='one_to_one')


def run(args):
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    keys=pd.read_parquet(args.dataset,columns=['Code','Date'])
    keys['Code']=keys.Code.map(B.code5)
    keys['Date']=pd.to_datetime(keys.Date)
    keys=keys.drop_duplicates().sort_values(['Code','Date']).reset_index(drop=True)
    codes=sorted(keys.Code.unique())
    sources=sorted(args.data_dir.glob('*.parquet'))+[args.data_dir/'jgb_yields.csv',args.dataset]
    modules=[Path(__file__),HERE/'build_roic_wacc_series.py',HERE/'fundamental_value.py',
             HERE/'fundamental_series.py',HERE/'roic_wacc_valuation.py',HERE/'valuation_gap.py',
             HERE/'capital_efficiency.py',HERE/'theoretical_value.py']
    source={'data':{p.name:B.digest(p) for p in sources},'modules':{p.name:B.digest(p) for p in modules},
            'rows':len(keys),'codes':len(codes),'fields':FIELDS,'primary_model':'eva_total_fade5',
            'cash_assumption':'no segregation; no separate cash deduction/add-back; approximate',
            'historical_revisions_verified':False,'chunk_codes':args.chunk_codes}
    fingerprint=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    manifest=args.output/'inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=source:
        raise ValueError('Inputs/code changed: use a fresh research output directory')
    B.json_write(manifest,source)
    print(f'Preparing {len(codes)} historical securities / {len(keys)} original security-date keys',flush=True)
    all_records=[]
    batches=[(offset,codes[offset:offset+args.chunk_codes],args.output,fingerprint,args.resume,args.chunk_codes)
             for offset in range(0,len(codes),args.chunk_codes)]
    if args.workers==1:
        initialize_worker(args.data_dir,args.dataset)
        for job in batches:
            all_records.extend(build_chunk(*job))
            print(f'Completed {len(all_records)}/{len(codes)} securities; {time.monotonic()-start:.1f}s',flush=True)
    else:
        # Spawn avoids inheriting Arrow/OpenMP thread pools. One process owns
        # each disjoint chunk; only this coordinator writes global manifests.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker,initargs=(args.data_dir,args.dataset)) as pool:
            pending=[pool.submit(build_chunk,*job) for job in batches]
            try:
                for result in as_completed(pending):
                    all_records.extend(result.result())
                    print(f'Completed {len(all_records)}/{len(codes)} securities; {time.monotonic()-start:.1f}s',flush=True)
            except BaseException:
                for result in pending: result.cancel()
                raise
    pd.DataFrame(all_records).sort_values('Code').to_csv(args.output/'security_coverage.csv',index=False)
    all_features=pd.concat([pd.read_parquet(p) for p in sorted(args.output.glob('features-*.parquet'))],ignore_index=True)
    if all_features.duplicated(['Code','Date']).any() or len(all_features)!=len(keys):
        raise ValueError('Incomplete or duplicate historical feature keys')
    pd.testing.assert_frame_equal(all_features[['Code','Date']].sort_values(['Code','Date']).reset_index(drop=True),keys)
    result={'status':'completed','rows':len(keys),'codes':len(codes),'all_original_keys_attempted':True,
            'missing_price_history_codes':sum(not r['price_history'] for r in all_records),
            'eva_gap_rows':int(all_features.valuation_gap.notna().sum()),
            'rim_gap_rows':int(all_features.rim_valuation_gap.notna().sum()),
            'common_gap_rows':int((all_features.valuation_gap.notna()&all_features.rim_valuation_gap.notna()).sum()),
            'fingerprint':fingerprint,'elapsed_seconds':round(time.monotonic()-start,1),
            'automatic_adoption':False,'historical_versions_verified':False}
    B.json_write(args.output/'completed.json',result)
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','dataset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chunk-codes',type=int,default=100)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=1,help='1 is the sequential fallback; bounded independent code chunks')
    args=p.parse_args()
    if args.chunk_codes<1 or not 1<=args.workers<=3 or args.output.resolve().is_relative_to((HERE/'model').resolve()) or args.output.resolve().is_relative_to((HERE.parent/'public').resolve()):
        p.error('Use positive chunk size and an independent private output directory')
    run(args)


if __name__=='__main__':
    main()
