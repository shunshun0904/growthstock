"""Tune and train one fixed-return production model; emit only aggregate metadata."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import features as F
import label_specs as LS
import model_artifacts as A
import run_tuning, train_production, train_multi
from exp import e15_tune_all


def main():
    ap=argparse.ArgumentParser();ap.add_argument('algo',choices=LS.BASE_MODELS+LS.STRICT_MODELS)
    args=ap.parse_args();algo=args.algo
    if algo=='lgbm':
        run_tuning.main(['--features',F.DEFAULT_PRESET,'--n-trials','50','--cv','year_cap_date'])
        train_production.main([])
    else:
        old=sys.argv;sys.argv=['e15_tune_all.py',algo,'--force']
        try:e15_tune_all.main()
        finally:sys.argv=old
        if train_multi.main(['--models',algo])!=0:raise RuntimeError('training did not complete')
    data=Path('research/_data');dest=A.directory(algo)
    meta=json.loads((dest/'meta.json').read_text())
    meta.update(A.data_snapshot(data))
    meta['dataThrough']=str(max(pd.read_parquet(p,columns=['Date']).Date.max() for p in data.glob('bars_*.parquet')))[:10]
    (dest/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    A.require(meta,algo)
    print(json.dumps({k:meta[k] for k in ['trainedAt','trainTo','dataThrough','labelSpec','nTrain','nOof','dataSnapshot']},ensure_ascii=False))

if __name__=='__main__':main()
