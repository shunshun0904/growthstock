"""Check the complete fixed-return production set before activation (no raw data needed)."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import label_specs as LS
import model_artifacts as A
import models as M


def verify(root=A.MODEL_ROOT):
    import lightgbm as lgb
    root=Path(root);A.verify_all(root)
    records=[];labels={};snapshots=set();dates=set();features=set()
    for algo in LS.BASE_MODELS+LS.STRICT_MODELS:
        d=A.directory(algo,root);meta=json.loads((d/'meta.json').read_text())
        LS.require(meta.get('tuning',{}),algo,field='label_spec')
        assert meta['tuning']['n_trials']==50, f'{algo}: incomplete tuning'
        assert meta['tuning']['n_splits']==5, f'{algo}: different CV'
        assert meta['tuning']['n_estimators']==200, f'{algo}: different tree budget'
        snapshots.add(meta['dataSnapshot']);dates.add((meta['dataThrough'],meta['trainTo']))
        features.add(tuple(meta['features']))
        oof=pd.read_parquet(d/'oof.parquet')
        assert len(oof)==meta['nOof'] and not oof.duplicated(['Date','Code']).any()
        assert oof['score'].between(0,1).all() and oof['label'].isin([0,1]).all()
        x=np.zeros((3,len(meta['features'])));x[1,0]=np.nan;x[2,:]=.1
        if algo=='lgbm':
            model=lgb.Booster(model_file=str(d/'model.txt'));p=model.predict(x)
            trees=model.current_iteration()
        else:
            model,_=M.load(algo,str(root));p=M.predict(model,x)
            trees=(model.booster_.current_iteration() if algo.startswith('lgbm')
                   else model.get_booster().num_boosted_rounds() if algo.startswith('xgb')
                   else model.tree_count_)
        assert trees==200, f'{algo}: incomplete fitted trees'
        assert np.asarray(p).shape==(3,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        labels[algo]=oof.set_index(['Date','Code'])['label'].astype('int8').sort_index()
        records.append({'algo':algo,'threshold':meta['labelSpec']['threshold'],
                        'trainTo':meta['trainTo'],'dataThrough':meta['dataThrough'],
                        'nTrain':meta['nTrain'],'nOof':len(oof),'trees':int(trees),'hashesVerified':True})
    assert len(snapshots)==1,'Models were trained on different raw snapshots'
    assert len(dates)==1 and len(features)==1,'Model training cohorts differ'
    for group in [LS.BASE_MODELS,LS.STRICT_MODELS]:
        for algo in group[1:]:pd.testing.assert_series_equal(labels[group[0]],labels[algo])
    base=labels['lgbm'];strict=labels['lgbm_k15']
    assert base.index.equals(strict.index) and (strict<=base).all(),'+15% is not a subset of +10%'
    return {'complete':True,'models':records,'dataSnapshot':next(iter(snapshots)),
            'features':len(next(iter(features))),'checks':['label contract','binary and OOF hashes',
            '50 trials x 5 folds','common fresh snapshot','synthetic missing-value prediction',
            'OOF keys and label consistency','nested fixed-return outcomes']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',default=str(A.MODEL_ROOT));args=ap.parse_args()
    print(json.dumps(verify(args.model_dir),ensure_ascii=False,indent=2))
