"""Publish current training metadata independently of frozen historical predictions."""
import json
import label_specs as LS
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALGOS=['lgbm','xgb','cat','lgbm_k15','xgb_k15','cat_k15']

def export(root=ROOT):
    rows=[]
    for a in ALGOS:
        p=root/'research/model'/('meta.json' if a=='lgbm' else f'models/{a}/meta.json')
        m=json.loads(p.read_text())
        rows.append({'algo':a,**{k:m.get(k) for k in ['labelSpec','trainedAt','trainTo','dataThrough','nTrain','nOof','dataSnapshot']}})
    fixed=[m['labelSpec'] is not None for m in rows]
    if any(fixed) and not all(fixed):raise ValueError('Active production contains mixed label definitions')
    if all(fixed):
        for model in rows:LS.require(model, model['algo'])
    out={'version':1,'kind':'fixed_return' if all(fixed) else 'legacy_sigma','models':rows}
    p=root/'public/data/active_models.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out
if __name__=='__main__':export()
