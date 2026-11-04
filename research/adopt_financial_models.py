"""Assemble disjoint training outputs; verify every model before publishing any."""
from pathlib import Path
import json
import shutil
import sys
import features as F
import financial_inputs as FI
import model_artifacts as A
import label_specs as LS
import ltr5


def assemble(incoming):
    incoming=Path(incoming)
    for scope in ['base','strict','ranking']:
        source=incoming/('financial-inputs-'+scope)/'research'
        shutil.copytree(source/'model',Path(__file__).parent/'model',dirs_exist_ok=True)
        for path in source.glob('*params.json'):
            if path.name != 'strict_params.json': shutil.copyfile(path,Path(__file__).parent/path.name)
    strict=json.loads((incoming/'financial-inputs-strict/research/strict_params.json').read_text())
    path=Path(__file__).parent/'multi_params.json'
    merged=json.loads(path.read_text())
    for name in LS.STRICT_MODELS: merged[name]=strict[name]
    path.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n')
    verify(Path(__file__).parent / "model")
    shutil.copyfile(incoming/'financial-inputs-mart/public/data/valuations.json',Path(__file__).parents[1]/'public/data/valuations.json')
    print('Verified eight models; consistent 373/377-column schemas and classifier data snapshot')

def verify(root):
    expected=F.columns(F.DEFAULT_PRESET)
    if len(expected)!=373 or expected[-13:]!=FI.COLUMNS:
        raise ValueError('Production feature contract changed during adoption')
    snapshots=[]
    for name in [*LS.BASE_MODELS,*LS.STRICT_MODELS,'ltr','ltr5']:
        d=A.directory(name,root);meta=json.loads((d/'meta.json').read_text())
        if meta['features']!=expected+(list(ltr5.MOVE_COLS) if name=='ltr5' else []):
            raise ValueError('Unexpected model features: '+name)
        if name in LS.BASE_MODELS+LS.STRICT_MODELS:
            A.require(meta,name,root)
            if not meta.get('dataSnapshot'): raise ValueError('Missing data snapshot: '+name)
            snapshots.append(meta['dataSnapshot'])
        else:
            for local, digest in A.hashes(name,root).items():
                if meta.get('artifactHashes',{}).get(local) != digest:
                    raise ValueError('Ranking artifact mismatch: '+name)
    if len(set(snapshots))!=1: raise ValueError('Training data snapshots differ across classifiers')


if __name__=='__main__': assemble(sys.argv[1])
