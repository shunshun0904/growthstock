"""An eight-model migration cannot publish partial, mixed or mismatched outputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import adopt_financial_models as C
import features as F
import model_artifacts as A
import label_specs as LS
import ltr5

class Adoption(unittest.TestCase):
    def make_models(self, root):
        for name in [*LS.BASE_MODELS,*LS.STRICT_MODELS,'ltr','ltr5']:
            d=A.directory(name,root);d.mkdir(parents=True,exist_ok=True)
            for kind in ['model','oof']:(d/A.local_name(name,kind)).write_bytes((name+kind).encode())
            meta={'features':F.columns(F.DEFAULT_PRESET)+(list(ltr5.MOVE_COLS) if name=='ltr5' else []),
                  'artifactHashes':A.hashes(name,root),'dataSnapshot':'same-frozen-mart'}
            if name not in ['ltr','ltr5']:meta['labelSpec']=LS.for_model(name)
            (d/'meta.json').write_text(json.dumps(meta))

    def test_long_training_cannot_overwrite_newer_display_data(self):
        with tempfile.TemporaryDirectory() as t:
            source, target=Path(t)/'old.json', Path(t)/'live.json'
            source.write_text(json.dumps({'generatedAt':'2026-11-04T07:00:00Z','stocks':[]}))
            target.write_text(json.dumps({'generatedAt':'2026-11-04T08:00:00+00:00','stocks':['new']}))
            self.assertFalse(C.copy_valuation_if_newer(source,target))
            self.assertEqual(json.loads(target.read_text())['stocks'],['new'])
            target.unlink()
            self.assertTrue(C.copy_valuation_if_newer(source,target))

    def test_complete_migration_and_corrupt_ranker(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_models(root);C.verify(root)
            (A.directory('ltr5',root)/'model.joblib').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Ranking artifact mismatch'):C.verify(root)

    def test_mixed_training_snapshots_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_models(root)
            p=A.directory('cat_k15',root)/'meta.json';m=json.loads(p.read_text());m['dataSnapshot']='different';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'snapshots differ'):C.verify(root)

    def test_old_or_partial_migration_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make_models(root)
            p=A.directory('cat',root)/'meta.json';m=json.loads(p.read_text());m['features']=F.columns(F.LEGACY_PRESET);p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Unexpected model features'):C.verify(root)
            p.unlink()
            with self.assertRaises(FileNotFoundError):C.verify(root)
