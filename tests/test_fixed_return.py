import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import label_specs as L
import build_dataset as B

class FixedReturn(unittest.TestCase):
    def rows(self):
        return pd.DataFrame({"future_rise": [.10, .15, .149, .2, .2, np.nan],
            "end_level": [.05, .075, .075, .049, .10, .1],
            "uptrend_end": [1, 1, 1, 1, 0, 1], "entry_price": [100.] * 6,
            "vol_20d": [np.nan, 1, 99, 2, 2, 2]})
    def test_fixed10_and_fixed15(self):
        f = self.rows()
        self.assertEqual(L.labels(f).iloc[:5].tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(L.labels(f, "cat_k15").iloc[:5].tolist(), [0, 1, 0, 0, 0])
        self.assertTrue(pd.isna(L.labels(f).iloc[5]))
    def test_fixed_threshold_does_not_require_sigma(self):
        self.assertEqual(L.labels(self.rows()).iloc[0], 1)
    def test_price_rounding_boundary(self):
        f = self.rows().iloc[[1]].copy()
        f["future_rise"] = 1.15 - 1
        f["end_level"] = 1.075 - 1
        self.assertEqual(L.labels(f, "lgbm_k15").iloc[0], 1)
        f["future_rise"] = .15 - 1e-8
        self.assertEqual(L.labels(f, "lgbm_k15").iloc[0], 0)
    def test_old_sigma_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "label definition mismatch"):
            L.require({"volNormK": 1.2})
        with self.assertRaises(ValueError):
            L.require({"labelSpec": L.for_model()}, "xgb_k15")
        L.require({"labelSpec": L.for_model("xgb_k15")}, "xgb_k15")
    def test_old_training_labels_rejected(self):
        f = self.rows();f["label"] = L.labels(f)
        L.check_training_frame(f)
        f.loc[0, "label"] = 0
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            L.check_training_frame(f)
    def test_production_default_is_fixed10(self):
        self.assertFalse(B.DEFAULT_RISE.normalised)
        self.assertEqual(B.DEFAULT_RISE.threshold, .10)
        self.assertEqual(B.DEFAULT_RISE.end_ratio, .05)
        self.assertEqual(B.DEFAULT_RISE.horizon, 20)


class ArtifactIsolationTest(unittest.TestCase):
    def test_weight_and_oof_tampering_are_rejected(self):
        import tempfile,json
        from pathlib import Path
        import model_artifacts as A
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            (d/'model.txt').write_bytes(b'fixed trained weights')
            (d/'oof.parquet').write_bytes(b'fixed out of fold')
            meta={'labelSpec':L.for_model(),'artifactHashes':A.hashes('lgbm',d)}
            A.require(meta,'lgbm',d)
            (d/'model.txt').write_bytes(b'old sigma weights')
            with self.assertRaisesRegex(ValueError,'training metadata'):A.require(meta,'lgbm',d)
    def test_release_names_cannot_overwrite_sigma_files(self):
        import model_artifacts as A
        names=[A.filename(a,k) for a in L.BASE_MODELS+L.STRICT_MODELS for k in ('model','oof')]
        self.assertEqual(len(set(names)),12)
        self.assertTrue(all(n.startswith('fixed_return_v1_') for n in names))
        self.assertNotIn('model.txt',names)
    def test_production_rejects_sigma_tuning(self):
        import features as F,train_multi as T,tuning_multi as TM
        cols=F.columns(F.DEFAULT_PRESET)
        cv={'features_sig':F.signature(cols),'n_features':len(cols),'n_estimators':TM.N_ESTIMATORS}
        self.assertIn('xgb',T.untuned(['xgb'],{'xgb':{'_cv':cv}},cols))
        cv['label_spec']=L.for_model('xgb')
        self.assertEqual(T.untuned(['xgb'],{'xgb':{'_cv':cv}},cols),{})
    def test_studies_separate_label_definitions(self):
        import tuning_multi as T
        args=('xgb',5,'2025-09-02',['a','b'])
        self.assertNotEqual(T.study_name(*args),T.study_name(*args,label_spec=L.for_model('xgb')))
        self.assertNotEqual(T.study_name(*args,label_spec=L.for_model('xgb')),T.study_name(*args,label_spec=L.for_model('xgb_k15')))

if __name__ == "__main__":
    unittest.main()
