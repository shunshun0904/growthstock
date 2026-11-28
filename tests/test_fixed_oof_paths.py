"""
固定 +10%・+15% の6本の out-of-fold の置き場所（research/model_artifacts.py の flat_oof_path・research/live_track.py の find_oof）。

2026-11-04 の予測で近傍の集計（research/knn_compare.py）が「lgbm: active OOF is missing or inconsistent」で止まった。日次の予測は
Release の fixed_return_v1_* を research/model/ に並べ直して置く（model_artifacts.install）のに、flat_oof_path は Release の形
（<data_dir>/fixed_return_v1_*oof.parquet）しか見ていなかったため。確かめること:
- 日次の予測が並べ直した形（<model_dir>/oof.parquet・<model_dir>/models/<algo>/oof.parquet）を、ハッシュが合えば使う
- Release の形も今までどおり使う（Backfill KNN・Run Experiment）。両方あれば Release の形
- どちらの形でもハッシュが合わなければ止める。旧σの名前の OOF（oof.parquet・<algo>_oof.parquet）は固定のモデルには使わない
- 近傍の集計の母集団（knn_compare.load_pool）が日次の予測と同じ置き方で作れる
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "research"))

import knn_compare as KC  # noqa: E402
import label_specs as LS  # noqa: E402
import live_track as LT  # noqa: E402
import model_artifacts as A  # noqa: E402


def oof_frame(seed: int, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-04", periods=n // 4)
    return pd.DataFrame({"Code": [f"{1000 + i:04d}0" for i in range(n)],
                         "Date": np.repeat(days, 4)[:n],
                         "score": rng.random(n)})


def install(root: Path, algo: str, frame: pd.DataFrame, meta_hash_of: Path = None) -> Path:
    """日次の予測と同じ形（research/model/…）に OOF と meta.json を置く。meta のハッシュは meta_hash_of（既定は置いた OOF）。"""
    d = A.directory(algo, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / A.local_name(algo, "oof")
    frame.to_parquet(path, index=False)
    digest = A.sha256(meta_hash_of or path)
    (d / "meta.json").write_text(json.dumps({"labelSpec": LS.for_model(algo),
                                             "artifactHashes": {"oof.parquet": digest}}))
    return path


class FlatOofPathTest(unittest.TestCase):
    def test_installed_layout_is_used(self):
        with tempfile.TemporaryDirectory() as t:
            root, data = Path(t) / "model", Path(t) / "data"
            data.mkdir()
            for i, algo in enumerate(("lgbm", "xgb", "cat_k15")):
                path = install(root, algo, oof_frame(i))
                self.assertEqual(A.flat_oof_path(algo, data, root), str(path))
                # 日次の予測の近傍の集計は oof_dir を指定しない（= model_dir）
                self.assertEqual(LT.find_oof(algo, str(root), str(root)), str(path))
            self.assertEqual(str(A.directory("lgbm", root) / "oof.parquet"), A.flat_oof_path("lgbm", data, root))

    def test_release_layout_still_used_and_comes_first(self):
        with tempfile.TemporaryDirectory() as t:
            root, data = Path(t) / "model", Path(t) / "data"
            data.mkdir()
            installed = install(root, "xgb", oof_frame(1))
            release = data / A.filename("xgb", "oof")
            release.write_bytes(installed.read_bytes())
            self.assertEqual(A.flat_oof_path("xgb", data, root), str(release))
            installed.unlink()
            self.assertEqual(A.flat_oof_path("xgb", data, root), str(release))

    def test_mismatched_hash_is_rejected_in_both_layouts(self):
        with tempfile.TemporaryDirectory() as t:
            root, data = Path(t) / "model", Path(t) / "data"
            data.mkdir()
            other = Path(t) / "other.parquet"
            oof_frame(9).to_parquet(other, index=False)
            installed = install(root, "lgbm", oof_frame(2), meta_hash_of=other)
            (data / A.filename("lgbm", "oof")).write_bytes(installed.read_bytes())
            with self.assertRaisesRegex(ValueError, "active OOF is missing or inconsistent"):
                A.flat_oof_path("lgbm", data, root)

    def test_legacy_sigma_names_are_never_used(self):
        with tempfile.TemporaryDirectory() as t:
            root, data = Path(t) / "model", Path(t) / "data"
            data.mkdir()
            installed = install(root, "cat", oof_frame(3))
            legacy = data / "cat_oof.parquet"
            legacy.write_bytes(installed.read_bytes())
            installed.unlink()
            with self.assertRaisesRegex(ValueError, "active OOF is missing or inconsistent"):
                LT.find_oof("cat", str(data), str(root))

    def test_legacy_models_keep_their_old_lookup(self):
        with tempfile.TemporaryDirectory() as t:
            root, data = Path(t) / "model", Path(t) / "data"
            data.mkdir()
            d = A.directory("xgb", root)
            d.mkdir(parents=True)
            (d / "meta.json").write_text(json.dumps({"labelSpec": None}))
            self.assertEqual(A.flat_oof_path("xgb", data, root), str(data / "xgb_oof.parquet"))


class KnnPoolTest(unittest.TestCase):
    def test_pool_builds_from_the_daily_prediction_layout(self):
        """2026-11-04 に止まった呼び方（knn_compare.load_pool(model_dir)。oof_dir は指定しない）で母集団が作れる。"""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "model"
            base = oof_frame(0, n=80)
            for i, algo in enumerate(KC.RULE_MODELS):
                f = base.copy()
                f["score"] = np.random.default_rng(10 + i).random(len(f))
                install(root, algo, f)
            pool = KC.load_pool(str(root), pool_line=0.0)
            self.assertEqual(len(pool), len(base))
            self.assertEqual(list(pool.columns), ["Code", "Date", "min15"])
            self.assertTrue(((pool["min15"] >= 0) & (pool["min15"] <= 100)).all())


if __name__ == "__main__":
    unittest.main()
