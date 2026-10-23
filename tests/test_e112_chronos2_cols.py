#!/usr/bin/env python3
"""
実験112（research/exp/e112_chronos2_cols.py）の部品。データにも Chronos-2 の重みにも触らない（手で解ける小さな例で確かめる）。

- パラメータ: 本番のモデルの名前（1.2σ は学習器の名前、1.5σ は <学習器>_k15）と記録の置き場、LightGBM の固定の値の補い
  （探索の値が勝つ・2回かけても同じ・本番の組み方 tuning_multi.build と同じ学習器になり、同じ点数を出す）、本番の記録から読む
- 3列: 20日目の中央値・80% 区間の幅・必要上昇率を超える確率（分位点の交差は並べ直す）、水準が無ければ止める
- 分位点の保存: 保存済みの束は Chronos-2 を読まずに使う（束の名前は行・入力の値・版で決まる）
- 規則の行（1.5σ の回の参考）: 前向きの百分位はそれより前の窓の分布で付け、最初の窓は欠測。3本とも線以上の行だけ残る

  python3 tests/test_e112_chronos2_cols.py
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e109_chronos2 as E109  # noqa: E402
import e112_chronos2_cols as E  # noqa: E402
import models as M  # noqa: E402
import tuning_multi as TM  # noqa: E402

LV = [0.1, 0.5, 0.9]


class Params(unittest.TestCase):
    def test_model_name_and_meta_path(self):
        self.assertEqual(E.model_name("lgbm", 1.2), "lgbm")
        self.assertEqual(E.model_name("cat", 1.5), "cat_k15")
        with self.assertRaises(ValueError):
            E.model_name("xgb", 1.8)
        self.assertEqual(E.meta_path("lgbm", "/r"), os.path.join("/r", "meta.json"))
        self.assertEqual(E.meta_path("xgb_k15", "/r"), os.path.join("/r", "models", "xgb_k15", "meta.json"))

    def test_full_lgbm_fills_and_keeps_tuned(self):
        p = {"learning_rate": 0.05, "num_leaves": 31, "subsample_freq": 3}
        q = E.full_lgbm(p)
        self.assertEqual(q["n_estimators"], TM.N_ESTIMATORS)
        self.assertEqual(q["objective"], "binary")
        self.assertEqual(q["subsample_freq"], 3)            # 探索の値が勝つ
        self.assertEqual(q["learning_rate"], 0.05)
        self.assertEqual(E.full_lgbm(q), q)                 # 2回かけても同じ
        self.assertNotIn("n_estimators", p)                  # 元の dict は変えない

    def test_full_lgbm_matches_production_build(self):
        """本番の組み方（models.fit → tuning_multi.build）と、実験の組み方（E41.oof_folds の LightGBM）が同じ点数を出す。"""
        import lightgbm as lgb
        import tuning

        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 6))
        y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0.6).astype(int)
        p = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10, "subsample": 0.8,
             "colsample_bytree": 0.9, "reg_alpha": 0.1, "reg_lambda": 0.2}
        old = TM.SEED
        try:
            TM.SEED = 7
            want = M.predict(M.fit("lgbm_k15", X, y, [f"c{i}" for i in range(6)], params=p), X)
        finally:
            TM.SEED = old
        m = lgb.LGBMClassifier(**{**E.full_lgbm(p), "random_state": 7}, scale_pos_weight=tuning.scale_pos_weight(y))
        m.fit(X, y)
        np.testing.assert_array_equal(m.predict_proba(X)[:, 1], want)

    def test_prod_params_from_repository_records(self):
        import json

        for a, k in (("lgbm", 1.2), ("xgb", 1.2), ("cat", 1.5), ("lgbm", 1.5)):
            name = E.model_name(a, k)
            with open(E.meta_path(name), encoding="utf-8") as fh:
                meta = json.load(fh)
            p, info = E.prod_params(a, k)
            self.assertEqual(info["model"], name)
            self.assertEqual(info["preset"], meta["preset"])
            self.assertEqual(info["hash"], E.params_hash(p))
            if a == "lgbm":
                self.assertEqual(p, E.full_lgbm(meta["params"]))
                self.assertEqual(p["n_estimators"], TM.N_ESTIMATORS)
            else:
                self.assertEqual(p, meta["params"])


class Columns(unittest.TestCase):
    def test_level_index(self):
        lv = [0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]
        self.assertEqual(len(lv), 21)
        self.assertEqual(E.level_index(lv, 0.1), 2)
        self.assertEqual(E.level_index(lv, 0.9), 18)
        with self.assertRaises(SystemExit):
            E.level_index(LV, 0.25)

    def test_three_columns_by_hand(self):
        # 2行・水準3つ・先2行。20日目（最後の先）の分位点で数える。2行目は交差している（並べ直して読む）
        q = np.array([[[0.0, 1.0], [0.0, 1.2], [0.0, 1.4]],
                      [[0.0, 2.4], [0.0, 2.0], [0.0, 2.2]]])
        last = np.array([1.0, 2.0])
        need = np.array([np.exp(0.1) - 1, np.exp(0.3) - 1])     # log(1 + need) = 0.1 / 0.3
        c = E.c2_columns(q, LV, last, need)
        np.testing.assert_allclose(c["c2_med"], [0.2, 0.2])
        np.testing.assert_allclose(c["c2_width"], [0.4, 0.4])
        # 1行目: しきい値 1.1 は 1.0〜1.2 の真ん中 → 1 − 0.3 = 0.7。2行目: 並べ直すと 2.0 / 2.2 / 2.4、しきい値 2.3 → 1 − 0.7
        np.testing.assert_allclose(c["c2_p_end"], [0.7, 0.3])
        np.testing.assert_allclose(c["c2_p_end"], E109.scores(q, LV, last, need)["p_end"])


class SavedQuantiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = (E.OOF_DIR, E.TAG)
        E.OOF_DIR, E.TAG = self.tmp, "t112"

    def tearDown(self):
        E.OOF_DIR, E.TAG = self.old
        shutil.rmtree(self.tmp)

    def test_reads_saved_chunks_without_the_model(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        panel = pd.DataFrame({"Code": ["11110"] * 6 + ["22220"] * 6, "Date": list(dates) * 2,
                              "close": np.arange(1.0, 13.0)})
        rows = pd.DataFrame({"Code": ["11110", "22220"], "Date": [dates[5], dates[3]]})
        start, end = E109.context_bounds(panel, rows, E.CONTEXT)
        items = E109.make_items(panel["close"].to_numpy(dtype=float), start, end)
        meta = {"arm": "Z0", "ctx": E.CONTEXT, "h": E.HORIZON, "model": E.MODEL_ID, "rev": E.REVISION}
        key = E109.chunk_key(meta, rows[["Code", "Date"]], items)
        q = np.arange(2 * 3 * E.HORIZON, dtype=np.float32).reshape(2, 3, E.HORIZON)
        np.savez_compressed(E.cpath(f"z0_{key}.npz"), q=q, levels=np.asarray(LV))
        got, lv, info = E.z0_quantiles(rows, panel)                 # 保存済みなので Chronos-2 は読まない
        np.testing.assert_array_equal(got, q)
        self.assertEqual(lv, LV)
        self.assertEqual((info["chunks"], info["computed"]), (1, 0))
        # 文脈は t の行で終わり、銘柄をまたがない
        self.assertEqual(list(end - start + 1), [6, 4])
        np.testing.assert_allclose(items[1], np.log(np.arange(7.0, 11.0)).astype(np.float32))


class RuleRows(unittest.TestCase):
    def test_forward_percentile(self):
        o = pd.DataFrame({"fold": [1, 1, 2, 2, 3, 3, 3], "score": [1.0, 2.0, 3.0, 4.0, 0.0, 2.5, 5.0]})
        p = E.forward_pct(o, first=2)
        self.assertTrue(np.isnan(p[:2]).all())                      # 最初の窓は前の分布が無い
        np.testing.assert_allclose(p[2:4], [100.0, 100.0])           # 窓2 は窓1（1・2）の分布で
        np.testing.assert_allclose(p[4:], [0.0, 50.0, 100.0])        # 窓3 は窓1・2（1〜4）の分布で
        self.assertTrue(np.isnan(E.forward_pct(o, first=3)[:4]).all())

    def test_all_three_at_or_above_the_line(self):
        def oof(scores):
            return pd.DataFrame({"Code": ["a", "b", "c", "d", "e", "f"], "Date": pd.to_datetime(["2024-01-05"] * 6),
                                 "fold": [1, 1, 1, 2, 2, 2], "label": [0, 1, 0, 1, 0, 1],
                                 "hit10": [0, 1, 0, 1, 0, 1], "tp10c": [-1.0, 10.0, 0.5, 10.0, -3.0, 10.0],
                                 "score": scores})
        oofs = {"lgbm": oof([0.1, 0.2, 0.3, 0.9, 0.8, 0.25]),
                "xgb": oof([0.1, 0.2, 0.3, 0.9, 0.1, 0.9]),
                "cat": oof([0.1, 0.2, 0.3, 0.95, 0.9, 0.9])}
        r = E.rule_rows(oofs, line=95.0, first=2)
        # 窓2 の3行のうち、3本とも窓1 の全部（0.1〜0.3）より上なのは d だけ（e は xgb が、f は lgbm が下）
        self.assertEqual(list(r["Code"]), ["d"])
        st = E.rule_stats(r)
        self.assertEqual((st["n"], st["days"]), (1, 1))
        self.assertAlmostEqual(st["tp10c"], 10.0)


if __name__ == "__main__":
    unittest.main()
