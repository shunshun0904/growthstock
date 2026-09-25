#!/usr/bin/env python3
"""
LightGBM の木の本数を 200 -> 500 にしたことのテスト（2026-09-25、運用者の指示
「どうせならn_estimatorの数を500に変更して下さい。特徴量も増えてきたので」）。

- 本番の探索と学習は500本（tuning.SEARCH_N_ESTIMATORS）
- 学習率の探索範囲は本数に合わせて動かす（学習率 × 本数 = 歩幅の合計を 2〜40 に保つ）
- 本番の学習は、探索したときの本数が今の本数と違えば止まる
- 画面に並べる追加モデル（xgb / cat）は200本のまま（週次のジョブの時間の上限）。
  本数を変えたときに前の本数の探索を使い回さない作りも入れた
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import tuning  # noqa: E402


def frame(n_dates=30, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2018-01-31", periods=n_dates, freq="ME"):
        x1 = rng.normal(0, 1, n)
        p = 1 / (1 + np.exp(-(1.5 * x1 - 1.0)))
        rows.append(pd.DataFrame({"Date": d, "x1": x1, "x2": rng.normal(0, 1, n),
                                  "label": (rng.random(n) < p).astype(int)}))
    return pd.concat(rows, ignore_index=True)


class TestSetting(unittest.TestCase):
    def test_production_search_uses_500_trees(self):
        self.assertEqual(tuning.SEARCH_N_ESTIMATORS, 500)
        self.assertEqual(tuning.FIXED["n_estimators"], 500)
        self.assertEqual(tuning.DEFAULT_PARAMS["n_estimators"], 500)

    def test_learning_rate_range_follows_the_tree_count(self):
        """200本のときの範囲（0.01〜0.2）は変わらず、500本では 0.004〜0.08。"""
        self.assertEqual(tuning.lr_range(200), (0.01, 0.2))
        lo, hi = tuning.lr_range(500)
        self.assertAlmostEqual(lo, 0.004)
        self.assertAlmostEqual(hi, 0.08)
        for n in (100, 200, 500, 1000):
            lo, hi = tuning.lr_range(n)
            self.assertAlmostEqual(lo * n, tuning.LR_TOTAL[0])
            self.assertAlmostEqual(hi * n, tuning.LR_TOTAL[1])

    def test_past_choices_would_fall_below_the_old_lower_bound(self):
        """
        範囲を動かす理由。200本の探索で選ばれてきた学習率（0.016〜0.024）と同じ歩幅の
        合計を500本で出す学習率は、以前の下限 0.01 より下にある。新しい範囲には入る。
        """
        lo, hi = tuning.lr_range(500)
        for lr200 in (0.0158, 0.0184, 0.0222, 0.0232, 0.0234, 0.0242):
            lr500 = lr200 * 200 / 500
            self.assertLess(lr500, 0.01)
            self.assertTrue(lo <= lr500 <= hi)

    def test_default_learning_rate_keeps_the_total_step(self):
        """既定値（探索しなかったとき）も、200本で 0.05 だった歩幅の合計 10 を保つ。"""
        d = tuning.DEFAULT_PARAMS
        self.assertAlmostEqual(d["learning_rate"] * d["n_estimators"], 10.0)


class TestTuneHonoursTreesAndRange(unittest.TestCase):
    def test_default_is_the_production_setting(self):
        best = tuning.tune(frame(), ["x1", "x2"], n_trials=3, verbose=False, n_splits=3)
        self.assertEqual(best["n_estimators"], 500)
        self.assertEqual(tuning.LAST_CV["n_estimators"], 500)
        lo, hi = tuning.lr_range(500)
        self.assertEqual(tuning.LAST_CV["lr_range"], [lo, hi])
        self.assertTrue(lo <= best["learning_rate"] <= hi)

    def test_trees_can_be_set_for_experiments(self):
        """実験49 の腕 A（200本・0.01〜0.2）は、これまでの週次実行と同じ探索になる。"""
        best = tuning.tune(frame(), ["x1", "x2"], n_trials=3, verbose=False, n_splits=3,
                           n_estimators=200)
        self.assertEqual(best["n_estimators"], 200)
        self.assertEqual(tuning.LAST_CV["lr_range"], [0.01, 0.2])
        self.assertTrue(0.01 <= best["learning_rate"] <= 0.2)

    def test_range_can_be_set_for_experiments(self):
        """実験49 の腕 C（500本・0.01〜0.2）。"""
        best = tuning.tune(frame(), ["x1", "x2"], n_trials=3, verbose=False, n_splits=3,
                           n_estimators=500, lr_bounds=(0.01, 0.2))
        self.assertEqual(best["n_estimators"], 500)
        self.assertEqual(tuning.LAST_CV["lr_range"], [0.01, 0.2])
        self.assertTrue(0.01 <= best["learning_rate"] <= 0.2)

    def test_every_trial_uses_the_fixed_tree_count(self):
        """試行ごとに本数が変わらない（本数は目的関数の外で決まる）。"""
        seen = []
        orig = tuning._fit_one

        def spy(params, *a, **kw):
            seen.append(params["n_estimators"])
            return orig(params, *a, **kw)
        tuning._fit_one = spy
        try:
            tuning.tune(frame(), ["x1", "x2"], n_trials=2, verbose=False, n_splits=3,
                        n_estimators=123)
        finally:
            tuning._fit_one = orig
        self.assertEqual(len(seen), 2 * 3)             # 2試行 × 3分割
        self.assertEqual(set(seen), {123})


class TestProductionChecksTheTreeCount(unittest.TestCase):
    """本番の学習（train_production.py）は、探索したときの本数が違えば止まる。"""

    def _run(self, store, expect):
        import train_production as TP
        orig = tuning.load_params
        tuning.load_params = lambda path=None: store
        try:
            with self.assertRaises(expect) as cm:
                TP.main(["--dataset", "/nonexistent/dataset.parquet"])
        finally:
            tuning.load_params = orig
        return str(cm.exception)

    def _rec(self, trees):
        import features as F
        cols = F.columns(F.DEFAULT_PRESET)
        return {F.DEFAULT_PRESET: {"_n_features": len(cols),
                                   "_features_sig": F.signature(cols),
                                   "n_estimators": trees, "learning_rate": 0.02}}

    def test_a_200_tree_search_is_not_used(self):
        msg = self._run(self._rec(200), SystemExit)
        self.assertIn("木200本で探索したもの", msg)
        self.assertIn("500本", msg)

    def test_a_record_without_the_tree_count_is_not_used(self):
        rec = self._rec(500)
        for v in rec.values():
            v.pop("n_estimators")
        self.assertIn("探索したもの", self._run(rec, SystemExit))

    def test_the_current_tree_count_goes_on_to_training(self):
        """本数が合えばデータセットを読みに行く（ここでは無いので落ちる）。"""
        self._run(self._rec(tuning.SEARCH_N_ESTIMATORS), (FileNotFoundError, OSError))


class TestExtraModelsStayAt200(unittest.TestCase):
    """追加モデル（xgb / cat）は200本のまま。本数を変えたら前の探索を使い回さない。"""

    def test_extra_models_keep_200_trees(self):
        """
        週次のジョブ（上限330分）で本番の探索のあとに続けて回る。500本にすると
        見込みで上限に収まらない（docs/MODEL_ADOPTION_RULES.md §12）。
        本番の値を参照しない（本番を変えても一緒に動かない）。
        """
        import tuning_multi as TM
        self.assertEqual(TM.N_ESTIMATORS, 200)
        m = TM.build("xgb", {}, np.array([0, 1, 0, 0]))
        self.assertEqual(m.get_params()["n_estimators"], 200)
        c = TM.build("cat", {}, np.array([0, 1, 0, 0]))
        self.assertEqual(c.get_params()["iterations"], 200)

    def test_retune_when_the_tree_count_changed(self):
        import e15_tune_all as E15
        prev = {"train_to": "2025-07-23", "features_sig": "aaaaaaaaaaaa", "n_estimators": 200}
        self.assertIsNone(E15.why_retune(prev, "2025-07-23", "aaaaaaaaaaaa", 200))
        self.assertIn("本数", E15.why_retune(prev, "2025-07-23", "aaaaaaaaaaaa", 500))
        # 木の無いモデル（logit / mlp）は本数を見ない
        self.assertIsNone(E15.why_retune(prev, "2025-07-23", "aaaaaaaaaaaa", None))

    def test_study_name_has_the_tree_count_for_tree_models(self):
        """同じ週に本数を変えて探索し直しても、前の本数で測った試行を引き継がない。"""
        import features as F
        import tuning_multi as TM
        cols = F.columns("all")
        xgb = TM.study_name("xgb", 5, "2025-07-23", cols)
        self.assertTrue(xgb.endswith(f"_t{TM.N_ESTIMATORS}"))
        orig = TM.N_ESTIMATORS
        TM.N_ESTIMATORS = 500
        try:
            self.assertNotEqual(TM.study_name("xgb", 5, "2025-07-23", cols), xgb)
        finally:
            TM.N_ESTIMATORS = orig
        self.assertNotIn("_t", TM.study_name("logit", 5, "2025-07-23", cols)[-6:])

    def test_train_multi_skips_a_search_with_another_tree_count(self):
        import features as F
        import train_multi as TMlt
        cols = F.columns(F.DEFAULT_PRESET)
        cv = {"features_sig": F.signature(cols), "n_features": len(cols)}
        store = {"xgb": {"_cv": {**cv, "n_estimators": 500}},
                 "cat": {"_cv": {**cv, "n_estimators": 200}},
                 "logit": {"_cv": {**cv, "n_estimators": 500}}}
        got = TMlt.untuned(["xgb", "cat", "logit"], store, cols)
        self.assertIn("xgb", got)
        self.assertIn("本数", got["xgb"])
        self.assertNotIn("cat", got)
        self.assertNotIn("logit", got)          # 木の無いモデルは本数を見ない


class TestExperiment49(unittest.TestCase):
    def test_arms(self):
        import e49_trees as E
        self.assertEqual(E.ARMS["A"], (200, (0.01, 0.2)))
        self.assertEqual(E.ARMS["B"], (500, tuning.lr_range(500)))
        self.assertEqual(E.ARMS["C"], (500, (0.01, 0.2)))
        # 日曜からの週次実行と同じ作りが B
        self.assertEqual(E.ARMS["B"][0], tuning.SEARCH_N_ESTIMATORS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
