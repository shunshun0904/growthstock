#!/usr/bin/env python3
"""
木の本数のテスト（2026-09-25）。

運用者の指示「どうせならn_estimatorの数を500に変更して下さい。特徴量も増えてきたので」で
本番の LightGBM を一度500本にしたが、追加モデル（xgb / cat）まで500本にすると週次のジョブ
（上限330分）に収まらない見込みと分かり、運用者の判断で全モデル200本のままにした
（「であれば、lgbmも200のままでよいです」。docs/MODEL_ADOPTION_RULES.md §12）。

- 本番の探索と学習は200本（tuning.SEARCH_N_ESTIMATORS）。追加モデルも同じ値
- 学習率の探索範囲は本数に合わせて動く（学習率 × 本数 = 歩幅の合計を 2〜40 に保つ）。
  200本では従来どおり 0.01〜0.2
- 本番の学習は、探索したときの本数が今の本数と違えば止まる
- 追加モデルも、本数を変えたときに前の本数の探索を使い回さない
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
    def test_all_models_use_200_trees(self):
        """500本は週次のジョブに収まらないので全モデル200本（運用者の判断）。"""
        import tuning_multi as TM
        self.assertEqual(tuning.SEARCH_N_ESTIMATORS, 200)
        self.assertEqual(tuning.FIXED["n_estimators"], 200)
        self.assertEqual(tuning.DEFAULT_PARAMS["n_estimators"], 200)
        self.assertEqual(TM.N_ESTIMATORS, tuning.SEARCH_N_ESTIMATORS)

    def test_search_space_is_unchanged_at_200_trees(self):
        """200本では学習率の範囲も既定値も以前と同じ（探索は以前と同じ）。"""
        self.assertEqual(tuning.lr_range(200), (0.01, 0.2))
        self.assertEqual(tuning.lr_range(tuning.SEARCH_N_ESTIMATORS), (0.01, 0.2))
        self.assertEqual(tuning.DEFAULT_PARAMS["learning_rate"], 0.05)

    def test_learning_rate_range_follows_the_tree_count(self):
        lo, hi = tuning.lr_range(500)
        self.assertAlmostEqual(lo, 0.004)
        self.assertAlmostEqual(hi, 0.08)
        for n in (100, 200, 500, 1000):
            lo, hi = tuning.lr_range(n)
            self.assertAlmostEqual(lo * n, tuning.LR_TOTAL[0])
            self.assertAlmostEqual(hi * n, tuning.LR_TOTAL[1])

    def test_past_choices_would_fall_below_the_old_lower_bound_at_500(self):
        """
        範囲を本数に合わせる理由。200本の探索で選ばれてきた学習率（0.016〜0.024）と同じ
        歩幅の合計を500本で出す学習率は、0.01 より下にある。本数に合わせた範囲には入る。
        """
        lo, hi = tuning.lr_range(500)
        for lr200 in (0.0158, 0.0184, 0.0222, 0.0232, 0.0234, 0.0242):
            lr500 = lr200 * 200 / 500
            self.assertLess(lr500, 0.01)
            self.assertTrue(lo <= lr500 <= hi)

    def test_default_learning_rate_keeps_the_total_step(self):
        d = tuning.DEFAULT_PARAMS
        self.assertAlmostEqual(d["learning_rate"] * d["n_estimators"], 10.0)


class TestTuneHonoursTreesAndRange(unittest.TestCase):
    def test_default_is_the_production_setting(self):
        best = tuning.tune(frame(), ["x1", "x2"], n_trials=3, verbose=False, n_splits=3)
        self.assertEqual(best["n_estimators"], 200)
        self.assertEqual(tuning.LAST_CV["n_estimators"], 200)
        self.assertEqual(tuning.LAST_CV["lr_range"], [0.01, 0.2])
        self.assertTrue(0.01 <= best["learning_rate"] <= 0.2)

    def test_trees_can_be_set_for_experiments(self):
        """実験49 の腕 B（500本・範囲を本数に合わせる）。"""
        best = tuning.tune(frame(), ["x1", "x2"], n_trials=3, verbose=False, n_splits=3,
                           n_estimators=500)
        self.assertEqual(best["n_estimators"], 500)
        lo, hi = tuning.lr_range(500)
        self.assertEqual(tuning.LAST_CV["lr_range"], [lo, hi])
        self.assertTrue(lo <= best["learning_rate"] <= hi)

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

    def test_a_search_with_another_tree_count_is_not_used(self):
        """例: 500本で探索した記録（fcbe4c1 の頃の週次実行なら書かれえた）で200本の本番を作らない。"""
        msg = self._run(self._rec(500), SystemExit)
        self.assertIn("木500本で探索したもの", msg)
        self.assertIn("200本", msg)

    def test_a_record_without_the_tree_count_is_not_used(self):
        rec = self._rec(200)
        for v in rec.values():
            v.pop("n_estimators")
        self.assertIn("探索したもの", self._run(rec, SystemExit))

    def test_the_current_tree_count_goes_on_to_training(self):
        """本数が合えばデータセットを読みに行く（ここでは無いので落ちる）。"""
        self._run(self._rec(tuning.SEARCH_N_ESTIMATORS), (FileNotFoundError, OSError))


class TestExtraModels(unittest.TestCase):
    """追加モデル（xgb / cat）は本番と同じ本数。本数を変えたら前の探索を使い回さない。"""

    def test_extra_models_use_the_production_tree_count(self):
        import tuning_multi as TM
        m = TM.build("xgb", {}, np.array([0, 1, 0, 0]))
        self.assertEqual(m.get_params()["n_estimators"], tuning.SEARCH_N_ESTIMATORS)
        c = TM.build("cat", {}, np.array([0, 1, 0, 0]))
        self.assertEqual(c.get_params()["iterations"], tuning.SEARCH_N_ESTIMATORS)

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
        self.assertFalse(TM.study_name("logit", 5, "2025-07-23", cols).endswith(
            f"_t{TM.N_ESTIMATORS}"))

    def test_train_multi_skips_a_search_with_another_tree_count(self):
        import features as F
        import train_multi as TMlt
        import tuning_multi as TM
        cols = F.columns(F.DEFAULT_PRESET)
        cv = {"features_sig": F.signature(cols), "n_features": len(cols)}
        store = {"xgb": {"_cv": {**cv, "n_estimators": 500}},
                 "cat": {"_cv": {**cv, "n_estimators": TM.N_ESTIMATORS}},
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
        # 週次実行（9/27 も）と同じ作りが A
        self.assertEqual(E.ARMS["A"], (tuning.SEARCH_N_ESTIMATORS,
                                       tuning.lr_range(tuning.SEARCH_N_ESTIMATORS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
