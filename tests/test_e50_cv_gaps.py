#!/usr/bin/env python3
"""
実験50（research/exp/e50_cv_gaps.py）の道具のテスト。

5分割 CV で3モデルの大外れ（正例なのに下位10%・負例なのに上位5%）を調べる。
- 分割は本番のパラメータ探索と同じ（year_cap_date・種0）。lgbm と xgb / cat で同じ分割になる
- 群は分割の中のスコアの順位で決める
- 比べるのは欠損率と値の順位（AUC）だけ。値そのものは出さない
- ラベルの3条件の組み直しは build_dataset と同じ式
- 銘柄ごとの一覧（--examples）は公開ログ（Actions）では出さない

  python3 tests/test_e50_cv_gaps.py
"""
import inspect
import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_dataset as B  # noqa: E402
import e50_cv_gaps as E  # noqa: E402
import features as F  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402


def frame(n_dates=120, per_day=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for i in range(per_day):
            rows.append({"Date": d, "Code": f"{1000 + i}",
                         "label": int(rng.random() < 0.2),
                         "cap_band": int(rng.integers(0, 4))})
    return pd.DataFrame(rows)


class TestSameFoldsAsProduction(unittest.TestCase):
    def test_seed_and_scheme_match_the_production_tunings(self):
        self.assertEqual(E.SEED, 0)
        self.assertEqual(TM.SEED, E.SEED)
        self.assertEqual(inspect.signature(tuning.tune).parameters["seed"].default, E.SEED)
        self.assertEqual(E.N_SPLITS, 5)

    def test_every_row_in_exactly_one_fold_and_dates_are_not_split(self):
        df = frame()
        fold = E.folds_of(df)
        self.assertEqual(set(np.unique(fold)), set(range(5)))
        per_date = pd.Series(fold).groupby(df["Date"]).nunique()
        self.assertTrue((per_date == 1).all())          # 同じ日は同じ分割

    def test_folds_equal_the_tuning_folds(self):
        df = frame(seed=3)
        fold = E.folds_of(df)
        folds = tuning.year_folds(df, n_splits=5, seed=0, by_year=True, by_cap=True,
                                  group_by_date=True)
        for k, (_, va) in enumerate(folds):
            self.assertTrue((fold[va.index.to_numpy()] == k).all())


class TestGroups(unittest.TestCase):
    def test_rank_is_within_fold(self):
        score = np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.4])
        fold = np.array([0, 0, 0, 1, 1, 1])
        r = E.within_fold_rank(score, fold)
        np.testing.assert_allclose(r, [1 / 3, 1.0, 2 / 3, 1 / 3, 2 / 3, 1.0])

    def test_boundaries(self):
        label = np.array([1, 1, 0, 0, 1, 0])
        rank = np.array([0.10, 0.11, 0.95, 0.94, 0.95, 0.10])
        g = E.groups_of(label, rank)
        np.testing.assert_array_equal(g["FN_ext"], [True, False, False, False, False, False])
        np.testing.assert_array_equal(g["FP_ext"], [False, False, True, False, False, False])
        np.testing.assert_array_equal(g["TP_top"], [False, False, False, False, True, False])
        np.testing.assert_array_equal(g["TN_bot"], [False, False, False, False, False, True])


class TestComparisons(unittest.TestCase):
    def test_rank_auc_matches_roc_auc(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0.5, 1, 40), rng.normal(0, 1, 60)
        want = roc_auc_score(np.r_[np.ones(40), np.zeros(60)], np.r_[a, b])
        self.assertAlmostEqual(E.rank_auc(a, b), want, places=10)

    def test_rank_auc_ignores_missing_and_handles_ties(self):
        self.assertEqual(E.rank_auc(np.array([2.0, np.nan]), np.array([1.0])), 1.0)
        self.assertEqual(E.rank_auc(np.array([1.0, 1.0]), np.array([1.0])), 0.5)
        self.assertTrue(np.isnan(E.rank_auc(np.array([np.nan]), np.array([1.0]))))

    def test_feature_diff(self):
        rng = np.random.default_rng(2)
        n = 400
        X = rng.normal(0, 1, (n, 3))
        a = np.zeros(n, dtype=bool)
        a[:100] = True
        X[a, 0] += 3.0                      # 群のほうが大きい列
        X[a, 1] = np.nan                    # 群では全部欠損の列
        X[np.flatnonzero(a)[:50], 2] = np.nan   # 群では半分欠損
        fold = np.arange(n) % 5
        d = E.feature_diff(X, a, ~a, ["x0", "x1", "x2"], fold).set_index("col")
        self.assertGreater(d.loc["x0", "auc"], 0.95)
        self.assertEqual(d.loc["x0", "agree"], 5)
        self.assertEqual(d.loc["x1", "miss_a"], 1.0)
        self.assertTrue(np.isnan(d.loc["x1", "auc"]))
        self.assertAlmostEqual(d.loc["x2", "miss_diff"], 0.5)

    def test_every_feature_has_a_family(self):
        cols = F.columns(F.DEFAULT_PRESET)
        fam = E.col_family(cols)
        self.assertEqual(set(fam), set(cols))
        self.assertNotIn("その他", set(fam.values()))
        self.assertIn("地合い（市場環境）", set(fam.values()))


class TestLabelParts(unittest.TestCase):
    def test_same_rule_as_build_dataset(self):
        cfg = B.DEFAULT_RISE
        vol = np.array([2.0, 2.0, 2.0, 2.0])
        need = cfg.vol_norm_k * vol / 100 * np.sqrt(cfg.horizon)
        end_need = cfg.end_ratio / cfg.threshold * need
        sub = pd.DataFrame({
            "vol_20d": vol,
            "future_rise": need * np.array([1.1, 0.9, 1.1, 1.1]),
            "end_level": end_need * np.array([1.2, 1.2, 0.8, 1.2]),
            "uptrend_end": [1.0, 1.0, 1.0, 0.0]})
        p = E.label_parts(sub)
        np.testing.assert_allclose(p["reach"], [1.1, 0.9, 1.1, 1.1])
        np.testing.assert_allclose(p["end"], [1.2, 1.2, 0.8, 1.2])
        # 到達・終盤・トレンドがそろったものだけが正例
        self.assertEqual(p["rebuilt"].tolist(), [True, False, False, False])


class TestPublicLogs(unittest.TestCase):
    def test_examples_are_refused_on_actions(self):
        old = os.environ.get("GITHUB_ACTIONS")
        os.environ["GITHUB_ACTIONS"] = "true"
        try:
            with self.assertRaises(SystemExit) as cm:
                E.main(["--params", "/nonexistent.json", "--examples", "3"])
            self.assertIn("公開ログ", str(cm.exception))
        finally:
            if old is None:
                del os.environ["GITHUB_ACTIONS"]
            else:
                os.environ["GITHUB_ACTIONS"] = old

    def test_production_config_is_not_written(self):
        self.assertTrue(E.PARAMS_PATH.startswith(E.OOF_DIR))
        self.assertNotIn("lgbm_params.json", E.PARAMS_PATH)
        self.assertNotIn("multi_params.json", E.PARAMS_PATH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
