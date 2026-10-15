#!/usr/bin/env python3
"""
実験92（research/exp/e92_knn_quantile_pick.py）。

- 近傍の指標: 手で数えた例（25%点・中央値・−10% 割れ・平均）と一致する
- 無作為の行列: 形が (回数 × 判断日) で、行ごとの平均は実験90 の random_means と同じ値（同じ乱数の順）
- 選び方: 同点は今の並べ方。−10% 割れの少なさは小さいほど良い向きで選ぶ
- 判定: 仕込んだ合成データ（形の違う銘柄だけ収益を高くする）で Q が M と無作為を 25%点で上回り「効く」。仕込まなければ「効く」にならない
- 面積の恒等式: 平均の差 = 上側 − 下側（Q 対 M）
- 報告が全部の段を出し、銘柄コードを出さない

  python3 tests/test_e92_knn_quantile_pick.py
"""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e90_wasserstein_knn_pick as E90  # noqa: E402
import e92_knn_quantile_pick as E  # noqa: E402
import test_e90_wasserstein_knn_pick as T90  # noqa: E402

LINE_T, POOL_T = T90.LINE_T, T90.POOL_T


class TestNeighborStats(unittest.TestCase):
    def test_hand_example(self):
        D = np.array([0.5, 0.1, 0.9, 0.2, 0.3])
        v = np.array([10.0, -20.0, 100.0, 3.0, 5.0])
        s = E.neighbor_stats(D, v, 3)                                    # 近い3件 = [-20, 3, 5]
        self.assertAlmostEqual(s["p25"], np.quantile([-20.0, 3.0, 5.0], 0.25))
        self.assertAlmostEqual(s["p50"], 3.0)
        self.assertAlmostEqual(s["bad"], 100.0 / 3)
        self.assertAlmostEqual(s["mean"], -4.0)

    def test_k_larger_than_pool(self):
        s = E.neighbor_stats(np.array([0.1, 0.2]), np.array([1.0, 3.0]), 5)
        self.assertAlmostEqual(s["mean"], 2.0)


class TestPicksAndDraws(unittest.TestCase):
    @staticmethod
    def frame():
        d = pd.Timestamp("2023-01-10")
        return pd.DataFrame({"Date": [d, d, d], "window": [5, 5, 5], "row": [0, 1, 2], "min15": [99.0, 98.0, 97.0],
                             "score": [0.5, 0.9, 0.1], E.TP: [1.0, 10.0, -12.0], E.HIT: [False, True, False],
                             "p25_50": [2.0, 2.0, -1.0], "bad_50": [10.0, 5.0, 5.0]})

    def test_tie_goes_to_current_ordering(self):
        pk = E.day_picks(self.frame(), "p25_50")
        self.assertEqual(len(pk), 1)
        self.assertAlmostEqual(pk["Q"].iat[0], 1.0)                      # 行0と行1が同点 → min15 の高い行0
        self.assertAlmostEqual(pk["M"].iat[0], 1.0)
        self.assertTrue(bool(pk["same"].iat[0]))

    def test_ascending_selector(self):
        pk = E.day_picks(self.frame(), "bad_50", ascending=True)
        self.assertAlmostEqual(pk["Q"].iat[0], 10.0)                     # −10% 割れが最少の行1・2のうち min15 の高い行1
        self.assertFalse(bool(pk["same"].iat[0]))

    def test_random_draws_match_e90(self):
        df, dist, split, starts, _ = T90.case(planted=4.0, seed=3)
        c, _ = E.collect(df, dist, split, starts, line=LINE_T, pool_line=POOL_T)
        mat = E.random_draws(c, draws=300, seed=7)
        self.assertEqual(mat.shape, (300, c["Date"].nunique()))
        np.testing.assert_allclose(mat.mean(axis=1), E90.random_means(c, draws=300, seed=7))


class TestPlanted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.dist, cls.split, cls.starts, cls.kinds = T90.case(planted=8.0)
        cls.c, cls.info = E.collect(cls.df, cls.dist, cls.split, cls.starts, line=LINE_T, pool_line=POOL_T)

    def test_columns_and_counts(self):
        for col in ("p25_50", "p50_50", "bad_50", "mean_50", "p25_100"):
            self.assertIn(col, self.c.columns)
        self.assertEqual(self.info["days"], self.c["Date"].nunique())
        self.assertGreater(self.info["days"], 40)

    def test_same_distances_as_e90(self):
        c90, _ = E90.collect(self.df, self.dist, self.split, self.starts, line=LINE_T, pool_line=POOL_T)
        m = self.c[["row", "mean_50"]].merge(c90[["row", "est_RV_50"]], on="row")
        self.assertEqual(len(m), len(self.c))
        np.testing.assert_allclose(m["mean_50"].to_numpy(), m["est_RV_50"].to_numpy())

    def test_effective(self):
        ev = E.evaluate(self.c, "p25_50", draws=500)
        self.assertTrue(ev["effective"], ev["conds"])
        self.assertGreaterEqual(ev["Q"]["p25"], ev["M"]["p25"])
        self.assertLess(ev["p_Q"], 0.05)
        a = ev["area_M"]
        self.assertAlmostEqual(ev["Q"]["mean"] - ev["M"]["mean"], a["up"] - a["down"], places=9)

    def test_report_sections_and_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.c, self.info, draws=200)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ まとめ", "判定", "優越からのずれ", "対の差"):
            self.assertIn(tag, out)
        for k in ("main", "ref", "verdict"):
            self.assertIn(k, s)
        self.assertIn("mean_50", s["ref"])
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


class TestNull(unittest.TestCase):
    def test_no_plant_is_not_effective(self):
        df, dist, split, starts, _ = T90.case(planted=0.0, seed=1)
        c, _ = E.collect(df, dist, split, starts, line=LINE_T, pool_line=POOL_T)
        ev = E.evaluate(c, "p25_50", draws=500)
        self.assertGreater(ev["days"], 40)
        self.assertFalse(ev["effective"])

    def test_empty(self):
        c = pd.DataFrame(columns=["Date", "window", "row", "min15", "score", E.TP, E.HIT, "p25_50"])
        self.assertEqual(E.evaluate(c, "p25_50")["days"], 0)
        self.assertEqual(E.dist_stats([])["n"], 0)


if __name__ == "__main__":
    unittest.main()
