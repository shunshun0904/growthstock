#!/usr/bin/env python3
"""
実験108（research/exp/e108_recurrence_tda.py）の部品。データには触らない（手で解ける小さな例で確かめる）。

- 持続図の数字: H0 の消えない1本は数えない。最大の割れ目・輪の合計・輪の最大・ランドスケープの L1 ノルム（Σ 寿命² / 4）。
  輪が無ければ 0
- 今日のまわりの点: 今日を入れて LOCAL + 1 点。タイラー窓（20ノード）の中の日は入らない。同じ距離なら古い日が先
- 軌跡の点: 直近 TRAJ ノード（今日を含む）。ノードが少なければあるだけ
- 判定: 98.75% 区間が 0.5 をまたがず、0.5 から 0.02 以上離れていれば「効く」（向きは問わない）
- 銘柄の中だけの AUC: 同じ銘柄の組だけを数える（銘柄どうしの差は効かない）
- ripser があれば: 1辺 1 の正方形の4点で、輪が1つ（距離 1 で生まれ √2 で消える）・割れ目 1

  python3 tests/test_e108_recurrence_tda.py
"""
import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e108_recurrence_tda as E  # noqa: E402

HAS_RIPSER = importlib.util.find_spec("ripser") is not None


class Summaries(unittest.TestCase):
    def test_numbers(self):
        h0 = np.array([[0, 0.5], [0, 1.0], [0, np.inf]])
        h1 = np.array([[0.2, 0.6], [0.3, 0.4]])
        s = E.summaries(h0, h1)
        self.assertAlmostEqual(s["h0_gap"], 1.0)
        self.assertAlmostEqual(s["h1_total"], 0.5)
        self.assertAlmostEqual(s["h1_max"], 0.4)
        self.assertAlmostEqual(s["h1_land"], (0.4 ** 2 + 0.1 ** 2) / 4)

    def test_no_loops(self):
        s = E.summaries(np.array([[0, np.inf]]), np.zeros((0, 2)))
        self.assertEqual(s, {"h0_gap": 0.0, "h1_total": 0.0, "h1_max": 0.0, "h1_land": 0.0})


class Clouds(unittest.TestCase):
    def test_local_index(self):
        m = 300
        rng = np.random.default_rng(0)
        s2 = rng.integers(0, 50, m)                  # 同じ距離がたくさんある
        got = E.local_index(s2, m)
        self.assertEqual(len(got), E.LOCAL + 1)
        self.assertEqual(got[-1], m - 1)             # 今日
        self.assertTrue((got[:-1] < m - 1 - E.TH).all())   # タイラー窓の中の日は入らない
        c = np.arange(m - 1 - E.TH)
        want = sorted(c, key=lambda i: (s2[i], i))[:E.LOCAL]
        self.assertEqual(got[:-1].tolist(), want)    # 距離の近い順、同じなら古い日が先

    def test_traj_index(self):
        self.assertEqual(E.traj_index(300).tolist(), list(range(240, 300)))
        self.assertEqual(E.traj_index(40).tolist(), list(range(40)))


class Verdict(unittest.TestCase):
    def test_rule(self):
        self.assertTrue(E.verdict(0.53, 0.51, 0.55))
        self.assertTrue(E.verdict(0.47, 0.45, 0.49))      # 逆向きも数える
        self.assertFalse(E.verdict(0.515, 0.505, 0.525))  # 0.5 から 0.02 離れていない
        self.assertFalse(E.verdict(0.53, 0.49, 0.57))     # 区間が 0.5 をまたぐ
        self.assertAlmostEqual(E.LEVEL, 0.9875)


class WithinAuc(unittest.TestCase):
    def test_only_pairs_in_the_same_stock(self):
        # 銘柄 A は数字が小さくて到達しやすく、B は数字が大きくて到達しにくい。どちらの銘柄の中でも数字が大きい日が到達する
        df = pd.DataFrame({
            "code": ["A"] * 4 + ["B"] * 4,
            "y": [0, 1, 0, 1, 0, 1, 0, 0],
            "q": [0.1, 0.2, 0.15, 0.3, 5.0, 6.0, 5.5, 4.0],
        })
        a, lo, hi = E.within_auc(df, "q", n_boot=50)
        self.assertAlmostEqual(a, 1.0)
        self.assertLess(E.auc(df["y"].to_numpy(float), df["q"].to_numpy(float)), 1.0)   # まとめると銘柄の差が混ざる


@unittest.skipUnless(HAS_RIPSER, "ripser が無い（pip install ripser）")
class Ripser(unittest.TestCase):
    def test_square(self):
        X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        s = E.summaries(*E.persistence(X))
        self.assertAlmostEqual(s["h0_gap"], 1.0)
        self.assertAlmostEqual(s["h1_max"], np.sqrt(2) - 1)
        self.assertAlmostEqual(s["h1_land"], (np.sqrt(2) - 1) ** 2 / 4)


if __name__ == "__main__":
    unittest.main()
