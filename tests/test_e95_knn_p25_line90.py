#!/usr/bin/env python3
"""
実験95（research/exp/e95_knn_p25_line90.py）: 後工程の候補の線を90以上に緩めて、実験93 と同じ判定を回す。

- 線: 後工程の候補は 90、買う規則の線（内訳）は 95、近傍の母集団は実験93 と同じ 85
- 線を下げると判断日は減らない（合成データで 75 と 80 を比べる）
- 内訳: 厳しい線以上の候補がある日の印、Q と M の3本の最小、Q が下の帯を選んだ割合
- 報告が全部の段を出し、銘柄コードを出さない

  python3 tests/test_e95_knn_p25_line90.py
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

import e93_knn_quantile_pick as E93  # noqa: E402
import e95_knn_p25_line90 as E  # noqa: E402
import test_e91_wasserstein_knn_pick as T91  # noqa: E402

POOL_T = T91.POOL_T


class TestLines(unittest.TestCase):
    def test_fixed(self):
        self.assertEqual(E.LINE, 90.0)
        self.assertEqual(E.STRICT, 95.0)
        self.assertEqual(E.POOL_LINE, 85.0)
        self.assertEqual(E.MAIN_KEY, "p25_50")

    def test_lower_line_gives_more_days(self):
        df, dist, split, starts, _ = T91.case(planted=4.0, seed=2)
        c80, i80 = E93.collect(df, dist, split, starts, line=80.0, pool_line=POOL_T)
        c75, i75 = E93.collect(df, dist, split, starts, line=75.0, pool_line=POOL_T)
        self.assertGreater(i75["days"], i80["days"])
        self.assertGreater(len(c75), len(c80))
        self.assertTrue((c75["min15"] >= 75.0).all())


class TestBreakdown(unittest.TestCase):
    @staticmethod
    def frame():
        d1, d2 = pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-11")
        return pd.DataFrame({"Date": [d1, d1, d1, d2, d2], "window": [5] * 5, "row": [0, 1, 2, 3, 4],
                             "min15": [96.0, 92.0, 91.0, 93.0, 90.5], "score": [0.5, 0.9, 0.1, 0.4, 0.3],
                             E93.TP: [1.0, 10.0, -12.0, 3.0, 4.0], E93.HIT: [False, True, False, False, False],
                             "p25_50": [0.0, 2.0, -1.0, -3.0, 1.0]})

    def test_flags_and_picks(self):
        c = self.frame()
        flags = E.day_flags(c, 95.0)
        self.assertEqual(flags.to_dict(), {pd.Timestamp("2023-01-10"): True, pd.Timestamp("2023-01-11"): False})
        pm = E.pick_min15(c, "p25_50")
        self.assertEqual(pm["Q_min15"].tolist(), [92.0, 90.5])        # 近傍の25%点の最上位
        self.assertEqual(pm["M_min15"].tolist(), [96.0, 93.0])        # 今の並べ方の先頭
        bd = E.breakdown(c, "p25_50", 95.0)
        self.assertEqual((bd["days"], bd["days_strict"], bd["days_band_only"]), (2, 1, 1))
        self.assertAlmostEqual(bd["Q_strict_share"], 0.0)
        self.assertAlmostEqual(bd["M_strict_share"], 50.0)
        self.assertAlmostEqual(bd["Q_band_when_strict"], 100.0)       # 95以上がある日に 92 を選んだ

    def test_empty(self):
        c = self.frame().iloc[0:0]
        self.assertEqual(E.breakdown(c)["days"], 0)
        self.assertEqual(len(E.day_flags(c)), 0)


class TestReport(unittest.TestCase):
    def test_sections_and_no_codes(self):
        df, dist, split, starts, _ = T91.case(planted=8.0)
        c, info = E93.collect(df, dist, split, starts, line=75.0, pool_line=POOL_T)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(c, info, draws=100, n_days_strict_line=10)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ まとめ", "判定", "内訳", "優越からのずれ"):
            self.assertIn(tag, out)
        for k in ("main", "breakdown", "ref", "verdict"):
            self.assertIn(k, s)
        for code in df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
