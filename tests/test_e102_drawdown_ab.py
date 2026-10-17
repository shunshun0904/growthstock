#!/usr/bin/env python3
"""
実験102（research/exp/e102_drawdown_ab.py）の3列の作り方。データには触らない（小さな日足の表で確かめる）。

- base_depth:     1つ前の更新日（base_length の起点）の高値から、その翌日〜t の前日の安値の最小までの下げ（%）。間が無ければ 0
- pullback_depth: 今回抜いた高値を付けた日（t − broken_high_age）の高値から、同じく翌日〜t の前日の安値の最小までの下げ（%）
- max_dd_120:     直近120行（t を含む）の終値の最大ドローダウン（%）。終値が60未満なら欠測
- keys の並びのまま返す。日足の表に無い行・起点が欠測の行は欠測。t より先の行は見ない
- 比べる土台は241列（2026-10-18 からの既定）で、3列はそこに無い

  python3 tests/test_e102_drawdown_ab.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e102_drawdown_ab as E  # noqa: E402
import features as F  # noqa: E402


class TestDepth(unittest.TestCase):
    def test_depth_from(self):
        high = np.array([10.0, 12.0, 9.0, 8.0, 11.0, 13.0])
        low = np.array([9.5, 11.0, 8.5, 7.0, 10.5, 12.0])
        self.assertAlmostEqual(E.depth_from(high, low, 1, 5), (1 - 7.0 / 12.0) * 100)   # 間の安値 8.5 / 7 / 10.5 の最小 7
        self.assertEqual(E.depth_from(high, low, 4, 5), 0.0)                           # 間の日が無い
        self.assertTrue(np.isnan(E.depth_from(high, low, -1, 5)))                      # 起点が無い
        nan_low = low.copy()
        nan_low[2:5] = np.nan
        self.assertTrue(np.isnan(E.depth_from(high, nan_low, 1, 5)))                   # 間の安値が全部欠測
        nan_low[3] = 9.0
        self.assertAlmostEqual(E.depth_from(high, nan_low, 1, 5), 25.0)                # 欠測は飛ばす

    def test_max_drawdown(self):
        close = np.array([10.0, 12.0, 9.0, 11.0, 6.0, 8.0])
        self.assertAlmostEqual(E.max_drawdown(close, 5, days=6, need=3), 50.0)         # 12 → 6
        self.assertAlmostEqual(E.max_drawdown(close, 5, days=3, need=3), (1 - 6 / 11) * 100)   # 窓 [11, 6, 8]
        self.assertTrue(np.isnan(E.max_drawdown(close, 5, days=3, need=4)))            # 終値が足りない
        self.assertEqual(E.max_drawdown(np.array([1.0, 2.0, 3.0]), 2, need=3), 0.0)    # 上がり続けたら 0
        self.assertTrue(np.isnan(E.max_drawdown(np.array([1.0, 0.0, 3.0]), 2, need=3)))   # 0 以下の終値


def panel():
    rows = []
    spec = {"10010": ([10, 12, 9, 8, 11, 13], [9.5, 11, 8.5, 7, 10.5, 12], [9.8, 11.5, 8.8, 7.5, 10.8, 12.5],
                      [None, None, None, None, None, 4], [None, None, None, None, None, 4]),
            "10020": ([5, 6, 7], [4.5, 5.5, 6.5], [4.8, 5.8, 6.8], [None, 1, 1], [None, 1, 1])}
    for code, (hi, lo, cl, bl, age) in spec.items():
        for i in range(len(hi)):
            rows.append({"Code": code, "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "high": float(hi[i]),
                         "low": float(lo[i]), "close": float(cl[i]),
                         "base_length": np.nan if bl[i] is None else float(bl[i]),
                         "broken_high_age": np.nan if age[i] is None else float(age[i])})
    return pd.DataFrame(rows)


class TestColumns(unittest.TestCase):
    def test_columns_in_key_order(self):
        keys = pd.DataFrame({"Code": ["10020", "10010", "99990"],
                             "Date": pd.to_datetime(["2024-01-03", "2024-01-06", "2024-01-06"])})
        got = E.drawdown_columns(keys, panel())
        self.assertEqual(list(got.columns), list(E.NEW_COLS))
        self.assertEqual(got.iloc[0, :2].tolist(), [0.0, 0.0])            # 前日に更新・前日の高値を抜いた → 間の日が無い
        self.assertAlmostEqual(got.iloc[1, 0], (1 - 7.0 / 12.0) * 100)     # 4行前（高値12）からの底 7
        self.assertAlmostEqual(got.iloc[1, 1], (1 - 7.0 / 12.0) * 100)     # 抜いた高値も4行前
        self.assertTrue(got["max_dd_120"].isna().all())                    # 終値が60に足りない（3つ・6つ）ので出さない
        self.assertTrue(got.iloc[2].isna().all())                          # 日足の表に無い行

    def test_future_rows_do_not_change_values(self):
        p = panel()
        keys = pd.DataFrame({"Code": ["10010"], "Date": [pd.Timestamp("2024-01-05")]})
        p.loc[(p["Code"] == "10010") & (p["Date"] == pd.Timestamp("2024-01-05")), ["base_length", "broken_high_age"]] = 3.0
        later = p.copy()
        later.loc[(later["Code"] == "10010") & (later["Date"] == pd.Timestamp("2024-01-06")),
                  ["high", "low", "close"]] = [99.0, 1.0, 50.0]
        a = E.drawdown_columns(keys, p)
        b = E.drawdown_columns(keys, later)
        pd.testing.assert_frame_equal(a, b)


class TestConstants(unittest.TestCase):
    def test_design(self):
        self.assertEqual(E.NEW_COLS, ("base_depth", "pullback_depth", "max_dd_120"))
        self.assertEqual((E.DD_DAYS, E.DD_MIN_CLOSES), (120, 60))
        self.assertEqual(E.PRIMARY_ROWS, "all")
        self.assertEqual(E.PERM_SEED, 20261018)
        self.assertEqual(E.KS[0], 1.5)
        base = F.columns(E.BASE_PRESET)
        self.assertEqual(len(base), 241)
        self.assertFalse(set(base) & set(E.NEW_COLS))
        self.assertEqual(E.TAG, "e102")


if __name__ == "__main__":
    unittest.main()
