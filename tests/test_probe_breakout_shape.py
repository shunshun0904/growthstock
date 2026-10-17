#!/usr/bin/env python3
"""
2列の probe（research/exp/probe_breakout_shape.py）の部品。ネットにもデータにも触らない。

- same: 両方欠測も同じと数える
- properties: base_length が窓以内・窓より長い行の数え分け（窓以内なら古さ = base_length・比は 100 超、が本物の形）
- extremes: 両端の行の内訳（その日の上げ・前回の高値を超えていた日・分割や併合・1日で2倍超か半分未満の終値）
- 列の組: 239列の組の指紋が台本の期待値と同じ、241列の組はその後ろに2列

  python3 tests/test_probe_breakout_shape.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import probe_breakout_shape as P  # noqa: E402


class TestParts(unittest.TestCase):
    def test_same(self):
        a = pd.Series([1.0, np.nan, 3.0, np.nan])
        b = pd.Series([1.0, np.nan, 4.0, 2.0])
        self.assertEqual(P.same(a, b), 2)

    def test_properties(self):
        df = pd.DataFrame({"base_length": [30.0, 30.0, 400.0, 400.0, np.nan, np.nan],
                           "broken_high_age": [30.0, 12.0, 50.0, 368.0, 7.0, 0.0],
                           "prev_nh_ratio": [104.0, 101.0, 80.0, 102.0, np.nan, 99.0]})
        p = P.properties(df, 368)
        self.assertEqual((p["near"], p["near_age_eq"], p["near_age_lt"], p["near_age_gt"]), (2, 1, 1, 0))
        self.assertEqual(p["near_ratio_gt100"], 2)
        self.assertEqual((p["far"], p["far_ratio_lt100"]), (2, 1))
        self.assertEqual(p["age_out_of_range"], 1)            # 0 は 1〜368 の外
        self.assertEqual(p["ratio_with_no_base"], 1)          # base_length が欠測なのに比がある（本物では無いはず）
        self.assertEqual(p["ratio_nan_with_base"], 0)

    def test_extremes(self):
        # 銘柄A: 前回の更新日（位置2、高値10）の後に 9 まで → 位置5 で前日終値 8 から 14（+75%）の更新。比 140
        # 銘柄B: 前回の更新日（位置1、高値10）の後、位置3 で 11（更新日にならなかった）→ 位置4 で 14。比 140・その間に2倍超の終値
        # 銘柄C: 前回の更新日（位置1、高値100）の後に 1:2 の分割（位置3。その前の素の終値は調整後の2倍）。下げは1日で半分未満に
        #        ならない。位置6 で前日終値 15 から 19（+27%）。比 19
        panel = pd.DataFrame({
            "Code": ["A"] * 6 + ["B"] * 5 + ["C"] * 7,
            "Date": list(pd.date_range("2024-01-01", periods=6)) + list(pd.date_range("2024-01-01", periods=5))
            + list(pd.date_range("2024-01-01", periods=7)),
            "high": [8, 9, 10, 9, 9, 14, 9, 10, 4, 11, 14, 90, 100, 55, 30, 17, 16, 19],
            "close": [8, 9, 9.5, 8.5, 8, 13, 9, 9.5, 4, 10.5, 13, 90, 95, 52, 28, 16, 15, 18.5],
            "close_raw": [8, 9, 9.5, 8.5, 8, 13, 9, 9.5, 4, 10.5, 13, 180, 190, 104, 28, 16, 15, 18.5]})
        ds = pd.DataFrame({"Code": ["A", "B", "C"], "Date": pd.to_datetime(["2024-01-06", "2024-01-05", "2024-01-07"]),
                           "base_length": [3.0, 3.0, 5.0], "prev_nh_ratio": [140.0, 140.0, 19.0]})
        ex = P.extremes(ds, panel)
        self.assertEqual(ex["hi"], {"rows": 2, "day": 2, "exceed": 1, "action": 0, "jump": 1, "far": 0})
        self.assertEqual(ex["lo"], {"rows": 1, "day": 0, "exceed": 0, "action": 1, "jump": 0, "far": 0})

    def test_column_sets(self):
        base = F.columns(P.BASE_PRESET)
        self.assertEqual(F.signature(base), P.BASE_SIG)
        self.assertEqual(F.columns(P.SHAPE_PRESET), base + P.NEW)


if __name__ == "__main__":
    unittest.main()
