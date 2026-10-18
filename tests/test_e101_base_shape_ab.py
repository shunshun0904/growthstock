#!/usr/bin/env python3
"""
実験101（research/exp/e101_base_shape_ab.py）の2列の作り方。データには触らない（小さな日足の表で確かめる）。

- broken_high_age: 今回抜いた「それまでの w 行の高値の最大」を付けた行から何行か（同じ値なら一番新しい行。1〜w）
- prev_nh_ratio:   1つ前の更新日（base_length の起点）の高値に対する今回の高値（%）。base_length が欠測なら欠測
- keys の並びのまま返す。日足の表に無い行は欠測。t より先の行は見ない
- 比べる土台は239列（名前で固定。2026-10-18 から既定は249列）で、2列はそこに無い

  python3 tests/test_e101_base_shape_ab.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e101_base_shape_ab as E  # noqa: E402
import features as F  # noqa: E402


def panel():
    rows = []
    for code, highs, bls in (("10010", [10, 12, 11, 15, 9, 8, 16], [None, None, None, None, None, None, 3]),
                             ("10020", [5, 5, 4, 6], [None, None, None, 2])):
        for i, (h, bl) in enumerate(zip(highs, bls)):
            rows.append({"Code": code, "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "high": float(h),
                         "base_length": np.nan if bl is None else float(bl)})
    return pd.DataFrame(rows)


class TestShape(unittest.TestCase):
    def test_columns(self):
        p = panel()
        keys = pd.DataFrame({"Code": ["10020", "10010", "10010", "99990"],
                             "Date": pd.to_datetime(["2024-01-04", "2024-01-07", "2024-01-05", "2024-01-07"])})
        got = E.shape_columns(keys, p)
        self.assertEqual(list(got.columns), ["broken_high_age", "prev_nh_ratio"])
        self.assertEqual(len(got), 4)
        # 10020 の4行目: 窓 [5, 5, 4] の最大 5 は1・2行目 → 新しい方（2行目）から2行。前回の更新（2行前、高値5）に対し 6 → 120%
        self.assertEqual(got.iloc[0].tolist(), [2.0, 120.0])
        # 10010 の7行目: 窓の最大 15（4行目）から3行。前回の更新（3行前、高値15）に対し 16
        self.assertEqual(got.iloc[1, 0], 3.0)
        self.assertAlmostEqual(got.iloc[1, 1], 16 / 15 * 100)
        # 10010 の5行目: 窓の最大 15 は前の行（1行）。base_length が欠測なので比は欠測
        self.assertEqual(got.iloc[2, 0], 1.0)
        self.assertTrue(np.isnan(got.iloc[2, 1]))
        # 日足の表に無い行は両方欠測
        self.assertTrue(got.iloc[3].isna().all())

    def test_window_and_future(self):
        p = panel()
        keys = pd.DataFrame({"Code": ["10010"], "Date": [pd.Timestamp("2024-01-07")]})
        self.assertEqual(E.shape_columns(keys, p, w=2).iloc[0, 0], 2.0)        # 窓 [9, 8] の最大 9 は2行前
        # t より先の行を変えても値は変わらない（先を見ていない）
        later = pd.concat([p, pd.DataFrame([{"Code": "10010", "Date": pd.Timestamp("2024-01-08"), "high": 99.0,
                                             "base_length": 1.0}])], ignore_index=True)
        later = later.sort_values(["Code", "Date"]).reset_index(drop=True)
        self.assertEqual(E.shape_columns(keys, later).iloc[0].tolist(), E.shape_columns(keys, p).iloc[0].tolist())

    def test_peak_age(self):
        h = np.array([5.0, 9.0, 3.0, 9.0, 4.0, 2.0, 10.0])
        self.assertEqual(E.peak_age(h, 6, 6), 3.0)
        self.assertTrue(np.isnan(E.peak_age(np.array([np.nan, np.nan, 1.0]), 2, 2)))

    def test_constants(self):
        self.assertEqual(E.NEW_COLS, ("broken_high_age", "prev_nh_ratio"))
        self.assertEqual(E.PRIMARY_ROWS, "all")
        self.assertEqual(E.KS, (1.5, 1.2))
        self.assertEqual(E.JUDGE_ALGOS, ("lgbm", "xgb", "cat"))
        base = F.columns(E.BASE_PRESET)                 # 回した時点の本番の239列（2026-10-18 から既定は249列）
        self.assertEqual(len(base), 239)
        self.assertFalse(set(base) & set(E.NEW_COLS))


if __name__ == "__main__":
    unittest.main()
