#!/usr/bin/env python3
"""
実験103（research/exp/e103_recent_change_ab.py）の5列の作り方。データには触らない（小さな日足の表で確かめる）。

- ret_1d / ret_2d / ret_5d: 1・2・5行前の終値 → 当日の終値の騰落率（%）
- gap_open: 前の行の終値 → 当日の始値（%）
- vol_chg_1d: 当日の出来高 ÷ 前の行の出来高（%）。前の行が 0・欠測なら欠測
- 前の行は同じ銘柄の中だけ（銘柄の切れ目をまたがない）。keys の並びのまま。日足の表に無い行は欠測。t より先は見ない
- 比べる土台は244列（2026-10-18 からの既定）で、5列はそこに無い

  python3 tests/test_e103_recent_change_ab.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e103_recent_change_ab as E  # noqa: E402
import features as F  # noqa: E402


def panel():
    rows = []
    spec = {"10010": ([100, 102, 101, 110, 108, 120], [99, 101, 102, 104, 109, 115], [10, 20, 0, 30, 15, 45]),
            "10020": ([50, 55, np.nan, 60], [50, 54, np.nan, 58], [5, 8, 0, 12])}
    for code, (cl, op, vo) in spec.items():
        for i in range(len(cl)):
            rows.append({"Code": code, "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                         "close": float(cl[i]), "open": float(op[i]), "vol": float(vo[i])})
    return pd.DataFrame(rows)


class TestRecent(unittest.TestCase):
    def test_hand_counted(self):
        keys = pd.DataFrame({"Code": ["10010"], "Date": [pd.Timestamp("2024-01-06")]})
        got = E.recent_columns(keys, panel()).iloc[0]
        self.assertAlmostEqual(got["ret_1d"], (120 / 108 - 1) * 100)
        self.assertAlmostEqual(got["ret_2d"], (120 / 110 - 1) * 100)
        self.assertAlmostEqual(got["gap_open"], (115 / 108 - 1) * 100)
        self.assertAlmostEqual(got["ret_5d"], (120 / 100 - 1) * 100)
        self.assertAlmostEqual(got["vol_chg_1d"], 45 / 15 * 100)

    def test_missing_and_boundaries(self):
        keys = pd.DataFrame({"Code": ["10010", "10010", "10020", "10020", "99990"],
                             "Date": pd.to_datetime(["2024-01-04", "2024-01-01", "2024-01-01", "2024-01-04",
                                                     "2024-01-04"])})
        got = E.recent_columns(keys, panel())
        self.assertEqual(list(got.columns), list(E.NEW_COLS))
        self.assertTrue(np.isnan(got.loc[0, "vol_chg_1d"]))         # 前の行の出来高が 0
        self.assertAlmostEqual(got.loc[0, "ret_1d"], (110 / 101 - 1) * 100)
        self.assertTrue(np.isnan(got.loc[0, "ret_5d"]))              # 5行前が無い
        self.assertTrue(got.loc[1].isna().all())                     # 銘柄の先頭（前の行が無い）
        self.assertTrue(got.loc[2].isna().all())                     # 2つ目の銘柄の先頭（前の銘柄をまたがない）
        self.assertTrue(np.isnan(got.loc[3, "ret_1d"]))              # 前の行の終値が欠測
        self.assertAlmostEqual(got.loc[3, "ret_2d"], (60 / 55 - 1) * 100)
        self.assertTrue(np.isnan(got.loc[3, "vol_chg_1d"]))          # 前の行の出来高が 0
        self.assertTrue(got.loc[4].isna().all())                     # 日足の表に無い行

    def test_future_rows_do_not_change_values(self):
        keys = pd.DataFrame({"Code": ["10010"], "Date": [pd.Timestamp("2024-01-05")]})
        p = panel()
        later = p.copy()
        later.loc[5, ["close", "open", "vol"]] = [999.0, 1.0, 1e6]
        pd.testing.assert_frame_equal(E.recent_columns(keys, p), E.recent_columns(keys, later))

    def test_no_open_column(self):
        keys = pd.DataFrame({"Code": ["10010"], "Date": [pd.Timestamp("2024-01-06")]})
        got = E.recent_columns(keys, panel().drop(columns=["open"])).iloc[0]
        self.assertTrue(np.isnan(got["gap_open"]))
        self.assertAlmostEqual(got["ret_1d"], (120 / 108 - 1) * 100)
        got = E.recent_columns(keys, panel().drop(columns=["open", "vol"])).iloc[0]     # 出来高も無い表
        self.assertTrue(np.isnan(got["vol_chg_1d"]))
        self.assertAlmostEqual(got["ret_5d"], (120 / 100 - 1) * 100)


class TestConstants(unittest.TestCase):
    def test_design(self):
        self.assertEqual(E.NEW_COLS, ("ret_1d", "ret_2d", "gap_open", "ret_5d", "vol_chg_1d"))
        self.assertEqual(E.PRIMARY_ROWS, "all")
        self.assertEqual(E.PERM_SEED, 20261019)
        self.assertEqual(E.KS[0], 1.5)
        self.assertEqual(E.TAG, "e103")
        base = F.columns(E.BASE_PRESET)
        self.assertEqual(len(base), 244)
        self.assertFalse(set(base) & set(E.NEW_COLS))
        self.assertTrue(set(E.NEAR) <= set(base))
        for c in E.NEW_COLS:
            self.assertEqual(list(E.BINS[c]), sorted(E.BINS[c]))      # 区切りは増える順


if __name__ == "__main__":
    unittest.main()
