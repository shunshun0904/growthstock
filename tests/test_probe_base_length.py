#!/usr/bin/env python3
"""
base_length の probe（research/exp/probe_base_length.py）の部品。ネットにもデータにも触らない。

- 数え直し（recompute）が build_dataset.add_breakout_context の base_length と一致する（銘柄をまたがない・最初の更新は欠測）
- 抜いた高値の古さ（peak_age）は、窓の中の最大を付けた一番新しい行からの行数

  python3 tests/test_probe_base_length.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import probe_base_length as P  # noqa: E402


def panel(flags):
    """銘柄ごとの更新日の印から、add_breakout_context に渡せる最小の表を作る。"""
    rows = []
    for code, fl in flags.items():
        for i, f in enumerate(fl):
            rows.append({"Code": code, "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                         "is_new_high": f, "close": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i,
                         "high52w_prior": 100.0, "vol_20d": 2.0})
    return pd.DataFrame(rows)


class TestRecompute(unittest.TestCase):
    def test_matches_build_dataset(self):
        df = panel({"10010": [False, True, False, False, True, True, False, True],
                    "10020": [False, True, False, False, True],         # 本物と同じく、銘柄の先頭は更新日にならない
                    "10030": [False, False, False]})
        got = B.add_breakout_context(df.copy())["base_length"]
        again = P.recompute(df)
        same = (got == again) | (got.isna() & again.isna())
        self.assertTrue(same.all(), (got.tolist(), again.tolist()))
        self.assertTrue(np.isnan(again.iloc[1]))                   # 銘柄の最初の更新日は欠測
        self.assertEqual(again.iloc[4], 3.0)
        self.assertEqual(again.iloc[5], 1.0)
        self.assertEqual(again.iloc[7], 2.0)
        self.assertTrue(np.isnan(again.iloc[9]))                   # 2つ目の銘柄の最初の更新日（前の銘柄をまたがない）
        self.assertEqual(again.iloc[12], 3.0)
        self.assertTrue(again.iloc[13:].isna().all())             # 更新日の無い銘柄は欠測


class TestPeakAge(unittest.TestCase):
    def test_peak_age(self):
        h = np.array([5.0, 9.0, 3.0, 9.0, 4.0, 2.0, 10.0])
        self.assertEqual(P.peak_age(h, 6, 6), 3.0)                 # 窓 0..5 の最大 9 は位置1と3。新しい方（3）から3行
        self.assertEqual(P.peak_age(h, 6, 2), 2.0)                 # 窓 4..5 の最大 4 は位置4
        self.assertEqual(P.peak_age(np.array([np.nan, 7.0, np.nan, 1.0]), 3, 3), 2.0)
        self.assertTrue(np.isnan(P.peak_age(np.array([np.nan, np.nan, 1.0]), 2, 2)))
        self.assertTrue(np.isnan(P.peak_age(h, 0, 5)))


if __name__ == "__main__":
    unittest.main()
