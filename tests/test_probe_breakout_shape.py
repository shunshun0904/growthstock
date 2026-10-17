#!/usr/bin/env python3
"""
2列の probe（research/exp/probe_breakout_shape.py）の部品。ネットにもデータにも触らない。

- same: 両方欠測も同じと数える
- properties: base_length が窓以内・窓より長い行の数え分け（窓以内なら古さ = base_length・比は 100 超、が本物の形）
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

    def test_column_sets(self):
        base = F.columns(P.BASE_PRESET)
        self.assertEqual(F.signature(base), P.BASE_SIG)
        self.assertEqual(F.columns(P.SHAPE_PRESET), base + P.NEW)


if __name__ == "__main__":
    unittest.main()
