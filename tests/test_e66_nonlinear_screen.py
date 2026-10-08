#!/usr/bin/env python3
"""
実験66（research/exp/e66_nonlinear_screen.py）の部品のテスト。

  python3 tests/test_e66_nonlinear_screen.py
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
import nonlinear_features as NL  # noqa: E402
import e66_nonlinear_screen as E66  # noqa: E402


class TestArms(unittest.TestCase):
    def test_v_is_t_plus_32_new_columns(self):
        base = F.columns(F.DEFAULT_PRESET)
        v = base + E66.NEW_COLS
        self.assertEqual(len(v), len(set(v)), "重複する列がある")
        self.assertEqual(len(E66.NEW_COLS), 32)
        self.assertFalse(set(E66.NEW_COLS) & set(F.all_columns()),
                         "候補の列は本番の設定に入れていない（採否が決まるまで）")

    def test_reference_columns_are_production_columns(self):
        prod = set(F.columns(F.DEFAULT_PRESET))
        for c in E66.REF_COLS:
            self.assertIn(c, prod, c)

    def test_labels_cover_arms(self):
        self.assertEqual(set(E66.LABELS), {"T", "V", "P"})


class TestWithinWindowPct(unittest.TestCase):
    def test_rank_is_within_window(self):
        d = pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-20", "2024-07-05", "2024-07-10"])
        df = pd.DataFrame({"Date": d, "x": [3.0, 1.0, 2.0, 10.0, np.nan]})
        windows = [(np.datetime64("2024-01-01"), np.datetime64("2024-06-30")),
                   (np.datetime64("2024-07-01"), np.datetime64("2024-12-31"))]
        p = E66.within_window_pct(df, "x", windows)
        # 1窓目: 1.0 < 2.0 < 3.0 → 1/3, 2/3, 3/3（並びは元の順）
        np.testing.assert_allclose(p[:3], [1.0, 1 / 3, 2 / 3])
        # 2窓目: 値のある行は1つなので 1.0、欠測は欠測のまま
        self.assertEqual(p[3], 1.0)
        self.assertTrue(np.isnan(p[4]))

    def test_outside_any_window_is_missing(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]), "x": [1.0]})
        p = E66.within_window_pct(df, "x", [(np.datetime64("2024-01-01"), np.datetime64("2024-12-31"))])
        self.assertTrue(np.isnan(p[0]))


class TestColumnsAgree(unittest.TestCase):
    def test_new_cols_match_module(self):
        self.assertEqual(E66.NEW_COLS, list(NL.NL_COLS))


if __name__ == "__main__":
    unittest.main()
