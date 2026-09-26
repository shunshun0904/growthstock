#!/usr/bin/env python3
"""
実験57（research/exp/e57_tune_239.py）の腕の定義のテスト。

- A は 206列（all_plus_prog_listing）、B / C は 239列（all_plus_prog_listing_vol_book = 本番）
- C は A のパラメータを使う（探索は無し）
- 探索の条件は本番の週次実行と同じ（木の本数・学習率の範囲・5分割・year_cap_date）
- 記録は research/_data/oof/e57_* で、本番の設定には書かない

  python3 tests/test_e57_tune_239.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import features as F  # noqa: E402
import tuning  # noqa: E402
import e57_tune_239 as E  # noqa: E402


class TestArms(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(E.ARMS["A"], ("all_plus_prog_listing", True))
        self.assertEqual(E.ARMS["B"], ("all_plus_prog_listing_vol_book", True))
        self.assertEqual(E.ARMS["C"], ("all_plus_prog_listing_vol_book", False))
        self.assertEqual(E.PARAMS_FROM, {"C": "A"})
        self.assertEqual(len(E.arm_cols("A")), 206)
        self.assertEqual(len(E.arm_cols("B")), 239)
        self.assertEqual(E.arm_cols("B"), E.arm_cols("A") + F.GROUPS["vol_factors"] + F.GROUPS["fund_book"])

    def test_new_preset_is_production(self):
        self.assertEqual(E.NEW_PRESET, F.DEFAULT_PRESET)

    def test_tuning_matches_weekly(self):
        self.assertEqual(E.N_SPLITS, 5)
        self.assertEqual(E.CV_SCHEME, "year_cap_date")
        self.assertEqual(tuning.SEARCH_N_ESTIMATORS, 200)
        self.assertEqual(tuning.lr_range(200), (0.01, 0.2))
        self.assertEqual(E.SEEDS, (42, 7, 123))

    def test_outputs_stay_out_of_production(self):
        self.assertTrue(E.OOF_DIR.endswith(os.path.join("_data", "oof")))
        self.assertNotIn("lgbm_params", E.OOF_DIR)
        self.assertEqual(set(E.LABELS), set(E.ARMS))
        for x, y in E.PAIRS:
            self.assertIn(x, E.ARMS)
            self.assertIn(y, E.ARMS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
