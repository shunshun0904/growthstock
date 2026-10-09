#!/usr/bin/env python3
"""実験67（research/exp/e67_few_nl.py）の部品のテスト。"""
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
import e67_few_nl as E67  # noqa: E402


class TestColumns(unittest.TestCase):
    def test_one_column_per_family_and_all_exist(self):
        self.assertEqual(len(E67.FEW_COLS), 6)
        self.assertEqual(len(set(E67.FEW_COLS)), 6)
        for c in E67.FEW_COLS:
            self.assertIn(c, NL.NL_COLS)
        self.assertFalse(set(E67.FEW_COLS) & set(F.all_columns()), "本番の列と重複しない")
        self.assertEqual(sorted(E67.FEW_BY_FAMILY), ["A 持続性", "B 複雑さ", "C 非線形性", "D 再帰性", "E 経路の形", "F 出来高"])

    def test_judge_shifts_exclude_the_selection_window(self):
        self.assertNotIn(0, E67.JUDGE_SHIFTS)


class TestJudge(unittest.TestCase):
    def frame(self, v6_delta, p6_delta, n_fold=5):
        rows = []
        for sh in (0, 2, 4):
            for fold in range(n_fold):
                for arm, dlt in (("V6", v6_delta), ("P6", p6_delta)):
                    rows.append({"shift": sh, "arm": arm, "algo": "lgbm", "fold": fold,
                                 "pr_base": 0.25, "pr_arm": 0.25 + dlt(sh, fold),
                                 "roc_base": 0.64, "roc_arm": 0.64 + dlt(sh, fold)})
        return pd.DataFrame(rows)

    def test_positive_majority_above_control_passes(self):
        s = self.frame(lambda sh, f: 0.01, lambda sh, f: 0.0)
        r = E67.judge(s)
        self.assertEqual(int(r.loc[0, "judge_V6_n"]), 10)      # ずらし2・4 の 5窓ずつ
        self.assertEqual(int(r.loc[0, "ref_V6_n"]), 5)         # ずらし0 は参考
        self.assertTrue(bool(r.loc[0, "ok"]))

    def test_only_half_of_windows_fails(self):
        # ずらし2・4 で 4/8 しか上がらない → 過半でない（窓は 0..3 の4つ）
        s = self.frame(lambda sh, f: 0.01 if f % 2 == 0 else -0.01, lambda sh, f: 0.0, n_fold=4)
        r = E67.judge(s)
        self.assertFalse(bool(r.loc[0, "ok"]))

    def test_below_control_fails_even_if_positive(self):
        s = self.frame(lambda sh, f: 0.01, lambda sh, f: 0.02)
        self.assertFalse(bool(E67.judge(s).loc[0, "ok"]))

    def test_shift0_does_not_decide(self):
        # ずらし0 だけ良く、2・4 は悪い → 不合格
        s = self.frame(lambda sh, f: 0.05 if sh == 0 else -0.01, lambda sh, f: 0.0)
        r = E67.judge(s)
        self.assertFalse(bool(r.loc[0, "ok"]))
        self.assertGreater(float(r.loc[0, "ref_V6_pr"]), 0)


if __name__ == "__main__":
    unittest.main()
