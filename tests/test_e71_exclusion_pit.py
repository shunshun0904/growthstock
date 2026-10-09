#!/usr/bin/env python3
"""実験71（research/exp/e71_exclusion_pit.py）の部品のテスト。"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e71_exclusion_pit as E71  # noqa: E402


def world(n_fold=5, n=400, seed=0):
    """x1 が高いほど収益が低い。x2 は無関係。母集団 ref と選定 sel（母集団の一部）を返す。"""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fold):
        for i in range(n):
            x1, x2 = rng.normal(), rng.normal()
            rows.append({"Code": f"{i:05d}", "Date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=f * 200 + i % 100),
                         "fold": f, "x1": x1, "x2": x2, "ret_o1_20": -0.02 * x1 + rng.normal(0, 0.01)})
    ref = pd.DataFrame(rows)
    sel = ref.sample(frac=0.5, random_state=1).sort_values(["fold", "Date"]).reset_index(drop=True)
    return ref, sel


class TestChooseColumn(unittest.TestCase):
    def test_picks_the_informative_column_and_side(self):
        ref, sel = world()
        prior = sel[sel["fold"] < 3]
        pick = E71.choose_column(prior, ref, 3, ["x1", "x2"])
        self.assertIsNotNone(pick)
        col, side, d = pick
        self.assertEqual(col, "x1")
        self.assertEqual(side, "top")      # 高いほど悪い → 上を外す
        self.assertLess(d, 0)

    def test_needs_prior_population(self):
        ref, sel = world()
        # 窓0 には前の窓が無い → しきい値が作れず None
        self.assertIsNone(E71.choose_column(sel[sel["fold"] < 0], ref, 0, ["x1", "x2"]))


class TestRunSelection(unittest.TestCase):
    def test_gain_is_positive_when_the_rule_is_right(self):
        ref, sel = world(n=600)          # 窓0 だけで 600行 → 窓1 から前の窓の母集団（500行以上）が作れる
        saved = E71.FIXED_COL
        E71.FIXED_COL = "x1"
        try:
            res = E71.run_selection(sel, ref, ["x1", "x2"], min_prior=100)
        finally:
            E71.FIXED_COL = saved
        self.assertGreater(len(res), 0)
        g = res["pit_半分_gain"].dropna()
        self.assertGreater(len(g), 0)
        self.assertTrue((g > 0).all())
        self.assertTrue(res.loc[g.index, "pit_半分_col"].str.startswith("x1:top").all())
        # 前の窓が無い窓0 は入らない
        self.assertNotIn(0, res["fold"].tolist())

    def test_prior_threshold_uses_only_earlier_folds(self):
        ref, _ = world()
        ref.loc[ref["fold"] == 4, "x1"] += 100.0     # 先の窓だけ値をずらす
        self.assertAlmostEqual(E71.prior_threshold(ref, "x1", 4, 50.0), float(np.percentile(ref.loc[ref["fold"] < 4, "x1"], 50)))
        self.assertTrue(np.isnan(E71.prior_threshold(ref, "x1", 0, 50.0)))


if __name__ == "__main__":
    unittest.main()
