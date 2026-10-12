#!/usr/bin/env python3
"""
実験87（research/exp/e87_monthly_return_search.py）。

- 速い回し方: 1枠なら、指値に届いた日の翌営業日から次を買える。届かなければ20営業日の翌営業日から。収益の無い候補は枠を
  ふさがない。月の上限
- 月の損益（資金全体に対する%）: 取引の無い月も数えた平均、前半・後半、累計の一番大きな落ち込み
- 合成データで、画面の形が実験83 の trade_sim と、月2件の先着が実験84 の first_by と同じ取引になる
- 報告に銘柄コードが出ない

  python3 tests/test_e87_monthly_return_search.py
"""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e79_model_clusters as E79  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


class TestSimulate(unittest.TestCase):
    def test_slot_frees_after_hit_or_hold(self):
        # 日 0 の候補: 2営業日目で届く → 買った日 1、枠は 1 + 2 = 3 から空く
        day = np.array([0, 1, 2, 3], dtype=float)
        ret = np.array([10.0, 5.0, -3.0, 1.0])
        hit = np.array([True, False, False, False])
        hd = np.array([2.0, np.nan, np.nan, np.nan])
        m = np.zeros(4, dtype=int)
        self.assertEqual(E.simulate(day, ret, hit, hd, m, slots=1, cap=0), [0, 2])     # 日1 は空き無し、日2 は買える
        # 届かなければ20営業日ふさぐ
        self.assertEqual(E.simulate(np.array([0.0, 19.0, 20.0]), np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=bool),
                                    np.full(3, np.nan), np.zeros(3, dtype=int), slots=1, cap=0), [0, 2])
        # 収益が無い候補は飛ばして枠をふさがない
        self.assertEqual(E.simulate(np.array([0.0, 0.0]), np.array([np.nan, 4.0]), np.zeros(2, dtype=bool),
                                    np.full(2, np.nan), np.zeros(2, dtype=int), slots=1, cap=0), [1])

    def test_monthly_cap(self):
        day = np.array([0.0, 5.0, 30.0])
        ret = np.array([1.0, 2.0, 3.0])
        m = np.array([0, 0, 1])
        self.assertEqual(E.simulate(day, ret, np.zeros(3, dtype=bool), np.full(3, np.nan), m, slots=5, cap=1), [0, 2])
        self.assertEqual(E.simulate(day, ret, np.zeros(3, dtype=bool), np.full(3, np.nan), m, slots=5, cap=0), [0, 1, 2])


class TestMetrics(unittest.TestCase):
    def test_monthly(self):
        rets = np.array([9.0, -6.0, 3.0])
        mpos = np.array([0, 0, 2])
        wins = np.array([3, 3, 7])
        month_win = np.array([3, 3, 7, 7])
        r = E.metrics(rets, mpos, wins, slots=3, n_months=4, month_win=month_win)
        self.assertAlmostEqual(r["avg"], (3.0 - 2.0 + 1.0) / 4)                        # 1件に資金の1/3、取引の無い月も数える
        self.assertAlmostEqual(r["avg_h1"], (1.0 + 0.0) / 2)
        self.assertAlmostEqual(r["avg_h2"], (1.0 + 0.0) / 2)
        self.assertEqual((r["n_h1"], r["n_h2"], r["zero_months"]), (2, 1, 2))
        r2 = E.metrics(np.array([12.0, -30.0, 6.0]), np.array([0, 1, 2]), np.array([3, 3, 7]), slots=1, n_months=3,
                       month_win=np.array([3, 3, 7]))
        self.assertAlmostEqual(r2["max_dd"], 30.0)                                     # 12 → −18 の落ち込み
        self.assertEqual(r2["ge_goal"], 1)
        self.assertAlmostEqual(r2["worst"], -30.0)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        parts = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
        cls.ft = E.build(cls.df, parts)
        cls.ft83 = E83.build(cls.df, parts)
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))
        cls.grid = dict(sets=[s for s in E.SETS if s[0] in ("L1", "T3")], lines=(95.0,), breaks=(1, 8), topks=(2, 3),
                        slots_list=(3, 5), caps=(0, 2), targets=(10.0, 20.0), h1=(3, 4), h2=(5, 6))
        cls.res = E.search(cls.ft, cls.bar_days, **cls.grid)

    def test_matches_reference_functions(self):
        c = E.check_against(self.ft, self.ft83, self.bar_days)
        self.assertTrue(c["screen"]["same"], c["screen"])
        self.assertTrue(c["month2"]["same"], c["month2"])
        self.assertGreater(c["screen"]["n_fast"], 0)

    def test_grid_and_monthly_return(self):
        self.assertEqual(len(self.res), 2 * 1 * 2 * 2 * 2 * 2 * 2)
        r = self.res[(self.res["set"] == "T3") & (self.res["breaks"] == 8) & (self.res["topk"] == 2)
                     & (self.res["slots"] == 3) & (self.res["cap"] == 0) & (self.res["target"] == 10.0)].iloc[0]
        tr = E.run_one(self.ft, self.bar_days, "T3", 95.0, 8, 2, 3, 0, 10.0)
        months = len(E.months_index(self.ft)[0])
        self.assertEqual(int(r["n"]), len(tr))
        self.assertAlmostEqual(r["avg"], tr["tp10c"].sum() / 3 / months)
        self.assertEqual(int(r["n_h1"]) + int(r["n_h2"]), int(r["n"]))
        self.assertTrue((self.res["n"] <= self.res["signals"]).all())
        capped = self.res[self.res["cap"] == 2]
        self.assertTrue((capped["n"] <= 2 * months).all())

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        months = len(E.months_index(self.ft)[0])
        with contextlib.redirect_stdout(buf):
            s = E.report(self.res, months, checks=E.check_against(self.ft, self.ft83, self.bar_days),
                         res_full=self.res, n_months_full=months)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ まとめ", "確かめ", "一致"):
            self.assertIn(tag, out)
        for k in ("top", "grid", "base", "spread"):
            self.assertIn(k, s)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
