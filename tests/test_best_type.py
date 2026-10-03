#!/usr/bin/env python3
"""
最も当たる型（research/best_type.py）の判定のテスト。

- 同じ日の候補の中の百分位は (順位 − 0.5) ÷ 件数
- 4条件は中央値より望ましい側なら ○、中央値ちょうどは ×
- 候補が3件未満の日は判定しない（None）、値が無い条件は None で数えない
- 予測の payload（predict_daily）に bestType として載る形

  python3 tests/test_best_type.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import best_type as BT  # noqa: E402


def day(date, rows):
    """rows = [(log_trading_value, r_high, close_position, book_yield), ...]"""
    return pd.DataFrame([{"Date": pd.Timestamp(date), "Code": f"{1000 + i}0", "log_trading_value": a,
                          "r_high": b, "close_position": c, "book_yield": d}
                         for i, (a, b, c, d) in enumerate(rows)])


class Profiles(unittest.TestCase):
    def test_percentile_is_centered(self):
        df = day("2026-10-03", [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4)])
        p = BT.within_day_pct(df, ["r_high"])["r_high"].tolist()
        self.assertEqual(p, [0.125, 0.375, 0.625, 0.875])
        self.assertAlmostEqual(float(np.mean(p)), 0.5)

    def test_conditions_against_the_day_median(self):
        # 5件。売買代金は小さいほど○、他は大きいほど○。中央値ちょうど（3番目）は×
        df = day("2026-10-03", [(10, 0.90, 0.9, 1.0), (20, 0.95, 0.8, 0.8), (30, 0.99, 0.5, 0.6),
                                (40, 0.80, 0.2, 0.4), (50, 0.70, 0.1, 0.2)])
        out = BT.profiles(df)
        self.assertEqual([o["small"] for o in out], [True, True, False, False, False])
        self.assertEqual([o["nearHigh"] for o in out], [False, True, True, False, False])
        self.assertEqual([o["closeHigh"] for o in out], [True, True, False, False, False])
        self.assertEqual([o["lowPbr"] for o in out], [True, True, False, False, False])
        self.assertEqual([o["n"] for o in out], [3, 4, 1, 0, 0])
        self.assertEqual(out[0]["nInDay"], 5)
        self.assertEqual(out[0]["total"], 4)
        self.assertEqual(out[1]["pct"]["small"], 0.3)        # 2番目に小さい → (2 − 0.5) / 5

    def test_too_few_candidates_is_undecided(self):
        df = day("2026-10-03", [(10, 0.9, 0.9, 1.0), (20, 0.8, 0.8, 0.8)])
        out = BT.profiles(df)
        self.assertEqual([o["n"] for o in out], [None, None])
        self.assertTrue(all(o[k] is None for o in out for k in BT.KEYS))
        self.assertEqual(out[0]["pct"]["small"], 0.25)       # 百分位そのものは出す

    def test_missing_values_are_not_counted(self):
        df = day("2026-10-03", [(10, np.nan, 0.9, 1.0), (20, 0.8, 0.8, 0.8), (30, 0.7, 0.5, 0.6)])
        out = BT.profiles(df)
        self.assertIsNone(out[0]["nearHigh"])
        self.assertEqual(out[0]["n"], 3)                     # 残り3条件は○
        self.assertIsNone(out[0]["pct"]["nearHigh"])
        # 列そのものが無ければ、その条件は全部 None
        out2 = BT.profiles(df.drop(columns=["book_yield"]))
        self.assertTrue(all(o["lowPbr"] is None for o in out2))
        self.assertEqual(out2[1]["n"], 1)                    # small 中央値(×) nearHigh(○) closeHigh 中央値(×) → 1

    def test_days_are_judged_separately(self):
        a = day("2026-10-02", [(10, 0.9, 0.9, 1.0), (20, 0.8, 0.8, 0.8), (30, 0.7, 0.5, 0.6)])
        b = day("2026-10-03", [(100, 0.5, 0.1, 0.1), (200, 0.4, 0.2, 0.2), (300, 0.3, 0.3, 0.3)])
        out = BT.profiles(pd.concat([a, b], ignore_index=True))
        self.assertEqual(out[0]["n"], 4)                     # 10/2 の1番: 全部○
        self.assertEqual(out[3]["n"], 2)                     # 10/3 の1番: small ○, nearHigh ○, 他 ×
        self.assertEqual(out[5]["n"], 2)                     # 10/3 の3番: closeHigh ○, lowPbr ○


if __name__ == "__main__":
    unittest.main(verbosity=2)
