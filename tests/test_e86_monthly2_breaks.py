#!/usr/bin/env python3
"""
実験86（research/exp/e86_monthly2_breaks.py）。

- 発火 = その日の行数、発火8件以上の印
- 主: R0 の先着2件を発火8件以上の日だけで（発火8件未満の日の R0 は飛ばして、次の日へ）
- 印を巡回してずらしても、窓ごとの印の付く日の数は変わらない。ずらす幅 0 なら元のまま
- 判定3（印をずらした絞り込み）の p は (0, 1]、同じ種なら同じ値
- 報告に銘柄コードが出ない

  python3 tests/test_e86_monthly2_breaks.py
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
import e86_monthly2_breaks as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


def mini(days):
    """days: [(日付, 行数, [(R0, pmin3, score), ...])]。R0 でない行で行数を埋める。"""
    out, i = [], 0
    for d, n, r0s in days:
        rows = list(r0s) + [(False, 50.0, 0.0)] * (n - len(r0s))
        for r0, p, s in rows:
            out.append({"Code": f"{1000 + i}0", "Date": pd.Timestamp(d), "window": 3, "R0": r0, "A15": False,
                        "pmin3": p, "avg6": p, "score": s, "tp10c": 1.0, "hit10": False})
            i += 1
    return E.add_breaks(pd.DataFrame(out))


class TestBreaks(unittest.TestCase):
    def setUp(self):
        self.ft = mini([("2024-03-04", 3, [(True, 99.0, 0.9)]),                         # 静かな日（3件）
                        ("2024-03-11", 9, [(True, 96.0, 0.1), (True, 98.0, 0.2)]),     # 発火9件
                        ("2024-03-18", 10, [(True, 97.0, 0.5)]),
                        ("2024-04-01", 8, [(True, 95.0, 0.3)])])                       # ちょうど8件

    def test_counts_and_flag(self):
        per_day = self.ft.groupby("Date")["n_break"].first()
        self.assertEqual(list(per_day), [3, 9, 10, 8])
        self.assertEqual(list(self.ft.groupby("Date")["hot"].first()), [False, True, True, True])

    def test_first_hot_skips_quiet_days(self):
        cur = E.first_by(self.ft, self.ft["R0"].to_numpy(dtype=bool), "pmin3")
        self.assertEqual(list(cur["Date"].dt.day), [4, 11, 1])
        self.assertEqual(list(cur["pmin3"]), [99.0, 98.0, 95.0])
        main = E.first_hot(self.ft)
        self.assertEqual(list(main["Date"].dt.day), [11, 11, 1])                       # 3月は11日の2件
        self.assertEqual(list(main["pmin3"]), [98.0, 96.0, 95.0])
        self.assertTrue(main["hot"].all())

    def test_rotate(self):
        idx = E.day_index(self.ft)
        same = E.rotate_hot(len(self.ft), idx, {3: 0})
        np.testing.assert_array_equal(same, self.ft["hot"].to_numpy())
        for s in range(1, 4):
            h = E.rotate_hot(len(self.ft), idx, {3: s})
            per_day = pd.Series(h).groupby(self.ft["Date"].to_numpy()).agg(["min", "max"])
            self.assertTrue((per_day["min"] == per_day["max"]).all())                  # 同じ日の行は同じ印
            self.assertEqual(int(per_day["max"].sum()), 3)                            # 印の付く日の数はそのまま
        h1 = E.rotate_hot(len(self.ft), idx, {3: 1})
        self.assertEqual(list(pd.Series(h1).groupby(self.ft["Date"].to_numpy()).first()), [True, False, True, True])


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        cls.ft = E.add_breaks(E.E85.build(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]))
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_synth_has_both_kinds_of_days(self):
        per_day = self.ft.groupby("Date")["hot"].first()
        self.assertTrue(per_day.any() and (~per_day).any())

    def test_shift_test(self):
        cur = E.first_by(self.ft, self.ft["R0"].to_numpy(dtype=bool), "pmin3")
        a = E.shift_test(self.ft, cur, draws=40, seed=1)
        b = E.shift_test(self.ft, cur, draws=40, seed=1)
        self.assertGreater(a["p"], 0.0)
        self.assertLessEqual(a["p"], 1.0)
        self.assertEqual(a["p"], b["p"])
        self.assertEqual(a["draws"], 40)

    def test_break_check(self):
        counts = pd.to_datetime(self.ft["Date"]).value_counts()
        c = E.break_check(self.ft, counts)
        self.assertEqual(c["same"], c["days"])
        self.assertEqual(c["flip"], 0)
        c2 = E.break_check(self.ft, counts + 100)
        self.assertEqual(c2["same"], 0)

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        counts = pd.to_datetime(self.ft["Date"]).value_counts()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, bar_days=self.bar_days, ff=self.ft, draws=30,
                         check=E.break_check(self.ft, counts))
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ まとめ", "判定", "今のやり方", "分位点"):
            self.assertIn(tag, out)
        for k in ("main_vs_cur", "shift", "passed", "rows", "other", "daily", "grid", "full"):
            self.assertIn(k, s)
        self.assertLessEqual(s["main"]["n"], s["cur"]["n"])                            # 月ごとに min(2, 発火8件以上の R0) ≤ min(2, R0)
        self.assertEqual(s["main"]["quiet_n"], 0)                                      # 主は発火8件未満の日に買わない
        self.assertEqual(s["grid"][0]["n"], s["cur"]["n"])                             # 線 1 = 絞らない = 今のやり方
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
