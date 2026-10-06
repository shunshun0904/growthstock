#!/usr/bin/env python3
"""
実験64（research/exp/e64_rev_order.py）のテスト。重い計算は回さない。

- 台本に写した旧版 forecast_revisions_old は、同じ日の修正が入力の順に並んでいれば直した版と同じ値、
  入力の順が逆なら（旧版だけ）値が変わる＝旧版の挙動を写せている
- 値の比較（欠測どうしは同じ）・差の集計・保存済みが無いときに計算しないこと

  python3 tests/test_e64_rev_order.py
"""
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e64_rev_order as E  # noqa: E402

warnings.filterwarnings("ignore")
D = pd.Timestamp


def fins():
    """1Q の予想 → 同じ日に修正2件（時刻の順に並べてある）→ 配当の修正。2銘柄。"""
    rows = []
    for i, code in enumerate(["13010", "13020"]):
        rows += [
            {"Code": code, "DiscDate": D("2024-08-05"), "DiscTime": "15:00:00", "DiscNo": f"1{i}01",
             "DocType": "1QFinancialStatements_Consolidated_JP", "CurFYSt": D("2024-04-01"), "FOP": 1000.0},
            {"Code": code, "DiscDate": D("2024-09-20"), "DiscTime": "15:00:00", "DiscNo": f"1{i}02",
             "DocType": "EarnForecastRevision", "CurFYSt": D("2024-04-01"), "FOP": 1200.0},
            {"Code": code, "DiscDate": D("2024-09-20"), "DiscTime": "16:00:00", "DiscNo": f"1{i}03",
             "DocType": "EarnForecastRevision", "CurFYSt": D("2024-04-01"), "FOP": 900.0 + 50 * i},
            {"Code": code, "DiscDate": D("2024-10-01"), "DiscTime": "15:00:00", "DiscNo": f"1{i}04",
             "DocType": "DividendForecastRevision", "CurFYSt": D("2024-04-01"), "FOP": np.nan},
        ]
    return pd.DataFrame(rows)


def samples():
    return pd.DataFrame({"Code": ["13010", "13020"] * 3,
                         "Date": [D("2024-09-19")] * 2 + [D("2024-09-30")] * 2 + [D("2025-03-01")] * 2})


class OldCopy(unittest.TestCase):
    def test_same_as_fixed_when_input_is_in_time_order(self):
        old = E.forecast_revisions_old(samples(), fins())
        new = B.forecast_revisions(samples(), fins())
        pd.testing.assert_frame_equal(old, new)

    def test_old_depends_on_input_order_fixed_does_not(self):
        rev = fins().iloc[::-1].reset_index(drop=True)
        new = B.forecast_revisions(samples(), fins())
        pd.testing.assert_frame_equal(B.forecast_revisions(samples(), rev), new)
        old_rev = E.forecast_revisions_old(samples(), rev)
        self.assertFalse(np.allclose(old_rev["rev_pct"].to_numpy(dtype=float),
                                     new["rev_pct"].to_numpy(dtype=float), equal_nan=True))
        # 直した版: 9/30 の行は 16:00 の修正（1,200 との比）: 900 → −25%、950 → −20.8%
        got = new.loc[samples()["Date"] == D("2024-09-30"), "rev_pct"].round(2).tolist()
        self.assertEqual(got, [-25.0, -20.83])


class Helpers(unittest.TestCase):
    def test_same_treats_nan_as_equal(self):
        a = pd.Series([1.0, np.nan, 2.0, np.nan])
        b = pd.Series([1.0, np.nan, 2.5, 3.0])
        self.assertEqual(E.same(a, b).tolist(), [True, True, False, False])
        s = pd.Series(["x", None, "y"])
        t = pd.Series(["x", None, "z"])
        self.assertEqual(E.same(s, t).tolist(), [True, True, False])

    def test_pair_counts_wins_and_ties(self):
        line, st = E.pair("x", np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.2, 0.1]))
        self.assertEqual((st["wins"], st["ties"], st["n"]), (1, 1, 3))
        self.assertAlmostEqual(st["mean_diff"], (0.1 + 0.0 - 0.2) / 3)
        self.assertIn("1/3", line)

    def test_nothing_is_computed_when_not_asked(self):
        old = E.OOF_DIR
        with tempfile.TemporaryDirectory() as d:
            E.OOF_DIR = d
            try:
                fr = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=10), "label": 0.0})
                stamp = {"built_utc": "x"}
                rec = E.tune("logit", "A", fr, ["a"], D("2020-01-10"), 50, stamp, compute=False)
                self.assertIsNone(rec)
                self.assertIsNone(E.oof("logit", "A", fr, ["a"], {"C": 1.0}, 0, (42,), compute=False))  # データ A
                self.assertIsNone(E.run("logit", {"A": fr, "B": fr}, ["a"], D("2020-01-10"), 50, stamp,
                                        [0], compute=False))
                self.assertEqual(os.listdir(d), [])
            finally:
                E.OOF_DIR = old

    def test_short_drops_fixed_keys(self):
        s = E.short({"objective": "binary", "learning_rate": 0.0234567, "num_leaves": 31, "n_jobs": -1})
        self.assertEqual(s, "{learning_rate 0.02346, num_leaves 31}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
