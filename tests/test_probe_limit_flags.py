#!/usr/bin/env python3
"""
ストップ高の自前の判定と API のフラグの突き合わせ（research/probe_limit_flags.py）。API は叩かず、応答の形を合成して数え方を確かめる。

- 標本の日の選び方: 暦日が休みなら手前の営業日。直近 N 営業日も足し、重なりは1つ
- 突き合わせ: 一致・API だけ・自前だけ の件数と内訳（売買の無い日・上限に届かず）。銘柄コードを出さない

  python3 tests/test_probe_limit_flags.py
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import probe_limit_flags as P  # noqa: E402


def day_frame(day: str, rows):
    """rows: (Code, H, L, C, UL, LL)。AdjC は C と同じ。"""
    return pd.DataFrame([{"Date": pd.Timestamp(day), "Code": c, "O": h, "H": h, "L": lo, "C": cl, "AdjC": cl, "UL": ul, "LL": ll}
                         for c, h, lo, cl, ul, ll in rows])


class TestPickDays(unittest.TestCase):
    def test_prior_trading_day_and_recent(self):
        days = [dt.date(2024, 5, d) for d in (13, 14, 15, 16, 17, 20, 21, 22)]
        pairs = P.pick_days(days, ["2024-05-18", "2024-05-15"], recent=2)
        self.assertEqual(pairs, [(dt.date(2024, 5, 17), dt.date(2024, 5, 16)), (dt.date(2024, 5, 15), dt.date(2024, 5, 14)),
                                 (dt.date(2024, 5, 21), dt.date(2024, 5, 20)), (dt.date(2024, 5, 22), dt.date(2024, 5, 21))])
        self.assertEqual(P.pick_days(days, ["2024-05-13"], recent=0), [])      # 前営業日が無い日は外す


class TestCompare(unittest.TestCase):
    def test_counts(self):
        prev = day_frame("2024-05-14", [("10010", 1000, 990, 1000, "0", "0"), ("20020", 500, 495, 500, "0", "0"),
                                        ("30030", 2000, 1990, 2000, "0", "0"), ("40040", 800, 790, 800, "0", "0"),
                                        ("50050", 300, 295, 300, "0", "0")])
        today = day_frame("2024-05-15", [("10010", 1300, 1200, 1300, "1", "0"),      # 一致（ちょうど上限）
                                         ("20020", 590, 560, 580, "1", "0"),          # API だけ（上限 600 に届かず。幅の 10% 手前）
                                         ("30030", 2500, 2450, 2500, "0", "0"),       # 自前だけ（上限 2,500。API は 0）
                                         ("40040", np.nan, np.nan, np.nan, "1", "0"),  # API だけ（売買が無い）
                                         ("50050", 300, 220, 220, "0", "1")])         # ストップ安が一致（幅 80）
        r = P.compare_day(today, prev)
        self.assertEqual((r["rows"], r["known"]), (5, 4))
        self.assertEqual((r["ul_api"], r["ul_ours"], r["ul_both"], r["ul_api_only"], r["ul_ours_only"]), (3, 2, 1, 2, 1))
        self.assertEqual((r["ul_api_only_no_trade"], r["ul_api_only_within_2pct"], r["ul_api_only_within_10pct"],
                          r["ul_api_only_farther"]), (1, 0, 1, 0))
        self.assertEqual(r["ul_ours_only_le_1_05"], 1)
        self.assertEqual((r["ll_api"], r["ll_ours"], r["ll_both"]), (1, 1, 1))
        lines = P.report([(dt.date(2024, 5, 15), r)])
        text = "\n".join(lines)
        self.assertIn("合計", text)
        self.assertIn("ストップ高: API のフラグ 3件のうち自前も1 1件", text)
        for code in ("10010", "20020", "30030", "40040", "50050"):
            self.assertNotIn(code, text)

    def test_missing_flag_column(self):
        prev = day_frame("2024-05-14", [("10010", 1000, 990, 1000, "0", "0")]).drop(columns=["UL", "LL"])
        today = day_frame("2024-05-15", [("10010", 1300, 1200, 1300, "0", "0")]).drop(columns=["UL", "LL"])
        r = P.compare_day(today, prev)
        self.assertEqual(r["api_ul_na"], 1)
        self.assertEqual((r["ul_api"], r["ul_ours"], r["ul_ours_only"]), (0, 1, 0))     # フラグが読めない行は「自前だけ」に数えない


if __name__ == "__main__":
    unittest.main()
