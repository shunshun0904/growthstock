#!/usr/bin/env python3
"""
窓の値動き・予想修正の向き・決算までの日数の区分（research/exp/probe_earn_direction.py）。

- 窓の最大・最小・最後は t+1〜t+h の終値、基準は翌営業日の寄り。窓が足りない行は NaN
- 開示日をまたぐ2日の値動きは「開示日の前の営業日の終値 → 次の営業日の終値」
- 予想修正の向きは同じ事業年度の直前の開示比。新年度の最初の予想・予想なしは「比べられない」。同じ日は最後の1件
- 決算までの日数の区分は境界を含む側が決まっている（7日 → 0〜7日、8日 → 8〜14日、NaN → 予定不明）
- 報告に銘柄コードが出ない

  python3 tests/test_probe_earn_direction.py
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

import probe_earn_direction as D  # noqa: E402


def bars_one(code: str, n: int = 30, start: str = "2024-01-01"):
    dates = pd.bdate_range(start, periods=n)
    close = np.array([100.0 + i for i in range(n)])          # 100, 101, ..., 129
    return pd.DataFrame({"Date": dates, "Code": code, "AdjO": close - 0.5, "AdjH": close + 1.0, "AdjC": close})


class TestWindowStats(unittest.TestCase):
    def test_values(self):
        b = bars_one("AAAA")
        keys = pd.DataFrame({"Code": ["AAAA", "AAAA"], "Date": [b["Date"].iloc[2], b["Date"].iloc[20]]})
        react = np.array([b["Date"].iloc[6].to_datetime64(), np.datetime64("NaT")], dtype="datetime64[ns]")
        ws = D.window_stats(b, keys, 5, react)
        # t = 3日目(終値102): entry = 4日目の寄り 102.5、窓は 4〜8日目の終値 103..107
        self.assertAlmostEqual(ws["entry"].iloc[0], 102.5)
        self.assertAlmostEqual(ws["w_max"].iloc[0], 107.0)
        self.assertAlmostEqual(ws["w_min"].iloc[0], 103.0)
        self.assertAlmostEqual(ws["w_end"].iloc[0], 107.0)
        self.assertAlmostEqual(ws["rise"].iloc[0], 107.0 / 102.5 - 1.0)
        self.assertAlmostEqual(ws["fall"].iloc[0], 103.0 / 102.5 - 1.0)
        # 開示日 = 7日目(終値106): 6日目の終値 105 → 8日目の終値 107
        self.assertAlmostEqual(ws["react"].iloc[0], 107.0 / 105.0 - 1.0)
        # t = 21日目: 窓 22〜26日目は足りる（30日ある）が、t = 26日目以降は足りない
        self.assertTrue(np.isfinite(ws["ret"].iloc[1]))
        self.assertTrue(np.isnan(ws["react"].iloc[1]))
        keys2 = pd.DataFrame({"Code": ["AAAA"], "Date": [b["Date"].iloc[26]]})
        self.assertTrue(np.isnan(D.window_stats(b, keys2, 5)["ret"].iloc[0]))

    def test_unknown_code_or_date(self):
        b = bars_one("AAAA")
        keys = pd.DataFrame({"Code": ["BBBB", "AAAA"], "Date": [b["Date"].iloc[2], pd.Timestamp("2024-01-06")]})
        ws = D.window_stats(b, keys, 5)
        self.assertTrue(ws["ret"].isna().all())


class TestDisclosureTable(unittest.TestCase):
    def test_revision_direction(self):
        fins = pd.DataFrame({
            "Code": ["AAAA"] * 5,
            "DiscDate": ["2024-05-10", "2024-08-09", "2024-08-09", "2024-11-08", "2025-05-12"],
            "DiscTime": ["15:00", "15:00", "16:00", "15:00", "15:00"],
            "DiscNo": ["1", "1", "2", "1", "1"],
            "CurFYSt": ["2024-04-01"] * 4 + ["2025-04-01"],
            "CurPerType": ["FY", "1Q", "1Q", "2Q", "FY"],
            "Sales": [100, 30, 31, 60, 120],
            "NP": [10, 3, 3, 6, 12],
            "FOP": [20, 22, 21, 21, 25],
        })
        d = D.disclosure_table(fins)
        self.assertEqual(len(d), 4)                             # 同じ日の2件は最後の1件
        cats = dict(zip(d["DiscDate"].dt.strftime("%Y-%m-%d"), d["rev_cat"]))
        self.assertEqual(cats["2024-05-10"], "比べられない")      # 年度の最初の予想
        # 同じ日の2件目は1件目（22）との比（forecast_revisions と同じ並び）: 22 → 21 で下方
        self.assertEqual(cats["2024-08-09"], "下方")
        self.assertEqual(cats["2024-11-08"], "据え置き")
        self.assertEqual(cats["2025-05-12"], "比べられない")      # 新年度

    def test_same_day_last_is_used_but_prev_is_in_order(self):
        # 同じ日の1件目 → 2件目の順で直前比を取るので、2件目は「1件目との比」になる（下方）
        fins = pd.DataFrame({
            "Code": ["AAAA"] * 3,
            "DiscDate": ["2024-05-10", "2024-08-09", "2024-08-09"],
            "DiscTime": ["15:00", "15:00", "16:00"], "DiscNo": ["1", "1", "2"],
            "CurFYSt": ["2024-04-01"] * 3, "CurPerType": ["FY", "1Q", "1Q"],
            "Sales": [1, 1, 1], "NP": [1, 1, 1], "FOP": [20, 30, 25],
        })
        d = D.disclosure_table(fins)
        self.assertEqual(d["rev_cat"].iloc[-1], "下方")


class TestBins(unittest.TestCase):
    def test_edges(self):
        b = D.ahead_bin(np.array([0, 7, 8, 14, 15, 28, 29, 60, 61, 120, np.nan]))
        self.assertEqual(list(b), ["0〜7日", "0〜7日", "8〜14日", "8〜14日", "15〜28日（窓の中）", "15〜28日（窓の中）",
                                   "29〜60日", "29〜60日", "61日〜", "61日〜", "予定不明"])


class TestTables(unittest.TestCase):
    def test_direction_table_no_codes(self):
        ws = pd.DataFrame({"ret": [0.1, -0.1, 0.05, np.nan], "rise": [0.2, 0.0, 0.1, np.nan],
                           "fall": [-0.05, -0.3, 0.0, np.nan], "entry": [1, 1, 1, np.nan]})
        flag = np.array([True, False, True, False])
        need = {"1.2σ": np.array([0.15, 0.15, 0.15, 0.15])}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            D.print_direction(flag, ws, need, "t")
        out = buf.getvalue()
        self.assertIn("決算が窓に入る", out)
        self.assertRegex(out, r"決算が窓に入る\s+2\s+50\.0%\s+0\.0%")   # 上に抜け 1/2、下に抜け 0/2
        self.assertRegex(out, r"入らない\s+1\s+0\.0%\s+100\.0%")


if __name__ == "__main__":
    unittest.main()
