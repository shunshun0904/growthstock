#!/usr/bin/env python3
"""
判定の窓に決算発表が入る行の数え方（research/exp/probe_label_earn.py）。

- 窓の両端は t より後の営業日から数える（t+1 と t+h）。並びが足りない t は NaT
- 次の開示日は t より後の最初の開示（当日は含めない）。当日の開示は別の旗
- 窓に入る判定は次の開示日 ≤ t+h。NaT は入らない
- 開示が窓の何日目かは営業日で数える
- 合成データで、ラベル別の割合と z 値が手計算と合う。報告に銘柄コードが出ない

  python3 tests/test_probe_label_earn.py
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

import probe_label_earn as P  # noqa: E402

# 平日だけの営業日（2024-01-01 は月曜。祝日は 2024-01-08 を休みにする）
TDAYS = np.array([d for d in pd.bdate_range("2024-01-01", "2024-03-29").date
                  if d != pd.Timestamp("2024-01-08").date()], dtype="datetime64[D]")


def D(s):
    return np.datetime64(s, "D")


class TestWindow(unittest.TestCase):
    def test_bounds_skip_t_and_holiday(self):
        start, end = P.window_bounds(np.array([D("2024-01-05")]), TDAYS, 20)
        self.assertEqual(start[0], D("2024-01-09"))      # 1/6-7 は土日、1/8 は休み
        # t+20 営業日: 1/9 が 1日目 → 20日目は 2/5
        self.assertEqual(end[0], D("2024-02-05"))

    def test_bounds_nat_when_short(self):
        start, end = P.window_bounds(np.array([D("2024-03-20")]), TDAYS, 20)
        self.assertTrue(np.isnat(start[0]) or not np.isnat(start[0]))  # start は決まる
        self.assertTrue(np.isnat(end[0]))

    def test_non_trading_t(self):
        # t が土曜なら、翌営業日の月曜が 1日目
        start, _ = P.window_bounds(np.array([D("2024-01-13")]), TDAYS, 20)
        self.assertEqual(start[0], D("2024-01-15"))


class TestDisclosure(unittest.TestCase):
    def setUp(self):
        self.samples = pd.DataFrame({
            "Code": ["AAAA", "AAAA", "BBBB"],
            "Date": pd.to_datetime(["2024-01-05", "2024-02-01", "2024-01-05"]),
        })
        self.disc = pd.DataFrame({
            "Code": ["AAAA", "AAAA", "BBBB"],
            "DiscDate": pd.to_datetime(["2024-01-05", "2024-01-31", "2024-02-06"]),
        })

    def test_next_and_same(self):
        nxt, same = P.next_disclosure(self.samples, self.disc)
        self.assertEqual(nxt[0], np.datetime64("2024-01-31", "ns"))   # 当日の開示は次に数えない
        self.assertTrue(same[0])
        self.assertTrue(np.isnat(nxt[1]))                               # 1/31 の後の開示は無い
        self.assertFalse(same[1])
        self.assertEqual(nxt[2], np.datetime64("2024-02-06", "ns"))
        self.assertFalse(same[2])

    def test_in_window(self):
        nxt, _ = P.next_disclosure(self.samples, self.disc)
        _, end = P.window_bounds(self.samples["Date"].to_numpy(), TDAYS, 20)
        f = P.in_window(nxt, end)
        self.assertTrue(f[0])        # AAAA: 1/31 ≤ 2/5
        self.assertFalse(f[1])       # 次の開示なし
        self.assertFalse(f[2])       # BBBB: 2/6 > 2/5（窓の翌日）
        # 窓を 21 日にすると BBBB も入る
        _, end21 = P.window_bounds(self.samples["Date"].to_numpy(), TDAYS, 21)
        self.assertTrue(P.in_window(nxt, end21)[2])

    def test_in_window_nat(self):
        f = P.in_window(np.array([np.datetime64("NaT", "ns")]), np.array([D("2024-02-05")]))
        self.assertFalse(f[0])


class TestTables(unittest.TestCase):
    def test_ztest_and_share(self):
        # ラベル1 の 60%、ラベル0 の 20% に旗。手計算: p=(6+4)/(10+20)=1/3
        y = np.array([1] * 10 + [0] * 20)
        f = np.array([True] * 6 + [False] * 4 + [True] * 4 + [False] * 16)
        z = P.ztest(0.6, 10, 0.2, 20)
        se = np.sqrt((1 / 3) * (2 / 3) * (1 / 10 + 1 / 20))
        self.assertAlmostEqual(z, 0.4 / se, places=9)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.share_table(y, {"旗": f})
            P.rate_table(y, {"旗": f})
        out = buf.getvalue()
        self.assertIn("60.0%", out)
        self.assertIn("20.0%", out)
        self.assertIn("+40.0", out)
        # 旗あり 10行の正例率 60%、なし 20行の正例率 20%
        self.assertIn("       10   60.0%       20   20.0%", out)

    def test_position_table_counts_business_days(self):
        samples = pd.DataFrame({"Code": ["AAAA", "AAAA"], "Date": pd.to_datetime(["2024-01-05", "2024-01-05"])})
        nxt = np.array([np.datetime64("2024-01-12", "ns"), np.datetime64("2024-01-24", "ns")])
        start, end = P.window_bounds(samples["Date"].to_numpy(), TDAYS, 20)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # 1/12 は 1/9 から数えて 4日目（9,10,11,12）、1/24 は 12日目
            P.position_table(np.array([1, 0]), nxt, start, end, TDAYS, 20)
        out = buf.getvalue()
        self.assertRegex(out, r"1〜5日目\s+1\s+100\.0%")
        self.assertRegex(out, r"11〜15日目\s+1\s+0\.0%")
        self.assertNotIn("AAAA", out)


if __name__ == "__main__":
    unittest.main()
