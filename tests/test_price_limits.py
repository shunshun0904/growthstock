#!/usr/bin/env python3
"""
値幅制限からのストップ高・ストップ安の判定（research/price_limits.py）。

- 制限値幅の表: 帯の境（以上・未満）で正しい幅になる
- 判定: 高値がちょうど上限の日・上限を超える日は ul、終値が上限なら張り付き、安値が下限なら ll。
  売買の無い日を挟めば最後の終値が基準値段。分割の日は前日の終値を倍率で直す。判定できない行は False・known=False
- 期間の集計: 翌営業日から20営業日目までの有無・日数・最初の日、そろわない行は NaN、前250営業日の回数

  python3 tests/test_price_limits.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import price_limits as PL  # noqa: E402


def bars_of(code: str, closes, highs=None, lows=None, adj_mult=1.0, start="2024-01-04"):
    """終値の列から日足を作る。highs / lows を省くと終値と同じ。adj_mult は AdjC ÷ C（分割の確かめ用）。"""
    n = len(closes)
    d = pd.bdate_range(start, periods=n)
    c = np.array(closes, dtype=float)
    h = c if highs is None else np.array(highs, dtype=float)
    lo = c if lows is None else np.array(lows, dtype=float)
    mult = np.full(n, adj_mult, dtype=float) if np.isscalar(adj_mult) else np.array(adj_mult, dtype=float)
    return pd.DataFrame({"Date": d, "Code": code, "H": h, "L": lo, "C": c, "AdjC": c * mult})


class TestWidth(unittest.TestCase):
    def test_table_edges(self):
        cases = {99.9: 30, 100: 50, 199.9: 50, 200: 80, 499: 80, 500: 100, 699: 100, 700: 150, 999.9: 150,
                 1000: 300, 1499: 300, 1500: 400, 1999: 400, 2000: 500, 2999: 500, 3000: 700, 4999: 700,
                 5000: 1000, 6999: 1000, 7000: 1500, 9999: 1500, 10000: 3000, 14999: 3000, 15000: 4000,
                 20000: 5000, 30000: 7000, 50000: 10000, 70000: 15000, 100000: 30000, 150000: 40000,
                 200000: 50000, 300000: 70000, 500000: 100000, 700000: 150000, 1000000: 300000,
                 50000000: 10000000, 123456789: 10000000}
        got = PL.width(list(cases))
        for (base, want), g in zip(cases.items(), got):
            self.assertEqual(g, want, base)
        self.assertTrue(np.isnan(PL.width([np.nan])[0]))
        self.assertEqual(PL.width(np.array([]))
                         .shape, (0,))

    def test_table_is_increasing(self):
        ups = [u for u, _ in PL.TABLE]
        ws = [w for _, w in PL.TABLE]
        self.assertEqual(ups, sorted(ups))
        self.assertEqual(ws, sorted(ws))
        self.assertEqual(len(PL.TABLE), 34)


class TestFlagLimits(unittest.TestCase):
    def test_exact_over_close_and_low(self):
        # 基準 1,000 → 幅 300。2日目 高値 1,300（ちょうど）、3日目 基準 1,200・高値 1,520（超える。幅 300）、
        # 4日目 終値も上限（張り付き）、5日目 基準 1,700 → 幅 400 で安値が下限 1,300（ストップ安）、6日目 何も無い
        b = bars_of("10010", closes=[1000, 1200, 1400, 1700, 1500, 1500],
                    highs=[1010, 1300, 1520, 1700, 1700, 1550],
                    lows=[990, 1150, 1380, 1650, 1300, 1480])
        f = PL.flag_limits(b).set_index("Date")
        d = list(b["Date"])
        self.assertFalse(f.loc[d[0], "known"])                        # 上場初日（前日が無い）
        self.assertEqual(f.loc[d[1], "base"], 1000)
        self.assertEqual(f.loc[d[1], "width"], 300)
        self.assertTrue(f.loc[d[1], "ul"] and f.loc[d[1], "ul_exact"])
        self.assertFalse(f.loc[d[1], "ul_close"])
        self.assertTrue(f.loc[d[2], "ul"])                            # 1,520 ≥ 1,200 + 300
        self.assertFalse(f.loc[d[2], "ul_exact"])
        self.assertTrue(f.loc[d[3], "ul"] and f.loc[d[3], "ul_close"])   # 1,700 = 1,400 + 300 で引けた
        self.assertTrue(f.loc[d[4], "ll"])                            # 1,300 = 1,700 − 400
        self.assertFalse(f.loc[d[4], "ul"])
        self.assertFalse(f.loc[d[5], "ul"] or f.loc[d[5], "ll"])
        self.assertEqual(list(f.columns), ["Code", "base", "width", "known", "ul", "ul_close", "ll", "ul_exact"])

    def test_decimal_ticks(self):
        b = bars_of("13050", closes=[2920.5, 3420.5], highs=[2930.0, 3420.5])
        f = PL.flag_limits(b)
        self.assertTrue(bool(f.iloc[1]["ul"]) and bool(f.iloc[1]["ul_exact"]))

    def test_no_trade_day_uses_last_close(self):
        b = bars_of("20020", closes=[500, np.nan, 600, 700], highs=[505, np.nan, 600, 700])
        f = PL.flag_limits(b)
        self.assertFalse(bool(f.iloc[1]["known"]))                    # 売買が無い日は判定しない
        self.assertEqual(f.iloc[2]["base"], 500)                       # 最後の終値 500 → 幅 100 → 上限 600
        self.assertTrue(bool(f.iloc[2]["ul"]))
        self.assertEqual(f.iloc[3]["base"], 600)

    def test_split_day_adjusts_base(self):
        # 1:2 の分割。分割前の AdjC は C の半分、分割後は同じ。分割の日の基準値段は前日の終値 2,000 ÷ 2 = 1,000
        b = bars_of("30030", closes=[2000, 1000, 1300], highs=[2000, 1300, 1250], adj_mult=[0.5, 1.0, 1.0])
        f = PL.flag_limits(b)
        self.assertEqual(f.iloc[1]["base"], 1000)
        self.assertEqual(f.iloc[1]["width"], 300)
        self.assertTrue(bool(f.iloc[1]["ul"]))
        self.assertEqual(f.iloc[2]["base"], 1000)
        self.assertFalse(bool(f.iloc[2]["ul"]))

    def test_rounding_noise_is_not_a_split(self):
        # AdjC ÷ C の丸めの揺れ（1e-4）は分割とみなさない。基準値段は前日の終値そのまま
        b = bars_of("40040", closes=[1000, 1300], highs=[1000, 1300], adj_mult=[0.33329, 0.333333])
        f = PL.flag_limits(b)
        self.assertEqual(f.iloc[1]["base"], 1000)
        self.assertTrue(bool(f.iloc[1]["ul"]))

    def test_codes_are_independent_and_sorted(self):
        a = bars_of("10010", closes=[1000, 1300], highs=[1000, 1300])
        b = bars_of("20020", closes=[500, 500], highs=[500, 600])
        f = PL.flag_limits(pd.concat([b, a], ignore_index=True))
        self.assertEqual(list(f["Code"]), ["10010", "10010", "20020", "20020"])
        self.assertEqual(f["ul"].tolist(), [False, True, False, True])
        s = PL.flags_summary(f)
        self.assertEqual((s["rows"], s["known"], s["ul"], s["ul_exact"]), (4, 2, 2, 2))
        self.assertAlmostEqual(s["exact_among_ul"], 100.0)

    def test_missing_column(self):
        with self.assertRaises(KeyError):
            PL.flag_limits(pd.DataFrame({"Date": [], "Code": [], "H": []}))


class TestLimitEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 300
        closes = np.full(n, 1000.0)
        highs = closes.copy()
        lows = closes.copy()
        # 基準日 t = 260。t+3 と t+7 にストップ高（高値 1,300）、t+5 に張り付き、t+12 にストップ安。t−10・t−100 にもストップ高。t そのものも
        t = 260
        for k in (t + 3, t + 7, t - 10, t - 100, t):
            highs[k] = 1300
        highs[t + 5] = 1300
        closes[t + 5] = 1300          # 翌日の基準が 1,300 になる（幅 300）→ t+6 の高値 1,000 は何でもない
        lows[t + 12] = 700
        cls.bars = bars_of("50050", closes=closes, highs=highs, lows=lows)
        cls.flags = PL.flag_limits(cls.bars)
        cls.arrays = PL.code_arrays(cls.flags)
        cls.t = cls.bars["Date"].iloc[t]
        cls.late = cls.bars["Date"].iloc[n - 5]

    def test_forward_and_past(self):
        keys = pd.DataFrame({"Code": ["50050", "50050", "99990"], "Date": [self.t, self.late, self.t]})
        ev = PL.limit_events(self.arrays, keys, hold=20, past=250).set_index(["Code", "Date"])
        r = ev.loc[("50050", self.t)]
        self.assertEqual(r["ul_t"], 1.0)
        self.assertEqual(r["fwd_ok"], 1.0)
        self.assertEqual(r["ul_fwd"], 1.0)
        self.assertEqual(r["ul_fwd_n"], 3.0)                 # t+3, t+5, t+7
        self.assertEqual(r["ul_fwd_first"], 3.0)
        self.assertEqual(r["ulc_fwd"], 1.0)
        self.assertEqual(r["ulc_fwd_n"], 1.0)
        self.assertEqual(r["ll_fwd"], 1.0)
        self.assertEqual(r["fwd_known"], 20.0)
        self.assertEqual(r["ul_past_n"], 2.0)                # t−10, t−100（t 自身は入れない）
        self.assertEqual(r["past_bars"], 250.0)
        self.assertEqual(r["ul_past20_n"], 1.0)
        self.assertEqual(r["ul_days_since"], 10.0)            # 最後は t−10
        self.assertAlmostEqual(r["ul_decay"], 0.98 ** 10 + 0.98 ** 100)   # t−10 と t−100（t 自身は含めない）
        self.assertEqual((r["ul_past60_n"], r["ul_past120_n"]), (1.0, 2.0))
        late = ev.loc[("50050", self.late)]
        self.assertEqual(late["ul_days_since"], float(295 - 267))   # 最後は t+7 = 267
        self.assertEqual(late["fwd_ok"], 0.0)                # 20営業日がそろわない
        self.assertTrue(np.isnan(late["ul_fwd"]))
        self.assertEqual(late["ul_past_n"], 6.0)             # t−100, t−10, t, t+3, t+5, t+7 の6つ（どれも前250営業日の中）
        none = ev.loc[("99990", self.t)]
        self.assertTrue(np.isnan(none["ul_fwd"]) and np.isnan(none["ul_past_n"]) and np.isnan(none["ul_days_since"]))
        early = PL.limit_events(self.arrays, pd.DataFrame({"Code": ["50050"], "Date": [self.bars["Date"].iloc[100]]}),
                                hold=20, past=50).iloc[0]
        self.assertTrue(np.isnan(early["ul_days_since"]))      # 前50営業日にストップ高が無い
        self.assertEqual(early["ul_decay"], 0.0)
        self.assertEqual(PL.DECAY, 0.98)
        self.assertEqual(PL.EXTRA_WINDOWS, (60, 120))
        self.assertEqual(len(ev), 3)
        self.assertEqual(r["fwd_days"], 20.0)
        self.assertTrue(np.isnan(late["fwd_days"]))

    def test_partial_window(self):
        # 途中の行も数える（実運用用）。late は残り4営業日ぶんだけ。最後の日の行は 0日（数えない）
        keys = pd.DataFrame({"Code": ["50050", "50050", "50050"], "Date": [self.t, self.late, self.bars["Date"].iloc[-1]]})
        ev = PL.limit_events(self.arrays, keys, hold=20, past=250, partial=True).set_index(["Code", "Date"])
        r = ev.loc[("50050", self.t)]
        self.assertEqual((r["fwd_ok"], r["fwd_days"], r["ul_fwd_n"]), (1.0, 20.0, 3.0))
        late = ev.loc[("50050", self.late)]
        self.assertEqual((late["fwd_ok"], late["fwd_days"], late["ul_fwd"]), (0.0, 4.0, 0.0))
        last = ev.loc[("50050", self.bars["Date"].iloc[-1])]
        self.assertTrue(np.isnan(last["ul_fwd"]) and np.isnan(last["fwd_days"]))

    def test_day_flag(self):
        got = PL.day_flag(self.arrays, ["50050", "50050", "50050", "00000"], [self.t] * 4, [3, 4, np.nan, 3])
        self.assertEqual(got[0], 1.0)
        self.assertEqual(got[1], 0.0)
        self.assertTrue(np.isnan(got[2]) and np.isnan(got[3]))


if __name__ == "__main__":
    unittest.main()
