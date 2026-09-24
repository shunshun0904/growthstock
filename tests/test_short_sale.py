#!/usr/bin/env python3
"""
空売り残高報告の特徴量（extra_features.short_positions）と、市場全体の空売り比率
（extra_features.short_ratio_market）の単体テスト。

固定すること
  - 報告は公表日（DiscDate）の翌営業日から使う（当日の夜に間に合うか測るまで）
  - 報告者ごとに最新の報告だけが効く。報告者が違えば足す
  - 0.5% を下回った報告は持ち高を閉じる。365日より古い報告は数えない
  - ShrtPosToSO は割合（0.008 = 0.8%）。% に直して持つ
  - 行の並びは入力のまま（並べ替えて返すと列の代入が静かにずれる）
  - 市場全体の空売り比率は全業種の合計から作り、業種が揃っていない日は使わない。
    本番の short_ratio（S33=9999 の1行）は計算を変えない

  python3 tests/test_short_sale.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import availability as AV  # noqa: E402
import extra_features as X  # noqa: E402
import trading_calendar as TC  # noqa: E402

T = pd.Timestamp
HOLIDAYS = {dt.date(2026, 9, 21), dt.date(2026, 9, 22), dt.date(2026, 9, 23)}


def calendar_rows(start="2025-06-01", end="2026-12-31"):
    out = []
    d = dt.date.fromisoformat(start)
    while d <= dt.date.fromisoformat(end):
        div = "1" if d.weekday() < 5 and d not in HOLIDAYS else "0"
        out.append({"Date": d.isoformat(), "HolDiv": div})
        d += dt.timedelta(days=1)
    return out


def report(disc, code, who, ratio, calc=None):
    return {"DiscDate": disc, "CalcDate": calc or disc, "Code": code, "SSName": who,
            "FundName": None, "DICName": None, "ShrtPosToSO": ratio}


class ShortPositions(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        TC.save(calendar_rows(), self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, rows):
        df = pd.DataFrame(rows)
        df["DiscDate"] = pd.to_datetime(df["DiscDate"])
        df.to_parquet(os.path.join(self.dir, "shortsale_2026.parquet"), index=False)

    def feats(self, dates, code="11110"):
        s = pd.DataFrame({"Code": [code] * len(dates), "Date": [T(d) for d in dates]})
        return X.short_positions(s, self.dir)

    def test_report_is_used_from_the_next_trading_day(self):
        # 9/18(金) 公表。9/21〜23 は休みなので、使えるのは 9/24 から
        self.write([report("2026-09-18", "11110", "A", 0.012)])
        f = self.feats(["2026-09-18", "2026-09-19", "2026-09-24"])
        self.assertEqual(list(f["ss_ratio"]), [0.0, 0.0, 1.2])
        self.assertEqual(list(f["ss_n"]), [0.0, 0.0, 1.0])

    def test_latest_report_of_a_reporter_replaces_the_earlier(self):
        self.write([report("2026-09-01", "11110", "A", 0.010),
                    report("2026-09-08", "11110", "A", 0.015)])
        f = self.feats(["2026-09-03", "2026-09-10"])
        self.assertAlmostEqual(f["ss_ratio"].iloc[0], 1.0)
        self.assertAlmostEqual(f["ss_ratio"].iloc[1], 1.5)
        self.assertEqual(list(f["ss_n"]), [1.0, 1.0])

    def test_different_reporters_are_summed(self):
        self.write([report("2026-09-01", "11110", "A", 0.010),
                    report("2026-09-02", "11110", "B", 0.008)])
        f = self.feats(["2026-09-10"])
        self.assertAlmostEqual(f["ss_ratio"].iloc[0], 1.8)
        self.assertEqual(f["ss_n"].iloc[0], 2.0)

    def test_report_below_threshold_closes_the_position(self):
        self.write([report("2026-09-01", "11110", "A", 0.010),
                    report("2026-09-08", "11110", "A", 0.004)])
        f = self.feats(["2026-09-10"])
        self.assertEqual(f["ss_ratio"].iloc[0], 0.0)
        self.assertEqual(f["ss_n"].iloc[0], 0.0)

    def test_old_report_expires(self):
        self.write([report("2025-09-01", "11110", "A", 0.010)])
        # 使えるのは 2025-09-02 から。365日後の 2026-09-02 で外れる
        f = self.feats(["2026-09-01", "2026-09-02"])
        self.assertAlmostEqual(f["ss_ratio"].iloc[0], 1.0)
        self.assertEqual(f["ss_ratio"].iloc[1], 0.0)

    def test_change_over_28_days(self):
        self.write([report("2026-08-03", "11110", "A", 0.010),
                    report("2026-09-01", "11110", "B", 0.007)])
        f = self.feats(["2026-09-10"])
        # 28日前（8/13）は A の 1.0 だけ。今は 1.0 + 0.7
        self.assertAlmostEqual(f["ss_chg_20"].iloc[0], 0.7)

    def test_days_since_last_report(self):
        self.write([report("2026-09-01", "11110", "A", 0.010)])
        f = self.feats(["2026-09-10"])
        self.assertEqual(f["ss_days"].iloc[0], 8.0)             # 使えるのは 9/2 から
        g = self.feats(["2026-09-10"], code="99990")            # 報告が無い銘柄
        self.assertTrue(np.isnan(g["ss_days"].iloc[0]))
        self.assertEqual(g["ss_ratio"].iloc[0], 0.0)

    def test_other_codes_do_not_leak_in(self):
        self.write([report("2026-09-01", "22220", "A", 0.030)])
        self.assertEqual(self.feats(["2026-09-10"])["ss_ratio"].iloc[0], 0.0)

    def test_row_order_is_kept(self):
        self.write([report("2026-09-01", "11110", "A", 0.010),
                    report("2026-09-01", "22220", "A", 0.020)])
        s = pd.DataFrame({"Code": ["22220", "11110", "22220"],
                          "Date": [T("2026-09-10"), T("2026-09-10"), T("2026-08-20")]},
                         index=[7, 3, 5])
        f = X.short_positions(s, self.dir)
        self.assertEqual(list(f.index), [7, 3, 5])
        self.assertEqual(list(f["ss_ratio"]), [2.0, 1.0, 0.0])

    def test_same_day_availability_switch(self):
        self.write([report("2026-09-18", "11110", "A", 0.012)])
        old = AV.SHORTSALE_SAME_DAY
        try:
            AV.SHORTSALE_SAME_DAY = True
            self.assertEqual(self.feats(["2026-09-18"])["ss_ratio"].iloc[0], 1.2)
        finally:
            AV.SHORTSALE_SAME_DAY = old

    def test_no_data_gives_no_columns(self):
        s = pd.DataFrame({"Code": ["11110"], "Date": [T("2026-09-10")]})
        self.assertEqual(X.short_positions(s, self.dir).shape[1], 0)


class MarketShortRatio(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, rows):
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df.to_parquet(os.path.join(self.dir, "shortratio_2026.parquet"), index=False)

    def day(self, date, other_short=10.0):
        """33業種は各 (売り 90, 空売り 10)、その他(9999) は (売り 90, 空売り other_short)。"""
        rows = [{"Date": date, "S33": f"{i:04d}", "SellExShortVa": 90.0,
                 "ShrtWithResVa": 5.0, "ShrtNoResVa": 5.0} for i in range(33)]
        rows.append({"Date": date, "S33": "9999", "SellExShortVa": 90.0,
                     "ShrtWithResVa": other_short, "ShrtNoResVa": 0.0})
        return rows

    def test_market_ratio_uses_every_sector(self):
        self.write(self.day("2026-09-17", other_short=60.0))
        s = pd.DataFrame({"Code": ["11110"], "Date": [T("2026-09-18")]})
        m = X.short_ratio_market(s, self.dir)
        # 空売り 33*10 + 60 = 390、合計 34*90 + 390 = 3450
        self.assertAlmostEqual(m["short_ratio_mkt"].iloc[0], 390 / 3450 * 100)
        # 本番の short_ratio は S33=9999 の1行だけ（60 / 150）。計算を変えていない
        o = X.short_ratio(s, self.dir)
        self.assertAlmostEqual(o["short_ratio"].iloc[0], 60 / 150 * 100)

    def test_days_with_missing_sectors_are_not_used(self):
        # 9/17 は全業種、9/18 は 9999 の1行だけ（潰れていた日の名残り）
        rows = self.day("2026-09-17") + [r for r in self.day("2026-09-18") if r["S33"] == "9999"]
        self.write(rows)
        s = pd.DataFrame({"Code": ["11110"], "Date": [T("2026-09-18")]})
        m = X.short_ratio_market(s, self.dir)
        self.assertAlmostEqual(m["short_ratio_mkt"].iloc[0], 340 / 3400 * 100)   # 9/17 の値


if __name__ == "__main__":
    unittest.main(verbosity=2)
