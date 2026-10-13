#!/usr/bin/env python3
"""
次の決算発表の予定（research/predict_daily.py の next_earnings）と、予測ログの列（export_sheets）。

運用者の依頼（2026-10-13）「予測が実質的に、次の決算の予測になっていないか確認するために、次の決算発表までの日数も
どこかに表示してほしい」。

- 予測日までに公表済み（PubDate ≤ 予測日）で、予測日以降の予定のうち一番早いもの（特徴量 days_to_earn と同じ規則）
- 予測日より後に公表された予定は使わない（先の情報を見ない）
- 営業日は取引所のカレンダーで数える（祝日を数えない）。予測日の当日に発表なら 0
- 予定が無い銘柄は入れない
- 予測ログの行に「次の決算日」「次の決算まで営業日」「前回開示からの日数」が入る

  python3 tests/test_next_earnings.py
"""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import pandas as pd  # noqa: E402

import export_sheets as ES  # noqa: E402
import predict_daily as PD  # noqa: E402
import trading_calendar as TC  # noqa: E402


def calendar_rows():
    """2026-10-01〜11-30。10/12（スポーツの日）・11/3（文化の日）・11/23（勤労感謝の日）と土日は休み。"""
    out = []
    for d in pd.date_range("2026-10-01", "2026-11-30"):
        hol = d.weekday() >= 5 or d.strftime("%m-%d") in ("10-12", "11-03", "11-23")
        out.append({"Date": d.date().isoformat(), "HolDiv": "0" if hol else "1"})
    return out


class NextEarnings(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        TC.save(calendar_rows(), self.dir)
        rows = [
            # 1111: 10/9 に 10/30 の予定を公表、10/20 に 10/28 へ前倒し（10/13 の時点では知らない）
            {"PubDate": "2026-10-09", "SchDate": "2026-10-30", "FQName": "2Q", "Code": "11110"},
            {"PubDate": "2026-10-20", "SchDate": "2026-10-28", "FQName": "2Q", "Code": "11110"},
            # 2222: 予定が2つ（近いほう）
            {"PubDate": "2026-09-01", "SchDate": "2026-11-13", "FQName": "2Q", "Code": "22220"},
            {"PubDate": "2026-09-15", "SchDate": "2026-10-14", "FQName": "1Q", "Code": "22220"},
            # 3333: 予測日の当日に発表
            {"PubDate": "2026-09-01", "SchDate": "2026-10-13", "FQName": "FY", "Code": "33330"},
            # 4444: 過去の予定だけ
            {"PubDate": "2026-07-01", "SchDate": "2026-08-07", "FQName": "1Q", "Code": "44440"},
        ]
        e = pd.DataFrame(rows)
        e["PubDate"] = pd.to_datetime(e["PubDate"])
        e.to_parquet(os.path.join(self.dir, "earndate_2026.parquet"), index=False)
        self.cal = TC.load(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_rule(self):
        keys = [("11110", "2026-10-13"), ("22220", "2026-10-13"), ("33330", "2026-10-13"),
                ("44440", "2026-10-13"), ("55550", "2026-10-13")]
        got = PD.next_earnings(keys, self.dir, self.cal)
        a = got[("11110", "2026-10-13")]
        self.assertEqual((a["date"], a["quarter"], a["days"]), ("2026-10-30", "2Q", 17))
        self.assertEqual(a["tradingDays"], 13)            # 10/13〜10/29 の営業日（10/30 が13営業日目）
        b = got[("22220", "2026-10-13")]
        self.assertEqual((b["date"], b["quarter"], b["tradingDays"]), ("2026-10-14", "1Q", 1))
        c = got[("33330", "2026-10-13")]
        self.assertEqual((c["days"], c["tradingDays"]), (0, 0))
        self.assertNotIn(("44440", "2026-10-13"), got)    # 過去の予定だけ
        self.assertNotIn(("55550", "2026-10-13"), got)    # 予定が無い

    def test_later_publication_is_used_only_after_it_is_published(self):
        got = PD.next_earnings([("11110", "2026-10-21")], self.dir, self.cal)
        self.assertEqual(got[("11110", "2026-10-21")]["date"], "2026-10-28")

    def test_trading_days_skip_holidays(self):
        # 10/9（金）から 10/14（水）: 10/12 は祝日なので 10/9・10/13 の2営業日（10/14 が2営業日目）
        got = PD.next_earnings([("22220", "2026-10-09")], self.dir, self.cal)
        self.assertEqual(got[("22220", "2026-10-09")]["tradingDays"], 2)

    def test_no_data(self):
        empty = tempfile.mkdtemp()
        try:
            self.assertEqual(PD.next_earnings([("11110", "2026-10-13")], empty, None), {})
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class SheetColumns(unittest.TestCase):
    def test_rows_carry_the_earnings_columns(self):
        pred = {"models": [], "candidates": [{
            "date": "2026-10-13", "code": "1111", "jqCode": "11110", "name": "A", "rankInDay": 1, "nInDay": 3,
            "score": 0.5, "band": 8, "nextEarnDate": "2026-10-30", "nextEarnTradingDays": 13, "daysSinceDisc": 75}]}
        row = ES.rows_from_predictions(pred)[0]
        self.assertEqual((row["次の決算日"], row["次の決算まで営業日"], row["前回開示からの日数"]), ("2026-10-30", 13, 75))
        for c in ES.EARN_COLS:
            self.assertIn(c, row)


if __name__ == "__main__":
    unittest.main()
