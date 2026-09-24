#!/usr/bin/env python3
"""
特徴量の時点整合（research/availability.py と、それを使う結合）の単体テスト。

**特徴量の値は、予測の時点で公表済みだったものでなければならない。**
2026-09-24 まで、週次の信用残を基準日（金曜）で、投資部門別を集計期間の
末日で結合していて、学習だけが公表前の値を見ていた（信用残で母集団の 38.5%、
投資部門別で 85.7% の行）。予測では同じ新しさの値は手に入らない。

ここでは「公表日の前日までは前の値、公表日から新しい値」を種別ごとに固定する。
値の分布や計算の正しさを見るテストでは、この種の誤りは捕まらなかった。

  python3 tests/test_availability.py
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
import build_dataset as B  # noqa: E402
import extra_features as X  # noqa: E402
import jq_bulk  # noqa: E402

T = pd.Timestamp


def trading_days_2026_09():
    """2026-09-01〜10-09 の営業日。21(敬老の日)・22(国民の休日)・23(秋分の日) と土日は休み。"""
    out = []
    d = dt.date(2026, 9, 1)
    while d <= dt.date(2026, 10, 9):
        if d.weekday() < 5 and d not in (dt.date(2026, 9, 21), dt.date(2026, 9, 22),
                                          dt.date(2026, 9, 23), dt.date(2026, 10, 12)):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


class MarginPublishDate(unittest.TestCase):
    def setUp(self):
        self.days = trading_days_2026_09()

    def pub(self, asof):
        return AV.next_week_trading_day(pd.Series([T(asof)]), self.days).iloc[0]

    def test_friday_is_published_next_tuesday(self):
        self.assertEqual(self.pub("2026-09-11"), T("2026-09-15"))

    def test_holidays_push_it_back(self):
        # 9/21〜23 が休み。翌週の営業日は 9/24(1)・9/25(2)
        self.assertEqual(self.pub("2026-09-18"), T("2026-09-25"))

    def test_thursday_asof_when_friday_is_a_holiday(self):
        self.assertEqual(self.pub("2026-09-10"), T("2026-09-15"))

    def test_outside_calendar_counts_weekdays(self):
        self.assertEqual(AV.next_week_trading_day(pd.Series([T("2025-01-10")]), self.days)
                         .iloc[0], T("2025-01-14"))


class CreditRatioUsesPublishDate(unittest.TestCase):
    """信用倍率は、公表日（翌週の第2営業日）より前には新しい週の値を見ない。"""

    def setUp(self):
        self.days = trading_days_2026_09()
        self.margin = pd.DataFrame({
            "Date": [T("2026-09-04"), T("2026-09-11")],     # 基準日（金曜）
            "Code": ["11110", "11110"],
            "LongVol": [200.0, 900.0], "ShrtVol": [100.0, 100.0]})  # 2.0 と 9.0

    def ratio(self, day):
        s = pd.DataFrame({"Code": ["11110"], "Date": [T(day)]})
        return B.attach_credit_ratio(s, self.margin, self.days)["credit_ratio"].iloc[0]

    def test_friday_row_does_not_see_its_own_week(self):
        self.assertEqual(self.ratio("2026-09-11"), 2.0)   # 9/11 分は 9/15 公表

    def test_monday_row_does_not_see_last_week(self):
        self.assertEqual(self.ratio("2026-09-14"), 2.0)

    def test_publish_day_sees_it(self):
        self.assertEqual(self.ratio("2026-09-15"), 9.0)

    def test_old_value_expires(self):
        # 9/11 分（9/15 公表）は 21日後の 10/6 まで。それより後は欠測
        self.assertEqual(self.ratio("2026-10-06"), 9.0)
        self.assertTrue(np.isnan(self.ratio("2026-10-07")))

    def test_row_order_is_by_date_like_before(self):
        s = pd.DataFrame({"Code": ["11110", "11110"],
                          "Date": [T("2026-09-15"), T("2026-09-11")]})
        out = B.attach_credit_ratio(s, self.margin, self.days)
        self.assertEqual(list(out["Date"]), [T("2026-09-11"), T("2026-09-15")])
        self.assertEqual(list(out["credit_ratio"]), [2.0, 9.0])


class InvestorFlowsUsePublishDate(unittest.TestCase):
    """投資部門別は、集計期間の末日（EnDate）ではなく公表日（PubDate）で結合する。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rows = []
        # 週ごとに、差引の比が分かる値にする（外国人の差引 = 週番号）
        for k, (st, en, pub) in enumerate([("2026-08-31", "2026-09-04", "2026-09-10"),
                                           ("2026-09-07", "2026-09-11", "2026-09-17")], 1):
            r = {"PubDate": pub, "StDate": st, "EnDate": en, "Section": "TSEPrime"}
            for g in X.INVESTOR_GROUPS:
                r[f"{g}Buy"], r[f"{g}Sell"], r[f"{g}Bal"] = 50.0, 50.0, 0.0
            r["FrgnBal"] = float(k)
            rows.append(r)
        pd.DataFrame(rows).to_parquet(os.path.join(self.dir, "investor.parquet"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def flow(self, day):
        s = pd.DataFrame({"Code": ["11110"], "Date": [T(day)]})
        out = X.investor_types(s, self.dir)
        return out["inv_foreign"].iloc[0] if "inv_foreign" in out else None

    def test_before_publish_sees_previous_week(self):
        a, b = self.flow("2026-09-14"), self.flow("2026-09-17")
        self.assertIsNotNone(a)
        self.assertLess(a, b)                 # 9/14 は 9/4 週、9/17 から 9/11 週

    def test_week_end_day_does_not_see_its_own_week(self):
        self.assertEqual(self.flow("2026-09-11"), self.flow("2026-09-14"))

    def test_nothing_before_the_first_publish(self):
        self.assertTrue(np.isnan(self.flow("2026-09-09")))


class RegistryIsComplete(unittest.TestCase):
    def test_every_fetched_kind_has_a_rule(self):
        kinds = set(jq_bulk.DAILY_KINDS) | set(jq_bulk.BULK_KINDS) | {
            "bars", "indices", "topix", "master_hist", "fins", "margin"}
        missing = kinds - set(AV.RULES)
        self.assertFalse(missing, f"使ってよい日が決まっていない種別: {missing}")

    def test_investor_is_recorded_by_publish_date(self):
        self.assertEqual(jq_bulk.BULK_KINDS["investor"][1], "PubDate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
