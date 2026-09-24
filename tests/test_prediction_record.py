#!/usr/bin/env python3
"""
予測の記録（public/data/prediction_history.json とスプレッドシートの追跡列）の
単体テスト。

実運用の検証はこの記録が「その日に実際に出した予測」であることが前提になる。
2026-09-23・24 に、予測済みの 9/18 を後から新しいモデルで予測し直した実行が
2回あり、どちらも 9/18 の記録に当時は選んでいない銘柄（日本ナレッジ）を
足した。ここではその経路を固定する。

  - その日の記録が1件でもあれば、後から足さない（最初の予測で凍結）
  - 追跡列（現在値・騰落率・経過日）の更新は続ける
  - 経過日は取引所の営業日で数える（祝日を数えない）

  python3 tests/test_prediction_record.py
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import pandas as pd  # noqa: E402

import export_sheets as ES  # noqa: E402
import predict_daily as PD  # noqa: E402
import trading_calendar as TC  # noqa: E402

D = dt.date.fromisoformat


def calendar_rows():
    """2026-09-14〜30。21(敬老の日)・22(国民の休日)・23(秋分の日) と土日は休み。"""
    out = []
    for day in range(14, 31):
        d = dt.date(2026, 9, day)
        holiday = d.weekday() >= 5 or day in (21, 22, 23)
        out.append({"Date": d.isoformat(), "HolDiv": "0" if holiday else "1"})
    return out


def pick(date, code, rank, close=1000.0):
    return {"date": date, "jqCode": f"{code}0", "code": code, "name": code,
            "rankInDay": rank, "nInDay": 8, "score": 0.5, "band": 8,
            "calibProb": 25.0, "needPct": 10.0, "close": close}


class HistoryIsFrozenPerDate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.data = os.path.join(self.dir, "data")
        self.out = os.path.join(self.dir, "out")
        os.makedirs(self.data)
        os.makedirs(self.out)
        TC.save(calendar_rows(), self.data)
        bars = pd.DataFrame({"Date": pd.to_datetime(["2026-09-18", "2026-09-24"] * 2),
                             "Code": ["11110", "11110", "22220", "22220"],
                             "C": [1000.0, 1100.0, 500.0, 450.0]})
        bars.to_parquet(os.path.join(self.data, "bars_2026.parquet"))
        self.args = types.SimpleNamespace(out_dir=self.out, data_dir=self.data)
        self.days = [pd.Timestamp("2026-09-18")]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def history(self):
        with open(os.path.join(self.out, "prediction_history.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def test_first_prediction_is_recorded(self):
        PD.update_history(self.args, [pick("2026-09-18", "1111", 1)], self.days)
        got = [(e["date"], e["code"]) for e in self.history()["entries"]]
        self.assertEqual(got, [("2026-09-18", "1111")])

    def test_later_run_for_the_same_date_adds_nothing(self):
        """9/18 を後から別のモデルで予測し直しても、9/18 の記録は増えない。"""
        PD.update_history(self.args, [pick("2026-09-18", "1111", 1)], self.days)
        # 予測し直し: 上位が入れ替わり、当時は選んでいない 2222 が上位に来た
        PD.update_history(self.args, [pick("2026-09-18", "2222", 1),
                                      pick("2026-09-18", "1111", 2)], self.days)
        got = sorted(e["code"] for e in self.history()["entries"])
        self.assertEqual(got, ["1111"])

    def test_tracking_is_still_updated(self):
        PD.update_history(self.args, [pick("2026-09-18", "1111", 1)], self.days)
        PD.update_history(self.args, [pick("2026-09-18", "2222", 1)], self.days)
        e = self.history()["entries"][0]
        self.assertEqual(e["closeNow"], 1100.0)
        self.assertEqual(e["returnPct"], 10.0)

    def test_elapsed_days_skip_holidays(self):
        """9/18(金) の予測を 9/24(木) の終値で見ると、経過は1営業日（9/18）。
        平日で数えると 9/18・21・22・23 の4日になっていた。"""
        PD.update_history(self.args, [pick("2026-09-18", "1111", 1)], self.days)
        self.assertEqual(self.history()["entries"][0]["daysElapsed"], 1)


class ElapsedDays(unittest.TestCase):
    def setUp(self):
        self.cal = TC.Calendar(TC.parse_rows(calendar_rows()),
                               [D(f"2026-09-{d:02d}") for d in range(14, 31)])

    def test_calendar_counts_trading_days(self):
        self.assertEqual(TC.elapsed_days(D("2026-09-18"), D("2026-09-24"), self.cal), 1)
        self.assertEqual(TC.elapsed_days(D("2026-09-16"), D("2026-09-18"), self.cal), 2)

    def test_falls_back_to_weekdays_outside_calendar(self):
        # カレンダーが無い・覆っていないときは np.busday_count と同じ
        self.assertEqual(TC.elapsed_days(D("2026-09-18"), D("2026-09-24"), None), 4)
        self.assertEqual(TC.elapsed_days(D("2026-10-02"), D("2026-10-06"), self.cal), 2)

    def test_same_day_is_zero(self):
        self.assertEqual(TC.elapsed_days(D("2026-09-18"), D("2026-09-18"), self.cal), 0)

    def test_sheet_tracking_uses_the_calendar(self):
        tv = ES.track_values("11110", 1000.0, "2026-09-18", pd.Series({"11110": 1100.0}),
                             pd.Timestamp("2026-09-24"), self.cal)
        self.assertEqual(tv["経過営業日"], 1)
        tv = ES.track_values("11110", 1000.0, "2026-09-18", pd.Series({"11110": 1100.0}),
                             pd.Timestamp("2026-09-24"))
        self.assertEqual(tv["経過営業日"], 4)       # カレンダー無しは従来どおり平日


if __name__ == "__main__":
    unittest.main(verbosity=2)
