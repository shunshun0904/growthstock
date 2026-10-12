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
  - 予測タブの過去の日は、その日に出した値のまま（predict_daily.freeze_past_days。運用者の指示 2026-10-12）。
    一度だけの移行（research/restore_first_predictions.py）は、日次予測の実行が出した版のうち、その日が最新だった
    最後の版に戻す

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
import restore_first_predictions as RF  # noqa: E402
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


def row(date, code, score, rank=1):
    return {"date": date, "jqCode": f"{code}0", "code": code, "score": score, "rankInDay": rank}


class PastDaysAreFrozen(unittest.TestCase):
    """過去の日は前の predictions.json の行のまま。新しい日だけ今回の採点。"""

    DAYS = ["2026-10-07", "2026-10-08", "2026-10-09"]
    RUN = {"trainedAt": "2026-10-11T00:00:00", "generatedAt": "2026-10-13T09:50:00+00:00"}

    def prev(self, day_info=True):
        p = {"asOf": "2026-10-08", "dates": ["2026-10-06", "2026-10-07", "2026-10-08"],
             "generatedAt": "2026-10-08T09:50:00+00:00", "model": {"trainedAt": "2026-10-04T06:00:00"},
             "candidates": [row("2026-10-06", "1000", 0.9), row("2026-10-07", "1111", 0.5),
                            row("2026-10-08", "2222", 0.6), row("2026-10-08", "3333", 0.4, 2)]}
        if day_info:
            p["dayInfo"] = {"2026-10-07": {"trainedAt": "2026-09-27T00:00:00", "generatedAt": "2026-10-07T07:21:00"}}
        return p

    def rescored(self):
        # 新しいモデルで採点し直すと、過去の日のスコア・順位・候補の数が変わる
        return [row("2026-10-07", "1111", 0.8), row("2026-10-08", "3333", 0.9), row("2026-10-08", "2222", 0.1, 2),
                row("2026-10-08", "4444", 0.05, 3), row("2026-10-09", "5555", 0.7)]

    def test_past_days_keep_previous_rows(self):
        rows, info, st = PD.freeze_past_days(self.rescored(), self.DAYS, self.prev(), self.RUN)
        got = sorted((x["date"], x["code"], x["score"]) for x in rows)
        self.assertEqual(got, [("2026-10-07", "1111", 0.5), ("2026-10-08", "2222", 0.6), ("2026-10-08", "3333", 0.4),
                               ("2026-10-09", "5555", 0.7)])
        self.assertEqual(st["frozen"], ["2026-10-07", "2026-10-08"])
        self.assertEqual(st["scored"], ["2026-10-09"])
        self.assertEqual((st["rows_frozen"], st["rows_dropped"]), (3, 4))
        self.assertFalse(st["latest_overwritten"])
        # dayInfo: 前のファイルの dayInfo を引き継ぎ、無い日は前のファイルの model.trainedAt / generatedAt
        self.assertEqual(info["2026-10-07"]["trainedAt"], "2026-09-27T00:00:00")
        self.assertEqual(info["2026-10-08"], {"trainedAt": "2026-10-04T06:00:00",
                                               "generatedAt": "2026-10-08T09:50:00+00:00"})
        self.assertEqual(info["2026-10-09"], self.RUN)
        self.assertNotIn("2026-10-06", info)                    # 窓から外れた日は出さない

    def test_without_previous_file_everything_is_scored(self):
        rows, info, st = PD.freeze_past_days(self.rescored(), self.DAYS, None, self.RUN)
        self.assertEqual(len(rows), 5)
        self.assertEqual(st["frozen"], [])
        self.assertTrue(all(v == self.RUN for v in info.values()))

    def test_new_past_day_is_scored_and_empty_day_stays_empty(self):
        prev = self.prev()
        prev["dates"] = ["2026-10-08"]                          # 10/7 は前のファイルに無い → 初めて出す日
        prev["candidates"] = []                                 # 10/8 は候補0件の日として出していた
        rows, info, st = PD.freeze_past_days(self.rescored(), self.DAYS, prev, self.RUN)
        got = sorted((x["date"], x["score"]) for x in rows)
        self.assertEqual(got, [("2026-10-07", 0.8), ("2026-10-09", 0.7)])
        self.assertEqual(st["frozen"], ["2026-10-08"])
        self.assertEqual(info["2026-10-07"], self.RUN)

    def test_latest_day_is_overwritten_only_by_an_explicit_rerun(self):
        prev = self.prev()
        prev["dates"].append("2026-10-09")                     # 同じ日を予測し直した（FORCE_PREDICT）
        prev["candidates"].append(row("2026-10-09", "9999", 0.99))
        rows, info, st = PD.freeze_past_days(self.rescored(), self.DAYS, prev, self.RUN)
        latest = [(x["code"], x["score"]) for x in rows if x["date"] == "2026-10-09"]
        self.assertEqual(latest, [("5555", 0.7)])
        self.assertTrue(st["latest_overwritten"])

    def test_chained_runs_keep_the_first_values(self):
        """次の日の実行でも、前の実行で固定した値（dayInfo ごと）がそのまま残る。"""
        rows1, info1, _ = PD.freeze_past_days(self.rescored(), self.DAYS, self.prev(), self.RUN)
        prev2 = {"asOf": "2026-10-09", "dates": self.DAYS, "generatedAt": self.RUN["generatedAt"],
                 "model": {"trainedAt": self.RUN["trainedAt"]}, "dayInfo": info1, "candidates": rows1}
        days2 = ["2026-10-08", "2026-10-09", "2026-10-13"]
        run2 = {"trainedAt": "2026-10-18T00:00:00", "generatedAt": "2026-10-13T09:50:00+00:00"}
        again = [row("2026-10-08", "2222", 0.01), row("2026-10-09", "5555", 0.02), row("2026-10-13", "6666", 0.3)]
        rows2, info2, _ = PD.freeze_past_days(again, days2, prev2, run2)
        got = sorted((x["date"], x["code"], x["score"]) for x in rows2)
        self.assertEqual(got, [("2026-10-08", "2222", 0.6), ("2026-10-08", "3333", 0.4), ("2026-10-09", "5555", 0.7),
                               ("2026-10-13", "6666", 0.3)])
        self.assertEqual(info2["2026-10-08"]["trainedAt"], "2026-10-04T06:00:00")
        self.assertEqual(info2["2026-10-09"], self.RUN)
        self.assertEqual(info2["2026-10-13"], run2)

    def test_load_previous_tolerates_missing_or_broken_file(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "predictions.json")
            self.assertIsNone(PD.load_previous(path))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{broken")
            self.assertIsNone(PD.load_previous(path))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"dates": ["2026-10-08"]}, fh)
            self.assertEqual(PD.load_previous(path)["dates"], ["2026-10-08"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class RestoreFirstPredictions(unittest.TestCase):
    """一度だけの移行: 各日を、その日が最新だった最後の版（日次予測の実行が出した版）の行に戻す。"""

    def test_restore_uses_the_last_version_of_each_day(self):
        v1 = {"asOf": "2026-10-07", "generatedAt": "g1", "model": {"trainedAt": "t1"},
              "candidates": [row("2026-10-07", "1111", 0.5)]}
        v2 = {"asOf": "2026-10-08", "generatedAt": "g2", "model": {"trainedAt": "t1"},
              "candidates": [row("2026-10-07", "1111", 0.55), row("2026-10-08", "2222", 0.6),
                             row("2026-10-08", "3333", 0.4, 2)]}
        v3 = {"asOf": "2026-10-08", "generatedAt": "g3", "model": {"trainedAt": "t1"},    # 同じ日に予測し直した版
              "candidates": [row("2026-10-08", "2222", 0.61), row("2026-10-08", "3333", 0.41, 2)]}
        cur = {"asOf": "2026-10-09", "dates": ["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"],
               "generatedAt": "g4", "model": {"trainedAt": "t2"},
               "candidates": [row("2026-10-06", "0000", 0.3), row("2026-10-07", "1111", 0.9),
                              row("2026-10-08", "3333", 0.8), row("2026-10-09", "5555", 0.7)]}
        new, rep = RF.restore(cur, [("a", v1), ("b", v2), ("c", v3), ("d", cur)])
        got = sorted((x["date"], x["code"], x["score"]) for x in new["candidates"])
        self.assertEqual(got, [("2026-10-06", "0000", 0.3), ("2026-10-07", "1111", 0.5), ("2026-10-08", "2222", 0.61),
                               ("2026-10-08", "3333", 0.41), ("2026-10-09", "5555", 0.7)])
        self.assertEqual(new["dayInfo"]["2026-10-07"], {"trainedAt": "t1", "generatedAt": "g1"})
        self.assertEqual(new["dayInfo"]["2026-10-08"]["generatedAt"], "g3")
        self.assertEqual(new["dayInfo"]["2026-10-09"], {"trainedAt": "t2", "generatedAt": "g4"})
        self.assertEqual(new["dayInfo"]["2026-10-06"]["generatedAt"], "g4")      # 最新だった版が無い日は今のまま
        by = {r["date"]: r for r in rep}
        self.assertEqual((by["2026-10-07"]["source"], by["2026-10-07"]["changed"]), ("a", 1))
        self.assertEqual((by["2026-10-08"]["before"], by["2026-10-08"]["after"], by["2026-10-08"]["changed"]), (1, 2, 2))
        self.assertEqual(by["2026-10-09"]["changed"], 0)
        self.assertEqual(new["dates"], cur["dates"])
        self.assertNotIn("dayInfo", cur)                                      # 元の中身は書き換えない


if __name__ == "__main__":
    unittest.main(verbosity=2)
