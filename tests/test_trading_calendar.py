#!/usr/bin/env python3
"""
research/trading_calendar.py と、それを使う鮮度チェック・営業日ゲートの単体テスト。

2026-09-22（敬老の日〜秋分の日の連休の中日）に日次予測が落ちた件を
固定する。落ちた経路は2つあり、どちらもここで押さえる。

  1. 鮮度チェックが遅れを平日で数えていた（連休で「遅れ2営業日」）
  2. ゲートが manifest の鮮度（12時間以内）を条件にしていたため、
     その日の取り込みがまだ走っていないと使えなかった（実測18時間前）

**カレンダーを言い訳にしない**ことも固定する。取引所が営業日だと言った日に
データが無ければ、遅れとして数えなければならない。

  python3 tests/test_trading_calendar.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import check_freshness  # noqa: E402
import jq_bulk  # noqa: E402
import trading_calendar as TC  # noqa: E402
import trading_day_gate as GATE  # noqa: E402

D = dt.date.fromisoformat


def rows_2026_09():
    """
    2026年9月の実際の並び。第3月曜(21)が敬老の日、23が秋分の日、
    その間の22は国民の休日。19-20 は土日。
    """
    out = []
    for day in range(14, 31):
        d = dt.date(2026, 9, day)
        weekend = d.weekday() >= 5
        holiday = day in (21, 22, 23)
        out.append({"Date": d.isoformat(),
                    "HolDiv": "0" if (weekend or holiday) else "1"})
    return out


class TestCalendar(unittest.TestCase):
    def setUp(self):
        self.cal = TC.Calendar(TC.parse_rows(rows_2026_09()),
                               [D(f"2026-09-{d:02d}") for d in range(14, 31)])

    def test_連休の3日は営業日ではない(self):
        for day in (21, 22, 23):
            self.assertFalse(self.cal.is_trading_day(dt.date(2026, 9, day)),
                             f"9/{day} を営業日と判定した")
        self.assertTrue(self.cal.is_trading_day(D("2026-09-18")))
        self.assertTrue(self.cal.is_trading_day(D("2026-09-24")))

    def test_連休中の遅れは1のまま(self):
        # 遅れ1 = 「前営業日ぶん」で正常。許容 1 を超えない。
        # ここが落ちた本体で、平日で数えると 9/22 は 2、9/23 は 3 になる
        self.assertEqual(self.cal.count_between(D("2026-09-18"), D("2026-09-22")), 1)
        self.assertEqual(self.cal.count_between(D("2026-09-18"), D("2026-09-23")), 1)
        # 連休明けの 9/24 も、その日のバーはまだ出ていないので 1 のまま
        self.assertEqual(self.cal.count_between(D("2026-09-18"), D("2026-09-24")), 1)
        # 9/24 を取りこぼしたまま 9/25 になったら 2。ここは落ちてよい
        self.assertEqual(self.cal.count_between(D("2026-09-18"), D("2026-09-25")), 2)

    def test_範囲外は分からないと答える(self):
        self.assertIsNone(self.cal.is_trading_day(D("2026-08-01")))
        self.assertIsNone(self.cal.count_between(D("2026-08-01"), D("2026-09-22")))
        self.assertIsNone(self.cal.last_trading_day(D("2026-10-31")))

    def test_直近の営業日(self):
        self.assertEqual(self.cal.last_trading_day(D("2026-09-22")), D("2026-09-18"))
        self.assertEqual(self.cal.last_trading_day(D("2026-09-18")), D("2026-09-18"))

    def test_半日立会は営業日に数える(self):
        cal = TC.Calendar(TC.parse_rows([{"Date": "2026-01-05", "HolDiv": "2"}]))
        self.assertTrue(cal.is_trading_day(D("2026-01-05")))

    def test_祝日取引ありは営業日に数えない(self):
        # HolDiv "3" は東証の立会が無い（デリバティブのみ）
        cal = TC.Calendar(TC.parse_rows([{"Date": "2026-01-02", "HolDiv": "3"}]),
                          [D("2026-01-02")])
        self.assertFalse(cal.is_trading_day(D("2026-01-02")))

    def test_空のカレンダーは何も答えない(self):
        cal = TC.Calendar([])
        self.assertFalse(bool(cal))
        self.assertIsNone(cal.is_trading_day(D("2026-09-22")))
        self.assertIsNone(cal.count_between(D("2026-09-18"), D("2026-09-22")))


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_保存して読み直せる(self):
        TC.save(rows_2026_09(), self.dir)
        cal = TC.load(self.dir)
        self.assertFalse(cal.is_trading_day(D("2026-09-22")))
        self.assertTrue(cal.is_trading_day(D("2026-09-18")))

    def test_追記しても古い行が残る(self):
        TC.save([{"Date": "2026-09-18", "HolDiv": "1"}], self.dir)
        TC.save([{"Date": "2026-09-24", "HolDiv": "1"}], self.dir)
        cal = TC.load(self.dir)
        self.assertTrue(cal.is_trading_day(D("2026-09-18")))
        self.assertTrue(cal.is_trading_day(D("2026-09-24")))

    def test_同じ日は後から来たほうで上書きする(self):
        TC.save([{"Date": "2026-09-22", "HolDiv": "1"}], self.dir)
        TC.save([{"Date": "2026-09-22", "HolDiv": "0"}], self.dir)
        self.assertFalse(TC.load(self.dir).is_trading_day(D("2026-09-22")))

    def test_非営業日の行も残す(self):
        # 営業日だけ保存すると「非営業日」と「範囲外」が区別できなくなる
        TC.save(rows_2026_09(), self.dir)
        df = pd.read_parquet(os.path.join(self.dir, TC.FILENAME))
        self.assertIn("2026-09-22", set(df["Date"]))
        self.assertEqual(TC.load(self.dir).is_trading_day(D("2026-09-22")), False)

    def test_ファイルが無くても落ちない(self):
        self.assertFalse(bool(TC.load(self.dir)))

    def test_壊れていても落ちない(self):
        with open(os.path.join(self.dir, TC.FILENAME), "w") as fh:
            fh.write("これは parquet ではない")
        self.assertFalse(bool(TC.load(self.dir)))

    def test_空の入力では書かない(self):
        self.assertIsNone(TC.save([], self.dir))
        self.assertFalse(os.path.exists(os.path.join(self.dir, TC.FILENAME)))


class TestFreshness(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_カレンダーがあれば連休で落ちない(self):
        # 2026-09-22 に実際に落ちた条件。許容1 を超えないことまで確かめる
        TC.save(rows_2026_09(), self.dir)
        for today in ("2026-09-22", "2026-09-23", "2026-09-24"):
            age, basis = check_freshness.count_age(D("2026-09-18"), D(today), self.dir)
            self.assertEqual(basis, "カレンダー")
            self.assertLessEqual(age, 1, f"{today} で遅れ {age}（許容1を超える）")

    def test_カレンダーが無ければ平日で数える(self):
        # 落ちていたときと同じ値。カレンダーが無ければ厳しい側に倒れる
        age, basis = check_freshness.count_age(D("2026-09-18"), D("2026-09-22"), self.dir)
        self.assertEqual(age, 2)
        self.assertEqual(basis, "平日")
        self.assertGreater(age, 1, "平日で数えると許容1を超える（これが落ちた理由）")

    def test_連休明けの取りこぼしは落とす(self):
        # 9/24 は営業日。そのバーが無いまま 9/25 になったら遅れ2で落ちる
        TC.save(rows_2026_09(), self.dir)
        age, basis = check_freshness.count_age(D("2026-09-18"), D("2026-09-25"), self.dir)
        self.assertEqual((age, basis), (2, "カレンダー"))

    def test_範囲外なら平日に倒す(self):
        TC.save(rows_2026_09(), self.dir)
        age, basis = check_freshness.count_age(D("2026-08-03"), D("2026-08-05"), self.dir)
        self.assertEqual(basis, "平日")

    def test_営業日にデータが無ければ遅れとして数える(self):
        # カレンダーを言い訳にしない。ここが緩むと障害を見逃す
        TC.save(rows_2026_09(), self.dir)
        age, basis = check_freshness.count_age(D("2026-09-16"), D("2026-09-18"), self.dir)
        self.assertEqual(age, 2)                     # 9/16, 9/17 の2営業日ぶん
        self.assertEqual(basis, "カレンダー")


class TestGate(unittest.TestCase):
    def setUp(self):
        self.cal = TC.Calendar(TC.parse_rows(rows_2026_09()),
                               [D(f"2026-09-{d:02d}") for d in range(14, 31)])

    def test_連休の中日は飛ばす(self):
        # manifest が古くても（18時間前でも）カレンダーで判定できる
        run, why = GATE.decide({}, D("2026-09-18"), dt.datetime.now(dt.timezone.utc),
                               cal=self.cal, today=D("2026-09-22"))
        self.assertFalse(run, why)
        self.assertIn("非営業日", why)

    def test_営業日なら走らせる(self):
        run, why = GATE.decide({}, D("2026-09-18"), dt.datetime.now(dt.timezone.utc),
                               cal=self.cal, today=D("2026-09-18"))
        self.assertTrue(run, why)

    def test_取り込みが遅れていれば休場日でも飛ばさない(self):
        # 最終バーが 9/16 で、直近の営業日 9/18 に届いていない
        run, why = GATE.decide({}, D("2026-09-16"), dt.datetime.now(dt.timezone.utc),
                               cal=self.cal, today=D("2026-09-22"))
        self.assertTrue(run, why)
        self.assertIn("届いていない", why)

    def test_カレンダーが範囲外なら_manifest_に落ちる(self):
        run, why = GATE.decide({}, D("2026-08-01"), dt.datetime.now(dt.timezone.utc),
                               cal=self.cal, today=D("2026-08-03"))
        self.assertTrue(run, why)
        self.assertIn("manifest", why)

    def test_カレンダーが無くても従来どおり動く(self):
        now = dt.datetime(2026, 9, 21, 13, 0, tzinfo=dt.timezone.utc)
        man = {"calendar": {"isTradingDay": False, "lastTradingDay": "2026-09-18",
                            "asOfJst": "2026-09-21"},
               "updatedAt": "2026-09-21T12:00:00+00:00"}
        run, why = GATE.decide(man, D("2026-09-18"), now, cal=None,
                               today=D("2026-09-21"))
        self.assertFalse(run, why)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestLookahead(unittest.TestCase):
    """
    カレンダーは end より先まで取っておく。下流は「今日」が範囲に入って
    いないと使えず、取り込みが当日まだ走っていない時刻に予測が動くため。
    """

    class FakeClient:
        def __init__(self):
            self.asked = None

        def get_paginated(self, path, params):
            self.asked = (path, dict(params))
            return rows_2026_09()

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_先の日まで取りに行く(self):
        c = self.FakeClient()
        jq_bulk.trading_days(c, D("2026-09-14"), D("2026-09-18"), self.dir)
        path, params = c.asked
        self.assertEqual(path, "/markets/calendar")
        self.assertEqual(params["from"], "2026-09-14")
        self.assertEqual(params["to"],
                         (D("2026-09-18")
                          + dt.timedelta(days=jq_bulk.CALENDAR_LOOKAHEAD_DAYS)).isoformat())

    def test_先の日は取りに行く対象にしない(self):
        c = self.FakeClient()
        days = jq_bulk.trading_days(c, D("2026-09-14"), D("2026-09-18"), self.dir)
        self.assertEqual(max(days), D("2026-09-18"))

    def test_先の日も保存はする(self):
        c = self.FakeClient()
        jq_bulk.trading_days(c, D("2026-09-14"), D("2026-09-18"), self.dir)
        cal = TC.load(self.dir)
        # 9/18 までしか取りに行かないが、9/24 が営業日だと分かる状態になる
        self.assertTrue(cal.is_trading_day(D("2026-09-24")))
        self.assertFalse(cal.is_trading_day(D("2026-09-22")))

    def test_保存に失敗しても取得は続く(self):
        c = self.FakeClient()
        days = jq_bulk.trading_days(c, D("2026-09-14"), D("2026-09-18"),
                                    "/存在しない/書けない場所")
        self.assertTrue(days)


class TestNewKinds(unittest.TestCase):
    """
    2026-09-22 に足した種別が、引数の選択肢と日付列の対応で取りこぼされて
    いないか。白リストの書き漏らしで列や種別が黙って落ちる事故を
    2回起こしているので、固定しておく（FIN_COLS / MASTER_COLS）。
    """

    def test_足した種別が7本ある(self):
        self.assertEqual(len(jq_bulk.DAILY_KINDS), 7)
        self.assertEqual(len(jq_bulk.BULK_KINDS), 1)

    def test_日付列はその行を知りえた日(self):
        # EDINET は提出日、信用規制は公表日。ここを取り違えると未来を見る
        self.assertEqual(jq_bulk.DAILY_KINDS["lvshld"][1], "SubDate")
        self.assertEqual(jq_bulk.DAILY_KINDS["mjrshld"][1], "SubDate")
        self.assertEqual(jq_bulk.DAILY_KINDS["xhold"][1], "SubDate")
        self.assertEqual(jq_bulk.DAILY_KINDS["marginalert"][1], "PubDate")

    def test_列を絞らない(self):
        # None = 白リストを書かない。書き漏らすと黙って落ちる
        for kind, (path, date_col, ja) in jq_bulk.DAILY_KINDS.items():
            self.assertTrue(path.startswith("/"), kind)
            self.assertTrue(date_col, kind)
            self.assertTrue(ja, kind)

    def test_種別名が既存とぶつからない(self):
        base = {"bars", "fins", "margin", "topix", "indices", "master", "master_hist"}
        for kind in list(jq_bulk.DAILY_KINDS) + list(jq_bulk.BULK_KINDS):
            self.assertNotIn(kind, base, f"{kind} は既存の種別と同名")
