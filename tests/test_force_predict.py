"""Explicit re-prediction must work on a non-trading day."""
import datetime as dt
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import trading_calendar as TC
import trading_day_gate as GATE


class FixedNow(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 3, 14, tzinfo=dt.timezone.utc).astimezone(tz)


class TestForcePredict(unittest.TestCase):
    def gate_output(self, force):
        days = [dt.date(2026, 11, 2), dt.date(2026, 11, 3)]
        cal = TC.Calendar([days[0]], days)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            predictions = Path(directory) / "predictions.json"
            predictions.write_text('{"asOf":"2026-11-02"}')
            with patch.dict(os.environ, {"FORCE_PREDICT": force,
                                         "GITHUB_OUTPUT": str(output)}), \
                    patch.object(GATE.dt, "datetime", FixedNow), \
                    patch.object(GATE.trading_calendar, "load", return_value=cal), \
                    patch.object(GATE, "last_bar_date", return_value=days[0]):
                self.assertEqual(GATE.main(["--data-dir", directory,
                                            "--predictions", str(predictions)]), 0)
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_force_repredicts_published_bars_on_holiday(self):
        self.assertEqual(self.gate_output("true")["run"], "true")

    def test_automatic_run_still_skips_published_bars_on_holiday(self):
        self.assertEqual(self.gate_output("false")["run"], "false")


def november(holidays=(dt.date(2026, 11, 3), dt.date(2026, 11, 23)), end=dt.date(2026, 11, 30)):
    """2026-11-01〜end のカレンダー。土日と holidays は休み。"""
    days, cov = [], []
    d = dt.date(2026, 11, 1)
    while d <= end:
        cov.append(d)
        if d.weekday() < 5 and d not in holidays:
            days.append(d)
        d += dt.timedelta(days=1)
    return TC.Calendar(days, cov)


class TestWeekendHold(unittest.TestCase):
    """
    次の営業日が週末の学習（土曜）の後になる日は、その土曜の 14:00 JST まで予測しない（運用者の選択 2026-11-15「金曜の夜は
    やめる」、2026-11-20「土曜20時まで止める」、2026-11-21「最後の予測を15時」「土曜14時まで止める（推奨）」）。1つの日付に
    予測は1つのまま（土曜の午後の予測がその日の最初の予測になる）。土曜の午前に動いても、学習前の古いモデルでは出さない。
    """

    def gate(self, now_jst, last_bar, as_of, cal=None, force="false"):
        jst = dt.timezone(dt.timedelta(hours=9))
        fixed = dt.datetime(*now_jst, tzinfo=jst)

        class Now(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz)

        cal = cal if cal is not None else november()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            predictions = Path(directory) / "predictions.json"
            predictions.write_text('{"asOf":"%s"}' % as_of.isoformat())
            with patch.dict(os.environ, {"FORCE_PREDICT": force, "GITHUB_OUTPUT": str(output)}), \
                    patch.object(GATE.dt, "datetime", Now), \
                    patch.object(GATE.trading_calendar, "load", return_value=cal), \
                    patch.object(GATE, "last_bar_date", return_value=last_bar):
                self.assertEqual(GATE.main(["--data-dir", directory, "--predictions", str(predictions)]), 0)
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    FRI, THU, SUN = dt.date(2026, 11, 13), dt.date(2026, 11, 12), dt.date(2026, 11, 15)

    def test_friday_night_waits_for_saturday_afternoon(self):
        out = self.gate((2026, 11, 13, 18, 50), self.FRI, self.THU)
        self.assertEqual(out["run"], "false")
        self.assertIn("2026-11-14 土曜の 14:00 以降", out["reason"])
        self.assertEqual(out["stale"], "false")          # 取り込みは揃っている（赤にしない）

    def test_saturday_before_2pm_still_waits(self):
        """土曜の 00:00（+15% の学習）・06:00（+10%）・12:00（ランキング）の間は、古いモデルで出さない。"""
        for hm in ((0, 30), (6, 0), (9, 0), (12, 0), (13, 59)):
            with self.subTest(hm=hm):
                self.assertEqual(self.gate((2026, 11, 14) + hm, self.FRI, self.THU)["run"], "false")

    def test_saturday_afternoon_predicts_friday(self):
        for hm in ((14, 0), (15, 0), (21, 0), (23, 50)):
            with self.subTest(hm=hm):
                out = self.gate((2026, 11, 14) + hm, self.FRI, self.THU)
                self.assertEqual(out["run"], "true")
                self.assertIn("未予測の日足がある", out["reason"])

    def test_after_saturday_prediction_does_not_repredict(self):
        self.assertEqual(self.gate((2026, 11, 14, 15, 30), self.FRI, self.FRI)["run"], "false")
        self.assertEqual(self.gate((2026, 11, 14, 23, 50), self.FRI, self.FRI)["run"], "false")
        self.assertEqual(self.gate((2026, 11, 15, 9, 0), self.FRI, self.FRI)["run"], "false")

    def test_sunday_still_predicts_if_saturday_was_missed(self):
        """土曜の午後に予測が動かなかったら、日曜に動いたときに出す（止めたままにしない）。"""
        self.assertEqual(self.gate((2026, 11, 15, 9, 0), self.FRI, self.THU)["run"], "true")

    def test_weekend_hold_accepts_a_date(self):
        """日付だけが来たらその日の 00:00 とみなす（土曜の日付だけでは出さない）。"""
        cal = november()
        self.assertIsNotNone(GATE.weekend_hold(cal, self.FRI, dt.date(2026, 11, 14)))
        self.assertIsNone(GATE.weekend_hold(cal, self.FRI, dt.date(2026, 11, 15)))

    def test_weekday_nights_are_unchanged(self):
        self.assertEqual(self.gate((2026, 11, 12, 18, 50), self.THU, dt.date(2026, 11, 11))["run"], "true")
        # 祝日（11/3 火）をまたぐ月曜の夜も、間に土曜が無いのでその夜に出す
        self.assertEqual(self.gate((2026, 11, 2, 18, 50), dt.date(2026, 11, 2), dt.date(2026, 10, 30))["run"], "true")

    def test_monday_holiday_after_the_weekend(self):
        """11/23（月）は祝日。11/20（金）の分は 11/21（土）の午後に出し、購入は 11/24（火）。"""
        fri = dt.date(2026, 11, 20)
        out = self.gate((2026, 11, 20, 18, 50), fri, dt.date(2026, 11, 19))
        self.assertEqual(out["run"], "false")
        self.assertIn("2026-11-24", out["reason"])                  # 次の営業日は火曜 11/24
        self.assertEqual(self.gate((2026, 11, 21, 13, 0), fri, dt.date(2026, 11, 19))["run"], "false")
        self.assertEqual(self.gate((2026, 11, 21, 15, 0), fri, dt.date(2026, 11, 19))["run"], "true")
        self.assertEqual(self.gate((2026, 11, 23, 18, 50), fri, fri)["run"], "false")   # 祝日の夜は新しい日足が無い

    def test_friday_holiday_moves_thursday_to_saturday_afternoon(self):
        cal = november(holidays=(dt.date(2026, 11, 3), dt.date(2026, 11, 23), dt.date(2026, 11, 27)))
        thu = dt.date(2026, 11, 26)
        out = self.gate((2026, 11, 26, 18, 50), thu, dt.date(2026, 11, 25), cal=cal)
        self.assertEqual(out["run"], "false")
        self.assertIn("2026-11-30", out["reason"])                  # 次の営業日は月曜 11/30
        self.assertIn("2026-11-28 土曜の 14:00 以降", out["reason"])
        self.assertEqual(self.gate((2026, 11, 28, 12, 0), thu, dt.date(2026, 11, 25), cal=cal)["run"], "false")
        self.assertEqual(self.gate((2026, 11, 28, 15, 0), thu, dt.date(2026, 11, 25), cal=cal)["run"], "true")

    def test_force_is_not_held(self):
        self.assertEqual(self.gate((2026, 11, 13, 18, 50), self.FRI, self.FRI, force="true")["run"], "true")

    def test_unknown_next_day_falls_back_to_running(self):
        """カレンダーが次の営業日を覆っていなければ、この規則は使わない（進む側に倒す）。"""
        cal = november(end=dt.date(2026, 11, 14))
        self.assertEqual(self.gate((2026, 11, 13, 18, 50), self.FRI, self.THU, cal=cal)["run"], "true")

    def test_long_holiday_uses_the_last_saturday(self):
        """連休で土曜が2つあれば、次の営業日に近いほうの土曜の 14:00 まで待つ。"""
        hol = tuple(dt.date(2026, 11, d) for d in (3, 16, 17, 18, 19, 20, 23))
        cal = november(holidays=hol)
        fri = dt.date(2026, 11, 13)                                 # 次の営業日は 11/24（火）
        self.assertEqual(self.gate((2026, 11, 14, 21, 0), fri, self.THU, cal=cal)["run"], "false")
        self.assertEqual(self.gate((2026, 11, 21, 13, 0), fri, self.THU, cal=cal)["run"], "false")
        self.assertEqual(self.gate((2026, 11, 21, 15, 0), fri, self.THU, cal=cal)["run"], "true")


class TestNextTradingDay(unittest.TestCase):
    def test_next_trading_day(self):
        cal = november()
        self.assertEqual(cal.next_trading_day(dt.date(2026, 11, 13)), dt.date(2026, 11, 16))
        self.assertEqual(cal.next_trading_day(dt.date(2026, 11, 20)), dt.date(2026, 11, 24))
        self.assertEqual(cal.next_trading_day(dt.date(2026, 11, 2)), dt.date(2026, 11, 4))
        self.assertIsNone(cal.next_trading_day(dt.date(2026, 11, 30)))   # 範囲の中に無い
        self.assertIsNone(cal.next_trading_day(dt.date(2026, 10, 30)))   # 範囲外


if __name__ == "__main__":
    unittest.main()
