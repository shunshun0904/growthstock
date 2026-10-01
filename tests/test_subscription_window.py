#!/usr/bin/env python3
"""
J-Quants の契約の範囲（今日の10年前から。毎日1日ずつ後ろへずれる）に合わせて、
取り込みの問い合わせの開始日を切り上げることのテスト（research/jq_bulk.py）。

2026-10-02 0:00 JST から /markets/calendar?from=2016-10-01 が HTTP 400
（"Your subscription covers the following dates: 2016-10-02 ~"）を返し、
取り込みが全種別で落ちた（run 36879424717）。

- subscription_start は JST の今日の10年前 + 1日（2/29 は 3/1 に寄せる）
- clamp_start は範囲より前だけを切り上げ、保存済みの最古日（2016-10-01）より前にはしない
- covered_from は J-Quants の 400 の文面から「この日から」を読む
- trading_days は 400 が返ったら、その日から取り直す（文面が無い失敗はそのまま上げる）

  python3 tests/test_subscription_window.py
"""
import datetime as dt
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import jq_bulk as J  # noqa: E402
from jquants_data_fetcher import JQuantsError  # noqa: E402

#: 2026-10-02 0:00 JST 過ぎに実際に返った文面（run 36879424717 のログ）
MSG = ("HTTP 400 https://api.jquants.com/v2/markets/calendar?from=2016-10-01&to=2026-11-15 : "
       "{\"message\": \"Your subscription covers the following dates: 2016-10-02 ~ . "
       "If you want more data, please\"}")


class TestWindow(unittest.TestCase):
    def test_subscription_start(self):
        self.assertEqual(J.subscription_start(dt.date(2026, 10, 2)), dt.date(2016, 10, 3))
        self.assertEqual(J.subscription_start(dt.date(2026, 10, 2), margin_days=0),
                         dt.date(2016, 10, 2))
        # 2/29 の10年前は無いので 3/1 に寄せてから余裕の1日
        self.assertEqual(J.subscription_start(dt.date(2028, 2, 29)), dt.date(2018, 3, 2))

    def test_clamp_start(self):
        today = dt.date(2026, 10, 2)
        self.assertEqual(J.clamp_start(dt.date(2016, 10, 1), today), dt.date(2016, 10, 3))
        self.assertEqual(J.clamp_start(dt.date(2020, 1, 1), today), dt.date(2020, 1, 1))
        # 範囲がまだ保存済みの最古日より前なら、最古日で止める
        self.assertEqual(J.clamp_start(dt.date(2016, 1, 1), dt.date(2026, 9, 20)),
                         J.EARLIEST_DATE)

    def test_covered_from(self):
        self.assertEqual(J.covered_from(JQuantsError(MSG)), dt.date(2016, 10, 2))
        self.assertIsNone(J.covered_from(JQuantsError("HTTP 503 temporarily unavailable")))


class FakeClient:
    """契約の範囲より前を含む問い合わせには 400 を返す。"""

    def __init__(self, first_ok: dt.date, msg: str = MSG):
        self.first_ok = first_ok
        self.msg = msg
        self.calls = []

    def get_paginated(self, path, params):
        self.calls.append((path, dict(params)))
        start = dt.date.fromisoformat(params["from"])
        if start < self.first_ok:
            raise JQuantsError(self.msg)
        rows = []
        d = start
        end = dt.date.fromisoformat(params["to"])
        while d <= end:
            rows.append({"Date": d.isoformat(), "HolDiv": "1" if d.weekday() < 5 else "0"})
            d += dt.timedelta(days=1)
        return rows


class TestTradingDays(unittest.TestCase):
    def test_retries_from_the_date_jquants_names(self):
        client = FakeClient(first_ok=dt.date(2016, 10, 2))
        with tempfile.TemporaryDirectory() as d:
            days = J.trading_days(client, dt.date(2016, 10, 1), dt.date(2016, 10, 31), d)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1][1]["from"], "2016-10-02")
        self.assertEqual(days[0], dt.date(2016, 10, 3))      # 10/2 は日曜
        self.assertTrue(all(x <= dt.date(2016, 10, 31) for x in days))

    def test_other_errors_are_not_swallowed(self):
        client = FakeClient(first_ok=dt.date(2016, 10, 2), msg="HTTP 500 server error")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(JQuantsError):
                J.trading_days(client, dt.date(2016, 10, 1), dt.date(2016, 10, 31), d)
        self.assertEqual(len(client.calls), 1)

    def test_in_range_start_is_one_call(self):
        client = FakeClient(first_ok=dt.date(2016, 10, 2))
        with tempfile.TemporaryDirectory() as d:
            J.trading_days(client, dt.date(2016, 10, 3), dt.date(2016, 10, 31), d)
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
