#!/usr/bin/env python3
"""
保存データと API の応答の行数を比べる検査（research/check_store_completeness.py）。

固定すること
  - 保存データが API より少なければ exit 1（2026-09-24 の「1行に潰していた」を捕まえる）
  - 同じ行数なら exit 0。まったく同じ行の重なりは1行と数える
  - 直近（公表のラグの中）の日は選ばない

  python3 tests/test_store_completeness.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import pandas as pd  # noqa: E402

import check_store_completeness as C  # noqa: E402
import trading_calendar as TC  # noqa: E402


def weekdays(start, end):
    out, d = [], dt.date.fromisoformat(start)
    while d <= dt.date.fromisoformat(end):
        if d.weekday() < 5:
            out.append({"Date": d.isoformat(), "HolDiv": "1"})
        d += dt.timedelta(days=1)
    return out


class StoreCompleteness(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        today = dt.datetime.now(C.J.JST).date()
        TC.save(weekdays((today - dt.timedelta(days=800)).isoformat(), today.isoformat()),
                self.dir)
        self.api_rows = 3
        test = self

        class Fake:
            def __init__(self, *a, **k):
                pass

            def get_paginated(self, path, params):
                day = list(params.values())[0]
                return [{"Date": day, "S33": f"{i:04d}", "SellExShortVa": float(i)}
                        for i in range(test.api_rows)]

        self.orig = (C.J.JQuantsClient, C.J.resolve_api_key)
        C.J.JQuantsClient = Fake
        C.J.resolve_api_key = lambda: "x"

    def tearDown(self):
        C.J.JQuantsClient, C.J.resolve_api_key = self.orig
        shutil.rmtree(self.dir, ignore_errors=True)

    def store(self, rows_per_day):
        """選ばれうる日すべてに、1日 rows_per_day 行を保存する。"""
        cal = TC.load(self.dir)
        frames = []
        for d in cal.days:
            frames.append(pd.DataFrame({"Date": [pd.Timestamp(d)] * rows_per_day,
                                        "S33": [f"{i:04d}" for i in range(rows_per_day)],
                                        "SellExShortVa": [float(i) for i in range(rows_per_day)]}))
        allrows = pd.concat(frames, ignore_index=True)
        for y, g in allrows.groupby(allrows["Date"].dt.year):
            g.to_parquet(os.path.join(self.dir, f"shortratio_{y}.parquet"), index=False)

    def run_check(self):
        return C.main(["--data-dir", self.dir, "--kinds", "shortratio", "--days", "3",
                       "--seed", "1"])

    def test_collapsed_store_fails(self):
        self.store(1)                     # 1日3行のはずが1行（2026-09-24 の状態）
        self.assertEqual(self.run_check(), 1)

    def test_complete_store_passes(self):
        self.store(3)
        self.assertEqual(self.run_check(), 0)

    def test_recent_days_are_not_picked(self):
        today = dt.date(2026, 9, 24)
        days = [today - dt.timedelta(days=i) for i in range(0, 60)]
        picked = C.pick_days(days, 50, 0, today)
        self.assertTrue(all((today - d).days >= C.RECENT_DAYS for d in picked))

    def test_identical_rows_count_once(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        self.assertEqual(C.distinct_rows(df), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
