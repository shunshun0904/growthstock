#!/usr/bin/env python3
"""
research/probe_update_time.py の打ち切りの単体テスト（ネットワークには出ない）。

2026-09-25 の run 36099557178 は、起動 14:40 JST・--until 20:30 で、ジョブの上限
（350分、20:30:43）にちょうど殺され、途中の観測も集計も残らなかった。
起動から決めた分数で自分で打ち切ること、最後の1回を終わりの時刻に見ることを固定する。
"""
import datetime as dt
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import probe_update_time as P  # noqa: E402

JST = P.JST


def at(h, m, s=0):
    return dt.datetime(2026, 9, 25, h, m, s, tzinfo=JST)


class TestDeadline(unittest.TestCase):
    def test_run_of_2026_09_25_would_stop_before_the_job_limit(self):
        started = at(14, 40, 43)
        job_killed = started + dt.timedelta(minutes=350)
        u = P.effective_until(at(20, 30), started, 335)
        self.assertLess(u, job_killed)
        self.assertEqual(u, started + dt.timedelta(minutes=335))

    def test_until_is_kept_when_it_comes_first(self):
        self.assertEqual(P.effective_until(at(18, 0), at(14, 40), 335), at(18, 0))

    def test_no_cap_without_max_minutes(self):
        self.assertEqual(P.effective_until(at(20, 30), at(14, 40), None), at(20, 30))

    def test_sleep_does_not_overshoot_until(self):
        self.assertEqual(P.sleep_seconds(at(20, 20), at(20, 30), 1800), 600.0)
        self.assertEqual(P.sleep_seconds(at(16, 0), at(20, 30), 1800), 1800.0)
        self.assertEqual(P.sleep_seconds(at(20, 31), at(20, 30), 1800), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
