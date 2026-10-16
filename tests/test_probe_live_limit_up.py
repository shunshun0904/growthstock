#!/usr/bin/env python3
"""
実運用の候補のストップ高を数える見るだけの集計（research/exp/probe_live_limit_up.py）。git も日足も使わず、表の形と数え方だけ確かめる。

  python3 tests/test_probe_live_limit_up.py
"""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import live_track as LT  # noqa: E402
import probe_live_limit_up as P  # noqa: E402


def frame():
    d = pd.Timestamp("2026-09-16")
    return pd.DataFrame({
        "Date": [d, d, d, d + pd.Timedelta(days=1)], "Code": ["75160", "84180", "90820", "10010"],
        "code": ["7516", "8418", "9082", "1001"], "name": ["甲", "乙", "丙", "丁"], "marketCap": [1289.0, 6572.0, 110.0, np.nan],
        "pass_PREV": [True, True, False, False], "rank_PREV": [1.0, 2.0, np.nan, np.nan], "pmin_PREV": [96.5, 95.4, 8.9, 50.0],
        "pass_K15": [False, False, False, False], "rank_K15": [np.nan] * 4, "pmin_K15": [np.nan] * 4,
        "ul_fwd": [0.0, 0.0, 1.0, np.nan], "ul_fwd_first": [np.nan, np.nan, 6.0, np.nan], "ulc_fwd": [0.0, 0.0, 1.0, np.nan],
        "fwd_ok": [0.0, 0.0, 1.0, 0.0], "fwd_days": [17.0, 17.0, 20.0, np.nan],
        "status": ["保有中", LT.HIT, LT.HIT, "日足なし"], "hit_day": [np.nan, 5.0, 6.0, np.nan], "ret": [np.nan, 10.0, 10.0, np.nan],
        "ret_now": [0.0, np.nan, np.nan, np.nan], "elapsed": [17.0, np.nan, np.nan, np.nan]})


class TestProbe(unittest.TestCase):
    def test_cap_band(self):
        self.assertEqual(P.cap_band([50, 100, 299, 300, 999, 1000, 2999, 3000]).tolist(), [0, 1, 1, 2, 2, 3, 3, 4])
        self.assertTrue(np.isnan(P.cap_band([np.nan]).iloc[0]))

    def test_market_caps_last_version_wins(self):
        versions = [{"data": {"candidates": [{"date": "2026-09-16", "jqCode": "75160", "marketCap": 1200}]}},
                    {"data": {"candidates": [{"date": "2026-09-16", "jqCode": "75160", "marketCap": 1289}]}}]
        mc = P.market_caps(versions)
        self.assertEqual(len(mc), 1)
        self.assertEqual(mc.iloc[0]["marketCap"], 1289)

    def test_summary_and_lines(self):
        g = frame()
        g["cap_band"] = P.cap_band(g["marketCap"]).to_numpy()
        s = P.summary(g, "候補全体")
        self.assertIn("4件", s)
        self.assertIn("判定できた 3件（20営業日そろった 1件）", s)
        self.assertIn("ストップ高 1件（33.3%。張り付き 1）", s)
        self.assertIn("+10% 到達 2件", s)
        self.assertIn("100〜300億 1/1", s)
        line = P.line_of({**g.iloc[2].to_dict(), "pmin": 8.9})
        self.assertIn("ストップ高 6日目（張り付き）", line)
        self.assertIn("+10% 到達 6日目", line)
        line0 = P.line_of({**g.iloc[0].to_dict(), "pmin": 96.5})
        self.assertIn("なし", line0)
        self.assertIn("途中 +0.0%", line0)
        line3 = P.line_of({**g.iloc[3].to_dict(), "pmin": 50.0})
        self.assertIn("日足なし", line3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.report(g, "2026-09-11")
        out = buf.getvalue()
        for tag in ("■ 実運用の候補", "以前の規則", "今の規則", "■ 規則を満たした候補", "■ 候補全体のうち、買ってからストップ高になった 1件"):
            self.assertIn(tag, out)
        self.assertIn("[旧・上位1]", out)
        with contextlib.redirect_stdout(io.StringIO()):
            P.report(g.iloc[0:0], "2026-09-11")          # 候補が無くても落ちない


if __name__ == "__main__":
    unittest.main()
