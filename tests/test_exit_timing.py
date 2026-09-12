#!/usr/bin/env python3
"""
research/exit_timing.py の単体テスト（合成データのみ・ネットワークなし）。

  python3 tests/test_exit_timing.py

先読みの並べ方を1つ間違えると「何日目に売るのが良いか」の答えが
まるごと入れ替わる。銘柄の境目・到達判定の当日・末尾の扱いを個別に固定する。
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import exit_timing as X  # noqa: E402


def panel(closes_by_code, start="2020-01-06"):
    """{code: [終値...]} から、_pos を振った整列済み panel を作る。"""
    d = dt.date.fromisoformat(start)
    recs = []
    for code, closes in closes_by_code.items():
        for i, c in enumerate(closes):
            recs.append({"Code": code,
                         "Date": pd.Timestamp(d + dt.timedelta(days=i)),
                         "close": c})
    p = pd.DataFrame(recs).sort_values(["Code", "Date"]).reset_index(drop=True)
    p["_pos"] = np.arange(len(p))
    return p


def events(p, picks):
    """picks: [(code, index)] をイベント行として取り出す。"""
    rows = []
    for code, i in picks:
        sub = p[p["Code"] == code].reset_index(drop=True)
        rows.append(sub.iloc[i])
    return pd.DataFrame(rows).reset_index(drop=True)


class TestForwardMatrix(unittest.TestCase):
    def test_aligns_to_event_day(self):
        p = panel({"A": [100, 110, 120, 130, 140]})
        ev = events(p, [("A", 1)])
        F = X.forward_matrix(p, ev, days=3)
        np.testing.assert_allclose(F[0], [110, 120, 130, 140])

    def test_does_not_cross_codes(self):
        p = panel({"A": [100, 110], "B": [999, 998, 997]})
        ev = events(p, [("A", 0)])
        F = X.forward_matrix(p, ev, days=3)
        np.testing.assert_allclose(F[0][:2], [100, 110])
        self.assertTrue(np.isnan(F[0][2]))
        self.assertTrue(np.isnan(F[0][3]))

    def test_multiple_events(self):
        p = panel({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40]})
        ev = events(p, [("A", 0), ("B", 1)])
        F = X.forward_matrix(p, ev, days=2)
        np.testing.assert_allclose(F[0], [1, 2, 3])
        np.testing.assert_allclose(F[1], [20, 30, 40])

    def test_requires_pos(self):
        p = panel({"A": [1, 2]}).drop(columns=["_pos"])
        ev = events(p, [("A", 0)])
        with self.assertRaises(SystemExit):
            X.forward_matrix(p, ev, days=1)


class TestFixedHorizon(unittest.TestCase):
    def test_return_and_fee(self):
        F = np.array([[100.0, 110.0, 120.0]])
        entry = np.array([100.0])
        rows = X.fixed_horizon(F, entry, fee_pct=0.0, horizons=(1, 2))
        self.assertAlmostEqual(rows[0]["mean"], 10.0)
        self.assertAlmostEqual(rows[1]["mean"], 20.0)
        paid = X.fixed_horizon(F, entry, fee_pct=0.5, horizons=(1,))
        self.assertAlmostEqual(paid[0]["mean"], 9.5)

    def test_hold_days_and_per_month(self):
        F = np.array([[100.0] + [110.0] * 40])
        entry = np.array([100.0])
        rows = X.fixed_horizon(F, entry, fee_pct=0.0, horizons=(10, 40))
        by = {r["k"]: r for r in rows}
        self.assertAlmostEqual(by[10]["hold_days"], 10)
        # 10日で+10% -> 1ヶ月(20日)換算で+20%
        self.assertAlmostEqual(by[10]["per_month"], 20.0)
        self.assertAlmostEqual(by[40]["per_month"], 5.0)

    def test_skips_horizons_beyond_matrix(self):
        F = np.array([[100.0, 101.0]])
        rows = X.fixed_horizon(F, np.array([100.0]), 0.0, horizons=(1, 5, 60))
        self.assertEqual([r["k"] for r in rows], [1])

    def test_nan_rows_are_dropped_not_zeroed(self):
        F = np.array([[100.0, 110.0], [100.0, np.nan]])
        rows = X.fixed_horizon(F, np.array([100.0, 100.0]), 0.0, horizons=(1,))
        self.assertEqual(rows[0]["n"], 1)
        self.assertAlmostEqual(rows[0]["mean"], 10.0)


class TestTargetExit(unittest.TestCase):
    def test_sells_on_first_day_at_or_above_level(self):
        # 基準100・need=0.10 -> 110 以上になる最初の日
        F = np.array([[100.0, 105.0, 112.0, 130.0, 90.0]])
        r = X.target_exit(F, np.array([100.0]), np.array([0.10]),
                          horizon=4, fee_pct=0.0)
        self.assertAlmostEqual(r["mean"], 12.0)     # 112 で売る
        self.assertAlmostEqual(r["hold_days"], 2)
        self.assertAlmostEqual(r["hit_rate"], 100.0)

    def test_exact_level_counts_as_hit(self):
        F = np.array([[100.0, 110.0, 130.0]])
        r = X.target_exit(F, np.array([100.0]), np.array([0.10]),
                          horizon=2, fee_pct=0.0)
        self.assertAlmostEqual(r["mean"], 10.0)

    def test_falls_back_to_horizon_close(self):
        F = np.array([[100.0, 101.0, 102.0, 103.0]])
        r = X.target_exit(F, np.array([100.0]), np.array([0.50]),
                          horizon=3, fee_pct=0.0)
        self.assertAlmostEqual(r["mean"], 3.0)      # 3日目の終値
        self.assertAlmostEqual(r["hold_days"], 3)
        self.assertAlmostEqual(r["hit_rate"], 0.0)

    def test_ratio_halves_the_level(self):
        F = np.array([[100.0, 106.0, 120.0]])
        full = X.target_exit(F, np.array([100.0]), np.array([0.10]),
                             horizon=2, fee_pct=0.0, ratio=1.0)
        half = X.target_exit(F, np.array([100.0]), np.array([0.10]),
                             horizon=2, fee_pct=0.0, ratio=0.5)
        self.assertAlmostEqual(full["mean"], 20.0)  # 110 に届くのは2日目(120)
        self.assertAlmostEqual(half["mean"], 6.0)   # 105 に届くのは1日目(106)

    def test_threshold_is_anchored_on_base_close_not_entry(self):
        # 翌日に窓を開けて寄っても、しきい値は基準日終値から測る
        F = np.array([[100.0, 118.0]])
        r = X.target_exit(F, np.array([115.0]), np.array([0.10]),
                          horizon=1, fee_pct=0.0)
        self.assertAlmostEqual(r["hit_rate"], 100.0)
        self.assertAlmostEqual(r["mean"], (118 / 115 - 1) * 100, places=2)


class TestPeakProfile(unittest.TestCase):
    def test_finds_day_and_level_of_max(self):
        F = np.array([[100.0, 105.0, 130.0, 110.0]])
        p = X.peak_profile(F, np.array([100.0]), upto=3, fee_pct=0.0)
        self.assertAlmostEqual(p["peak_mean"], 30.0)
        self.assertAlmostEqual(p["day_median"], 2.0)

    def test_ignores_nan(self):
        F = np.array([[100.0, np.nan, 120.0]])
        p = X.peak_profile(F, np.array([100.0]), upto=2, fee_pct=0.0)
        self.assertAlmostEqual(p["peak_mean"], 20.0)
        self.assertAlmostEqual(p["day_median"], 2.0)

    def test_histogram_buckets(self):
        F = np.array([[100.0] + [100.0 + i for i in range(1, 25)]])
        p = X.peak_profile(F, np.array([100.0]), upto=24, fee_pct=0.0)
        self.assertEqual(p["day_median"], 24.0)
        total = sum(b["n"] for b in p["day_hist"])
        self.assertEqual(total, 1)


class TestSubsets(unittest.TestCase):
    def make(self):
        return pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "score": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8],
            "label": [0, 0, 1, 0, 1, 1],
            "vol_20d": [2.0] * 6,
        })

    def test_top_n_is_per_day(self):
        ev = self.make()
        by = {m["name"]: m for m in X.subset_masks(ev, quantiles=(0.5,),
                                                   top_n=(1, 2), min_scored=1)}
        m1 = by["その日の上位1件"]["mask"]
        self.assertEqual(m1.sum(), 2)                    # 各日1件
        np.testing.assert_array_equal(ev["score"][m1].to_numpy(), [0.9, 0.8])
        m2 = by["その日の上位2件"]["mask"]
        self.assertEqual(m2.sum(), 4)

    def test_threshold_is_global(self):
        ev = self.make()
        masks = X.subset_masks(ev, quantiles=(0.5,), top_n=(), min_scored=1)
        thr = [m for m in masks if m["kind"] == "threshold"][0]
        # 中央値は 0.45。0.5/0.8/0.9 の3件が残る
        self.assertEqual(thr["mask"].sum(), 3)

    def test_top_n_can_include_low_scores(self):
        """その日の1位でも、全体で見れば低いことがある（これが差の主因の候補）。"""
        ev = self.make()
        ev.loc[3:, "score"] = [0.01, 0.02, 0.03]        # 2日目は全部弱い
        by = {m["name"]: m for m in X.subset_masks(ev, quantiles=(0.5,),
                                                   top_n=(1,), min_scored=1)}
        picked = ev["score"][by["その日の上位1件"]["mask"]].to_numpy()
        self.assertIn(0.03, picked)                      # 弱い日の1位も入る

    def test_no_score_column(self):
        ev = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"])})
        masks = X.subset_masks(ev)
        self.assertEqual([m["name"] for m in masks], ["全件"])

    def test_too_few_scored_rows_are_not_split(self):
        """数十件で「上位10%」を作っても読めないので、既定では分けない。"""
        ev = self.make()
        self.assertEqual([m["name"] for m in X.subset_masks(ev)], ["全件"])

    def test_profile_counts_and_label_rate(self):
        ev = self.make()
        prof = X.subset_profile(ev, np.ones(len(ev), dtype=bool))
        self.assertEqual(prof["n"], 6)
        self.assertAlmostEqual(prof["label_rate"], 50.0)
        self.assertAlmostEqual(prof["score_mean"], 0.4833, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
