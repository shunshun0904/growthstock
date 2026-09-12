#!/usr/bin/env python3
"""
research/label20.py の単体テスト（合成データのみ・ネットワークなし）。

  python3 tests/test_label20.py

2つの定義を同じ DataFrame に続けて掛けるので、間で列を消し忘れると
2回目が1回目を上書きする。そこを取り違えると「入れ替わった件数」が
丸ごと嘘になるので、消えていることを明示的に確かめる。
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402
import label20 as L  # noqa: E402


def panel(series, vol=2.0, code="10000", start="2020-01-06"):
    """終値の並びから、attach_rise_label が要る最小限の panel を作る。"""
    d = dt.date.fromisoformat(start)
    return pd.DataFrame({
        "Code": code,
        "Date": [pd.Timestamp(d + dt.timedelta(days=i)) for i in range(len(series))],
        "close": [float(v) for v in series],
        "high": [float(v) for v in series],
        "low": [float(v) for v in series],
        "vol": 100000.0,
        "vol_20d": float(vol),
    })


class TestConfigs(unittest.TestCase):
    def test_20d_keeps_the_persistence_conditions(self):
        """地平を縮めても、終盤とトレンドの条件は落としていない。"""
        self.assertEqual(L.CFG_20.horizon, 20)
        self.assertTrue(L.CFG_20.require_uptrend)
        self.assertIsNotNone(L.CFG_20.end_ratio)
        self.assertEqual(L.CFG_20.end_window, L.CFG_60.end_window)
        self.assertEqual(L.CFG_20.vol_norm_k, L.CFG_60.vol_norm_k)

    def test_20d_uses_a_faster_moving_average_pair(self):
        self.assertEqual((L.CFG_20.trend_short, L.CFG_20.trend_long), (5, 20))
        self.assertEqual((L.CFG_60.trend_short, L.CFG_60.trend_long), (20, 60))

    def test_threshold_scales_with_sqrt_horizon(self):
        n60, _ = B.rise_thresholds(pd.Series([3.0]), L.CFG_60)
        n20, _ = B.rise_thresholds(pd.Series([3.0]), L.CFG_20)
        self.assertAlmostEqual(float(n60.iloc[0]), 1.2 * 0.03 * np.sqrt(60), places=6)
        self.assertAlmostEqual(float(n20.iloc[0]), 1.2 * 0.03 * np.sqrt(20), places=6)
        self.assertLess(float(n20.iloc[0]), float(n60.iloc[0]))

    def test_end_level_is_half_of_the_threshold(self):
        for cfg in (L.CFG_60, L.CFG_20):
            need, end = B.rise_thresholds(pd.Series([3.0]), cfg)
            self.assertAlmostEqual(float(end.iloc[0]) / float(need.iloc[0]), 0.5,
                                   places=6)


class TestLabelsFor(unittest.TestCase):
    def test_removes_its_columns_so_the_next_config_is_clean(self):
        p = panel(list(np.linspace(100, 200, 120)))
        before = set(p.columns)
        m = np.zeros(len(p), dtype=bool)
        m[10] = True
        out = L.labels_for(p, L.CFG_60, m)
        self.assertEqual(set(p.columns), before, "入力に列が残っている")
        self.assertEqual(len(out), 1)
        self.assertIn("label", out.columns)

    def test_two_configs_give_independent_results(self):
        # 20日で急騰してから萎む形。20日では正例、60日では負例になりうる
        s = [100.0] * 30 + list(np.linspace(100, 145, 20)) + [110.0] * 70
        p = panel(s, vol=2.0)
        m = np.zeros(len(p), dtype=bool)
        m[29] = True
        d60 = L.labels_for(p, L.CFG_60, m)
        d20 = L.labels_for(p, L.CFG_20, m)
        self.assertNotEqual(float(d60["rise_need"].iloc[0]),
                            float(d20["rise_need"].iloc[0]))
        self.assertTrue(bool(d20["label"].iloc[0]))
        self.assertFalse(bool(d60["label"].iloc[0]))


class TestConditionsAndFunnel(unittest.TestCase):
    def frame(self, rise, need, end, end_need, trend, label):
        return pd.DataFrame({"future_rise": rise, "rise_need": need,
                             "end_level": end, "end_need": end_need,
                             "uptrend_end": trend, "label": label})

    def test_conditions_split(self):
        df = self.frame([0.3, 0.1], [0.2, 0.2], [0.15, 0.15],
                        [0.1, 0.1], [1.0, 0.0], [True, False])
        c = L.conditions(df)
        np.testing.assert_array_equal(c["reached"].to_numpy(), [True, False])
        np.testing.assert_array_equal(c["end_ok"].to_numpy(), [True, True])
        np.testing.assert_array_equal(c["trend_ok"].to_numpy(), [True, False])

    def test_funnel_counts_and_drop_reasons(self):
        df = self.frame(
            #  到達○終盤○トレ○  到達○終盤×    到達○終盤○トレ×  未到達
            [0.30, 0.30, 0.30, 0.05],
            [0.20, 0.20, 0.20, 0.20],
            [0.15, 0.02, 0.15, 0.00],
            [0.10, 0.10, 0.10, 0.10],
            [1.0, 1.0, 0.0, 0.0],
            [True, False, False, False])
        f = L.funnel(df, "x")
        self.assertEqual(f["n_determined"], 4)
        self.assertEqual(f["reached"], 3)
        self.assertEqual(f["reached_end"], 2)
        self.assertEqual(f["reached_end_trend"], 1)
        self.assertEqual(f["dropped_by_end"], 1)
        self.assertEqual(f["dropped_by_trend"], 1)
        self.assertEqual(f["positive"], 1)

    def test_undetermined_rows_are_excluded(self):
        df = self.frame([0.3, np.nan], [0.2, 0.2], [0.15, np.nan],
                        [0.1, 0.1], [1.0, np.nan], [True, np.nan])
        f = L.funnel(df, "x")
        self.assertEqual(f["n_determined"], 1)


class TestTransition(unittest.TestCase):
    def test_counts_each_cell(self):
        l60 = pd.Series([True, True, False, False])
        l20 = pd.Series([True, False, True, False])
        t = L.transition(l60, l20)
        self.assertEqual((t["pos_pos"], t["pos_neg"], t["neg_pos"], t["neg_neg"]),
                         (1, 1, 1, 1))
        self.assertEqual(t["flipped"], 2)
        self.assertEqual(t["n"], 4)

    def test_undetermined_rows_are_not_counted(self):
        l60 = pd.Series([True, np.nan, True])
        l20 = pd.Series([True, True, np.nan])
        t = L.transition(l60, l20)
        self.assertEqual(t["n"], 1)

    def test_masks_match_transition(self):
        l60 = pd.Series([True, True, False, False, np.nan])
        l20 = pd.Series([True, False, True, False, True])
        t = L.transition(l60, l20)
        m = L.masks(l60, l20)
        for k in ("pos_pos", "pos_neg", "neg_pos", "neg_neg"):
            self.assertEqual(int(m[k].sum()), t[k], k)
        # 判定不能の行はどの群にも入らない
        self.assertEqual(sum(int(m[k].sum()) for k in m), 4)


class TestBuildCase(unittest.TestCase):
    def test_two_horizon_blocks(self):
        s = [100.0] * 70 + list(np.linspace(100, 150, 25)) + [140.0] * 60
        p = panel(s, vol=2.0)
        c = L.build_case(p, "10000", p["Date"].iloc[69], "テスト", "pos_neg")
        self.assertIsNotNone(c)
        self.assertEqual(c["h20"]["horizon"], 20)
        self.assertEqual(c["h60"]["horizon"], 60)
        self.assertEqual(c["h20"]["trendPair"], [5, 20])
        self.assertEqual(c["h60"]["trendPair"], [20, 60])
        # 20日のしきい値のほうが低いので、目標価格も低い
        self.assertLess(c["h20"]["target"], c["h60"]["target"])
        self.assertEqual(len(c["close"]), len(c["dates"]))
        self.assertEqual(len(c["ma5"]), len(c["close"]))

    def test_missing_date_returns_none(self):
        p = panel([100.0] * 10)
        self.assertIsNone(
            L.build_case(p, "10000", pd.Timestamp("1990-01-01"), "", "pos_neg"))

    def test_label_matches_the_three_conditions(self):
        s = [100.0] * 70 + list(np.linspace(100, 150, 25)) + [140.0] * 60
        p = panel(s, vol=2.0)
        c = L.build_case(p, "10000", p["Date"].iloc[69], "", "pos_neg")
        for h in (c["h20"], c["h60"]):
            self.assertEqual(h["label"],
                             bool(h["reached"] and h["endOk"] and h["trendOk"]))


class TestPick(unittest.TestCase):
    def test_spreads_across_years(self):
        n = 400
        ev = pd.DataFrame({"Date": pd.to_datetime(
            ["2019-01-01"] * 200 + ["2020-01-01"] * 150 + ["2021-01-01"] * 50)})
        m = np.ones(n, dtype=bool)
        idx = L.pick(ev, m, 30, seed=1)
        years = pd.to_datetime(ev["Date"]).dt.year.to_numpy()[idx]
        self.assertEqual(len(set(years.tolist())), 3)
        self.assertLessEqual(len(idx), 30)

    def test_returns_all_when_fewer_than_requested(self):
        ev = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"] * 5)})
        idx = L.pick(ev, np.ones(5, dtype=bool), 30)
        self.assertEqual(len(idx), 5)

    def test_empty(self):
        ev = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"] * 5)})
        self.assertEqual(L.pick(ev, np.zeros(5, dtype=bool), 10), [])


class TestVolBands(unittest.TestCase):
    def test_bands_are_ordered_and_cover_all_rows(self):
        v = pd.Series(np.linspace(1.0, 10.0, 200))
        b = L.vol_bands(v, 5)
        self.assertEqual(len(b), 200)
        self.assertEqual(len(set(b)), 5)
        self.assertTrue(all(s[0].isdigit() for s in set(b)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
