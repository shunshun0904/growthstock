#!/usr/bin/env python3
"""
実験77（research/exp/e77_label_distribution.py）: 目的変数の定義どおりに上昇幅の分布を出す集計。

- vol_20d から引き直した必要な上昇率（1.2σ）・終盤の水準（0.6σ）は build_dataset.rise_thresholds と同じ
- 引き直したラベル = 到達 かつ 終盤 かつ トレンド。本番のラベルと一致すれば不一致 0
- 群の分け方（全体 / 正例 / 負例 / 到達したが負例）と分位の表、ファネルの件数

  python3 tests/test_e77_label_distribution.py
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e77_label_distribution as E  # noqa: E402


def frame():
    # vol 1%/日 → σ = 0.01 × √20 = 4.47% → 必要 5.37%、終盤 2.68%
    rows = [
        # future_rise, end_level, uptrend, 期待ラベル
        (0.10, 0.05, 1.0, 1),      # 到達・終盤・トレンド → 正例
        (0.10, 0.01, 1.0, 0),      # 到達したが終盤で落ちる
        (0.10, 0.05, 0.0, 0),      # 到達・終盤だがトレンドで落ちる
        (0.03, 0.02, 1.0, 0),      # 届かない
        (0.20, 0.15, 1.0, 1),      # 正例
        (-0.05, -0.08, 0.0, 0),    # 下げ
    ]
    df = pd.DataFrame(rows, columns=["future_rise", "end_level", "uptrend_end", "label"])
    df["vol_20d"] = 1.0
    df["Code"] = [f"{i:04d}0" for i in range(len(df))]
    df["Date"] = pd.to_datetime(["2024-01-05"] * 3 + ["2025-01-07"] * 3)
    df["ret_o1_20"] = df["end_level"]
    df["ret_o1_40"] = df["end_level"] * 1.5
    df["entry_gap"] = 0.01
    return df


class TestRederive(unittest.TestCase):
    def test_thresholds_and_label(self):
        d = E.rederive(frame())
        need, end_need = B.rise_thresholds(pd.Series([1.0]), B.DEFAULT_RISE)
        self.assertAlmostEqual(d["rise_need"].iloc[0], float(need.iloc[0]))
        self.assertAlmostEqual(d["rise_need"].iloc[0], 1.2 * 0.01 * np.sqrt(20))
        self.assertAlmostEqual(d["end_need"].iloc[0], float(end_need.iloc[0]))
        self.assertAlmostEqual(d["end_need"].iloc[0], 0.5 * d["rise_need"].iloc[0])
        self.assertEqual(list(d["label_re"]), [1, 0, 0, 0, 1, 0])
        self.assertEqual(list(d["hit"]), [True, True, True, False, True, False])
        f = E.funnel(d)
        self.assertEqual(f, {"n": 6, "hit": 4, "hit_end": 3, "hit_end_trend": 2, "pos": 2,
                             "hit_fail_end": 1, "hit_end_fail_trend": 1, "mismatch": 0})

    def test_tables_and_groups(self):
        d = E.rederive(frame())
        t = E.table(d, "future_rise", 100.0)
        self.assertEqual({g for g, _ in E.GROUPS}, set(t))
        self.assertEqual(t["all"]["n"], 6)
        self.assertEqual(t["pos"]["n"], 2)
        self.assertEqual(t["neg"]["n"], 4)
        self.assertEqual(t["hit_neg"]["n"], 2)
        self.assertAlmostEqual(t["pos"]["p50"], 15.0)                      # 10% と 20% の中央
        self.assertAlmostEqual(t["hit_neg"]["mean"], 10.0)
        self.assertAlmostEqual(t["all"]["p5"], float(pd.Series([10, 10, 10, 3, 20, -5]).quantile(0.05)))
        self.assertEqual(E.stats(pd.Series([np.nan, np.nan]), 100.0), {"n": 0})
        yr = E.by_year(d)
        self.assertEqual(list(yr.index), [2024, 2025])
        self.assertAlmostEqual(yr.loc[2024, "pos_rate"], 1 / 3)
        self.assertAlmostEqual(yr.loc[2025, "need_median"], 1.2 * 0.01 * np.sqrt(20) * 100)

    def test_report_runs(self):
        d = E.rederive(frame())
        with mock.patch.object(E, "OUT", os.path.join(os.path.dirname(E.OUT), "e77_test.json")):
            with mock.patch("builtins.print"):
                s = E.report(d)
            os.remove(E.OUT)
        self.assertEqual(s["funnel"]["pos"], 2)
        self.assertIn("rise_need", s["tables"])
        self.assertIn("2024", s["by_year"])


if __name__ == "__main__":
    unittest.main()
