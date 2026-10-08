#!/usr/bin/env python3
"""
実験66（research/exp/e66_label_strict.py）のラベルの作り方と集計の小道具のテスト。重い計算は回さない。

  - 固定しきい値のラベルは「翌営業日の寄り」を分母に、n 日以内の終値の最大で判定する
  - 先の足が n 日ぶん無い行は NaN（未確定）。False にはしない
  - ボラの帯は5等分、窓の中の上位10% は件数をそろえる、日付内1位は候補5件以上の日だけ

  python3 tests/test_e66_label_strict.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e66_label_strict as E  # noqa: E402

DAYS = list(pd.bdate_range("2024-01-01", periods=80))


def bars_of(code, closes, start=0):
    n = len(closes)
    return pd.DataFrame({"Date": DAYS[start:start + n], "Code": code,
                         "AdjO": closes, "AdjH": [c * 1.01 for c in closes],
                         "AdjL": [c * 0.99 for c in closes], "AdjC": closes})


class StrictLabels(unittest.TestCase):
    def test_threshold_on_close_from_next_open(self):
        # 基準日 t。買値は t+1 の始値（=100）。t+32〜36 に終値 125、t+42 以降に 131
        closes = [90.0, 100.0] + [100.0] * 30 + [125.0] * 5 + [100.0] * 5 + [131.0] * 30
        bars = bars_of("A", closes).sort_values(["Code", "Date"]).reset_index(drop=True)
        df = pd.DataFrame({"Code": ["A"], "Date": [DAYS[0]]})
        out = E.strict_labels(df, bars, k=60)
        self.assertAlmostEqual(out["entry_fwd"].iloc[0], 100.0)
        # t+1〜t+40 の終値の最大は 125（131 は t+42 で範囲外）→ +20% は届く、+30% は届かない
        self.assertEqual(out["y_L20_40"].iloc[0], 1.0)
        self.assertEqual(out["y_L30_40"].iloc[0], 0.0)
        # 60日以内なら 131 → +30% に届く
        self.assertEqual(out["y_L30_60"].iloc[0], 1.0)
        self.assertAlmostEqual(out["rise_40"].iloc[0], 0.25)

    def test_unknown_when_future_is_short(self):
        closes = [100.0] * 30
        bars = bars_of("A", closes).sort_values(["Code", "Date"]).reset_index(drop=True)
        df = pd.DataFrame({"Code": ["A"], "Date": [DAYS[0]]})
        out = E.strict_labels(df, bars, k=60)
        self.assertTrue(np.isnan(out["y_L30_40"].iloc[0]))
        self.assertTrue(np.isnan(out["y_L30_60"].iloc[0]))


class Helpers(unittest.TestCase):
    def test_vol_band_is_quintile(self):
        v = pd.Series(np.arange(100, dtype=float))
        b = E.vol_band(v)
        self.assertEqual(sorted(b.unique().tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(int(b.iloc[0]), 0)
        self.assertEqual(int(b.iloc[-1]), 4)
        self.assertTrue(np.isnan(E.vol_band(pd.Series([1.0, np.nan])).iloc[1]))

    def test_top_in_fold_keeps_share_per_fold(self):
        o = pd.DataFrame({"fold": [1] * 50 + [2] * 20, "score": np.arange(70, dtype=float)})
        t = E.top_in_fold(o, 0.10)
        self.assertEqual((t["fold"] == 1).sum(), 5)
        self.assertEqual((t["fold"] == 2).sum(), 2)
        self.assertEqual(t.loc[t["fold"] == 1, "score"].min(), 45)

    def test_pick1_needs_min_candidates(self):
        o = pd.DataFrame({"Date": [DAYS[0]] * 5 + [DAYS[1]] * 3, "score": [1, 5, 3, 2, 4, 9, 8, 7]})
        p = E.pick1(o, min_day=5)
        self.assertEqual(len(p), 1)
        self.assertEqual(int(p["score"].iloc[0]), 5)


if __name__ == "__main__":
    unittest.main()
