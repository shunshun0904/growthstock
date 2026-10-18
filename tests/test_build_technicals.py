#!/usr/bin/env python3
"""
build_dataset.attach_technicals（テクニカル指標111列を候補の行に付ける。本番導入の第1段階 2026-10-18、
docs/MODEL_ADOPTION_RULES.md §68）。データには触らない（作りものの日足）。

- 値は technicals.compute（日足の全行で数えたもの）の同じ行と一致する
- 行の並び・index は samples のまま（並べ替えた・飛び飛びの index でも位置で付く）。日足に無い行は欠測
- 銘柄を細かく分けて数えても、値は1ビットも変わらない（TECH_CODES_PER_CHUNK。移動和・分散は銘柄ごとに数える。
  technicals._Panel.roll）。株価の桁が大きく違う銘柄が前に並び、値動きの無い期間があっても同じ
- 同じ名前の列がすでにあれば止まる。日足に要る列が無ければ止まる。売買代金が無くても動く（終値 × 出来高）
- 360列の組（249列 + 111列）は既定ではない（第1段階）。データセットに作る列（features.all_columns）に111列が入る

  python3 tests/test_build_technicals.py
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import technicals as TX  # noqa: E402


def bars(code, n, seed):
    rng = np.random.default_rng(seed)
    close = 500 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    o = close * np.exp(rng.normal(0, 0.01, n))
    return pd.DataFrame({"Code": code, "Date": pd.bdate_range("2023-01-02", periods=n), "open": o,
                         "high": np.maximum(o, close) * 1.01, "low": np.minimum(o, close) * 0.99, "close": close,
                         "vol": rng.integers(1000, 50000, n).astype(float), "trading_value": close * 1000 / 1e8})


class TestAttach(unittest.TestCase):
    def setUp(self):
        p = pd.concat([bars("10010", 300, 0), bars("10020", 120, 1), bars("10030", 260, 2)], ignore_index=True)
        p["Date"] = p["Date"].astype("datetime64[ns]")
        p["extra"] = 1.0                                                          # 日足の他の列は使わない
        self.p = p
        self.full = TX.compute(p)
        pick = [290, 5, 410, 130, 299, 650, 300]
        s = p.iloc[pick][["Code", "Date"]].copy()
        s.loc[len(p) + 7] = ["99990", pd.Timestamp("2023-06-01")]                 # 日足に無い銘柄
        s.index = [70, 3, 55, 12, 9, 41, 8, 99]                                   # 飛び飛びの index
        s["keep"] = np.arange(len(s), dtype=float)
        self.s, self.pick = s, pick

    def check_values(self, got, rtol=1e-12, atol=0.0):
        self.assertEqual(list(got.index), list(self.s.index))
        self.assertEqual(list(got.columns), list(self.s.columns) + TX.COLUMNS)
        self.assertEqual(got["keep"].tolist(), self.s["keep"].tolist())
        for i, row in enumerate(self.pick):
            want = self.full.iloc[row].to_numpy(dtype=float)
            have = got[TX.COLUMNS].iloc[i].to_numpy(dtype=float)
            self.assertTrue(np.array_equal(np.isnan(want), np.isnan(have)))
            self.assertTrue(np.allclose(want[~np.isnan(want)], have[~np.isnan(have)], rtol=rtol, atol=atol))
        self.assertTrue(got[TX.COLUMNS].iloc[-1].isna().all())                     # 日足に無い行は欠測

    def test_values_and_order(self):
        self.check_values(B.attach_technicals(self.s, self.p))

    def test_chunks(self):
        with mock.patch.object(B, "TECH_CODES_PER_CHUNK", 1):
            self.check_values(B.attach_technicals(self.s, self.p), rtol=0.0, atol=0.0)

    def test_chunks_exact_with_flat_prices(self):
        # 値の桁が大きい銘柄の後ろに、値動きの無い期間がある安い銘柄（標準偏差 0）。銘柄の組で値が変わらない
        big = bars("10010", 300, 0)
        big[["open", "high", "low", "close"]] *= 1000.0
        flat = bars("10020", 120, 1)
        flat.loc[40:80, ["open", "high", "low", "close"]] = 123.0
        p = pd.concat([big, flat], ignore_index=True)
        p["Date"] = p["Date"].astype("datetime64[ns]")
        keys = p[["Code", "Date"]]
        together = TX.compute_at(p, keys, codes_per_chunk=2)
        alone = TX.compute_at(p, keys, codes_per_chunk=1)
        self.assertTrue(np.array_equal(together.to_numpy(), alone.to_numpy(), equal_nan=True))
        i = 300 + 70                                                              # 値動きの無い期間の中
        self.assertEqual(together.loc[i, "tx_bb_width"], 0.0)

    def test_without_trading_value(self):
        p = self.p.drop(columns=["trading_value"])
        got = B.attach_technicals(self.s, p)
        want = TX.compute(p)
        i = self.pick[0]
        self.assertTrue(np.allclose(got[TX.COLUMNS].iloc[0].to_numpy(dtype=float), want.iloc[i].to_numpy(dtype=float),
                                    equal_nan=True))

    def test_stops(self):
        with self.assertRaises(SystemExit):
            B.attach_technicals(self.s.assign(tx_rsi_14=1.0), self.p)               # 同じ名前の列がある
        with self.assertRaises(SystemExit):
            B.attach_technicals(self.s, self.p.drop(columns=["vol"]))               # 日足に要る列が無い

    def test_empty(self):
        got = B.attach_technicals(self.s.iloc[:0], self.p)
        self.assertEqual(len(got), 0)
        self.assertEqual(list(got.columns), list(self.s.columns) + TX.COLUMNS)


class TestPresets(unittest.TestCase):
    def test_first_stage(self):
        base = F.columns("all_plus_prog_listing_vol_book_shape_dd_recent")
        tx = F.columns("all_plus_prog_listing_vol_book_shape_dd_recent_tx")
        self.assertEqual(F.GROUPS["tech_indicators"], TX.COLUMNS)
        self.assertEqual(len(tx), 360)
        self.assertEqual(tx[:249], base)
        self.assertEqual(tx[249:], TX.COLUMNS)
        self.assertEqual(F.signature(base), "5ffd7c53c80f")
        self.assertEqual(F.signature(tx), "a0638dfd0880")                        # 実験105 の腕 C と同じ並び
        self.assertEqual(F.DEFAULT_PRESET, "all_plus_prog_listing_vol_book_shape_dd_recent")
        self.assertFalse(set(TX.COLUMNS) - set(F.all_columns()))


if __name__ == "__main__":
    unittest.main()
