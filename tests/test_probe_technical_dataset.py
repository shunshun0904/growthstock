#!/usr/bin/env python3
"""
probe_technical_dataset（research/exp/probe_technical_dataset.py）の部品。データには触らない。

- 列の組の確かめ: 249列の組の指紋・360列の組が249列の後ろに111列（実験105 の腕 C と同じ指紋）
- 付き方: 列ごとの欠測の数・分位（0.1%〜99.9%）・裾の目安・線形の前処理の型
- 予測用の突き合わせ: 実験105 の作り方で数え直した値と全行を比べ、食い違い・いちばん新しい日の欠測を数える
  （作りものの日足を build_panel の代わりに渡す）

  python3 tests/test_probe_technical_dataset.py
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
sys.path.insert(0, os.path.join(ROOT, "tests"))

import probe_technical_dataset as P  # noqa: E402
import technicals as TX  # noqa: E402
from test_build_technicals import bars  # noqa: E402


class TestPresets(unittest.TestCase):
    def test_ok(self):
        r = P.presets_report()
        self.assertTrue(r["ok"])
        self.assertEqual((r["base_n"], r["tx_n"]), (249, 360))
        self.assertTrue(r["tx_is_base_plus_new"])


class TestDistribution(unittest.TestCase):
    def test_table(self):
        rng = np.random.default_rng(0)
        n = 5000
        df = pd.DataFrame({c: rng.normal(size=n) for c in TX.COLUMNS})
        df.loc[:9, "tx_rsi_14"] = np.nan
        df["tx_vr_25"] = rng.lognormal(0, 1.5, n)                                   # 右の裾が長い
        d = P.distribution(df).set_index("col")
        self.assertEqual(len(d), 111)
        self.assertEqual(d.loc["tx_rsi_14", "missing"], 10)
        self.assertEqual(d.loc["tx_rsi_14", "kind"], "pos")
        self.assertEqual(d.loc["tx_vr_25", "kind"], "pos_heavy")
        self.assertGreater(d.loc["tx_vr_25", "tail_up"], d.loc["tx_rsi_14", "tail_up"] * 2)
        q = d.loc["tx_ma_gap_5"]
        self.assertTrue(q["q0.001"] < q["q0.05"] < q["q0.5"] < q["q0.95"] < q["q0.999"])
        self.assertAlmostEqual(q["tail_up"], (q["q0.999"] - q["q0.5"]) / (q["q0.95"] - q["q0.5"]))


class TestCheckPredict(unittest.TestCase):
    def setUp(self):
        p = pd.concat([bars("10010", 300, 0), bars("10020", 260, 1)], ignore_index=True)
        p["Date"] = p["Date"].astype("datetime64[ns]")
        self.p = p
        rows = p.iloc[[100, 299, 400, 559]][["Code", "Date"]].reset_index(drop=True)
        tx = TX.compute_at(p, rows)
        self.pred = pd.concat([rows, tx], axis=1).assign(label=pd.array([1, pd.NA, 0, pd.NA], dtype="Int64"))

    def run_check(self, pred):
        with mock.patch.object(P.E105, "build_panel", return_value=self.p):
            return P.check_predict(pred)

    def test_match(self):
        r = self.run_check(self.pred)
        self.assertEqual(r["bad_cells"], 0)
        self.assertEqual(r["rows"], 4)
        self.assertEqual(r["unlabeled"], 2)
        self.assertEqual(r["latest_rows"], 1)
        self.assertEqual(r["latest_all_nan"], 0)

    def test_mismatch(self):
        bad = self.pred.copy()
        bad.loc[0, "tx_rsi_14"] += 1.0
        bad.loc[1, "tx_macd"] = np.nan
        r = self.run_check(bad)
        self.assertEqual(r["bad_cells"], 2)
        self.assertEqual(r["bad_columns"], 2)

    def test_missing_columns(self):
        with self.assertRaises(SystemExit):
            self.run_check(self.pred.drop(columns=["tx_rsi_14"]))


if __name__ == "__main__":
    unittest.main()
