#!/usr/bin/env python3
"""
実験55（research/exp/e55_label_sigma_full.py）のテスト。

- 腕はラベルの σ の列に対応し、σ120 も実験54 の sigmas_from_bars が作る（90本たまるまで欠測）
- 評価用の付け替え（with_base_label）は label を L0 のラベルに置き換え、自分のラベルを own_label に残す
- 探索し直し（tune_label）は 2試行で通り、本番と同じ形のパラメータを返す

  python3 tests/test_e55_label_sigma_full.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import e54_label_sigma as E54  # noqa: E402
import e55_label_sigma_full as E  # noqa: E402
import tuning  # noqa: E402
from test_e54_label_sigma import bars  # noqa: E402


class TestSigma120(unittest.TestCase):
    def test_arms_map_to_sigma_columns(self):
        s = E54.sigmas_from_bars(bars(n_days=150))
        for a, col in E.ARMS.items():
            self.assertIn(col, s.columns, a)
        first = s[s["Code"] == "1000"].reset_index(drop=True)
        self.assertTrue(first["sigma120"].iloc[:89].isna().all())
        self.assertTrue(first["sigma120"].iloc[90:].notna().all())


class TestBaseLabel(unittest.TestCase):
    def test_with_base_label(self):
        df = pd.DataFrame({"Code": ["1", "2", "3"], "Date": pd.to_datetime(["2021-01-04"] * 3),
                           "y_L0": [1.0, 0.0, 1.0], "vol_20d": [1.0, 2.0, 3.0]})
        o = pd.DataFrame({"Code": ["1", "2", "3"], "Date": pd.to_datetime(["2021-01-04"] * 3),
                          "label": [0.0, 0.0, 1.0], "score": [0.1, 0.2, 0.3], "fold": [1, 1, 1],
                          "ret_o1_20": [0.0, 0.1, 0.2], "ret_o1_40": [0.0, 0.1, 0.2]})
        m = E.with_base_label(o, df)
        self.assertEqual(m["label"].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(m["own_label"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(m["vol_20d"].tolist(), [1.0, 2.0, 3.0])
        self.assertIn("ret_o1_40", m.columns)


class TestTune(unittest.TestCase):
    def test_tune_label_two_trials(self):
        rng = np.random.default_rng(0)
        rows = []
        for d in pd.bdate_range("2019-01-01", periods=400):
            for j in range(8):
                x = rng.normal()
                rows.append({"Date": d, "Code": f"{1000 + j}", "x1": x, "x2": rng.normal(),
                             "log_market_cap": float(rng.normal(10, 1)),
                             "label": float(x + rng.normal() > 1.0), "y_L1": float(x + rng.normal() > 0.8)})
        df = pd.DataFrame(rows)
        rec = E.tune_label(df, "y_L1", ["x1", "x2"], n_trials=2)
        p = rec["params"]
        self.assertEqual(p["objective"], "binary")
        self.assertEqual(p["n_estimators"], tuning.SEARCH_N_ESTIMATORS)
        self.assertIn("mean_pr_auc", rec["_cv"])
        self.assertGreater(rec["_pos_rate"], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
