#!/usr/bin/env python3
"""実験69（research/exp/e69_label_sigma_dfa.py）の部品のテスト。"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e54_label_sigma as E54  # noqa: E402
import e69_label_sigma_dfa as E69  # noqa: E402


class TestSigmaDfa(unittest.TestCase):
    def test_weight_clips_and_fills(self):
        w = E69.weight([0.3, 0.5, 0.65, 0.8, 0.95, np.nan])
        np.testing.assert_allclose(w.to_numpy(), [0.0, 0.0, 0.5, 1.0, 1.0, 0.5])

    def test_blend_endpoints_and_middle(self):
        s20 = pd.Series([2.0, 2.0, 2.0, 2.0, np.nan])
        s120 = pd.Series([8.0, 8.0, 8.0, np.nan, 8.0])
        a = pd.Series([0.9, 0.4, 0.65, 0.4, 0.9])
        out = E69.sigma_dfa(s20, s120, a)
        self.assertAlmostEqual(out[0], 2.0)              # α 高い → σ20
        self.assertAlmostEqual(out[1], 8.0)              # α 低い → σ120
        self.assertAlmostEqual(out[2], math.sqrt(16.0))  # 中間 → 幾何平均
        self.assertAlmostEqual(out[3], 2.0)              # σ120 が無い → σ20
        self.assertTrue(np.isnan(out[4]))                # σ20 が無い → 欠測

    def test_forward_sigma_is_twenty_rows_ahead(self):
        sg = pd.DataFrame({"Code": ["a"] * 30 + ["b"] * 30,
                           "sigma20": list(range(30)) + list(range(100, 130))})
        f = E69.forward_sigma(sg, 20)
        self.assertEqual(f.iloc[0], 20)
        self.assertEqual(f.iloc[30], 120)
        self.assertTrue(np.isnan(f.iloc[29]))   # 銘柄をまたがない

    def test_arms_and_labels_agree(self):
        self.assertEqual(set(E69.ARMS), set(E69.LABELS))
        self.assertEqual(E69.ARMS["L0"], E54.ARMS["L0"])


class TestE54Refactor(unittest.TestCase):
    def test_functions_exist(self):
        for name in ("report_labels", "evaluate", "summarize", "build_labels"):
            self.assertTrue(callable(getattr(E54, name)))

    def test_summarize_handles_dataframe_rows(self):
        # main は list を、実験69 は DataFrame を渡す。どちらでも落ちない
        ed = pd.DataFrame({"shift": [0, 2], "arm": ["L0", "L0"], "kind": ["thr95", "thr95"],
                           "thr_lift": [0.1, 0.2], "bad": [0.1, 0.1], "thr_fold_mean": [0.1, 0.2]})
        rows = pd.DataFrame({"shift": [0, 2], "arm": ["L0", "L0"], "top1": [0.01, 0.02]})
        E54.summarize(ed, rows, ["L0"], [0, 2])
        E54.summarize(ed, rows.to_dict("records"), ["L0"], [0, 2])


if __name__ == "__main__":
    unittest.main()
