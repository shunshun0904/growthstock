#!/usr/bin/env python3
"""
1.5σ の3本 95以上の見るだけの集計（research/exp/probe_k15_baseline.py）。

- 成績の要約: +10% に届いた割合・届いた日の分位・−10%割れ
- 月ごとの件数の合計が行の印の合計と一致する
- 幅（最大 − 最小）は 0 以上で、最小が線以上なら最大も線以上
- 報告に銘柄コードが出ない

  python3 tests/test_probe_k15_baseline.py
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
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e79_model_clusters as E79  # noqa: E402
import probe_k15_baseline as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


class TestStats(unittest.TestCase):
    def test_stats(self):
        g = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05", "2024-01-09", "2024-03-01", "2024-03-04"]),
                          "tp10c": [10.0, -12.0, 10.0, 2.0], "hit10": [True, False, True, False],
                          "hitday10": [3.0, np.nan, 7.0, np.nan]})
        s = E.stats(g, months=3)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean"], 2.5)
        self.assertAlmostEqual(s["hit"], 50.0)
        self.assertAlmostEqual(s["bad10"], 25.0)
        self.assertAlmostEqual(s["day50"], 5.0)
        self.assertEqual((s["two"], s["one"], s["zero"]), (2, 0, 1))


class TestProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        parts = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
        cls.ft = E.add_spread(E.E87.build(cls.df, parts), parts)
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_spread_and_counts(self):
        sp = self.ft["max_K15"] - self.ft["min_K15"]
        self.assertTrue((sp.dropna() >= 0).all())
        k = E.rule_mask(self.ft, "K15")
        self.assertTrue((self.ft.loc[k, "max_K15"] >= E.LINE).all())
        tbl = E.monthly_counts(self.ft)
        self.assertEqual(int(tbl["K15"].sum()), int(k.sum()))
        self.assertEqual(int(tbl["breaks"].sum()), len(self.ft))

    def test_blocks_print_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.scale_block(self.ft, self.bar_days, "前向き")
            p = E.panel_block(self.ft, s["months"])
        out = buf.getvalue()
        for tag in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "画面の3枠の形", "月2件の先着", "惜しい帯"):
            self.assertIn(tag, out)
        self.assertIn("K15", s)
        self.assertIn("bands", p)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
