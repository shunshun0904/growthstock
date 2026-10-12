#!/usr/bin/env python3
"""
月ごとの候補数（research/exp/probe_monthly_counts.py）。

- 月ごとの R0・A15 の件数の合計が、行ごとの印の合計と一致する。物差しが覆わない月は空（NaN）
- 発火8件以上の日に限った件数は、限らない件数以下
- 0件の月の数と、続いた最長
- 報告に銘柄コードが出ない

  python3 tests/test_probe_monthly_counts.py
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
import probe_monthly_counts as E  # noqa: E402
import test_e82_and_rule_variants as T82  # noqa: E402


class TestZeroRuns(unittest.TestCase):
    def test_runs(self):
        idx = pd.period_range("2024-01", periods=7, freq="M")
        z = E.zero_runs(pd.Series([3, 0, 0, 1, 0, 0, 0], index=idx, dtype=float))
        self.assertEqual((z["zero"], z["longest"], z["months"]), (5, 3, 7))
        self.assertEqual((z["longest_from"], z["longest_to"]), ("2024-05", "2024-07"))
        z2 = E.zero_runs(pd.Series([np.nan, 0, 2], index=idx[:3], dtype=float))
        self.assertEqual((z2["months"], z2["zero"], z2["longest"]), (2, 1, 1))
        self.assertEqual(E.zero_runs(pd.Series([1.0, 2.0], index=idx[:2]))["longest"], 0)


class TestMonthly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T82.synth()
        split = E79.window_split(cls.df, folds)
        cls.scales = {"s": E.flag_rows(cls.df, [(np.arange(len(cls.df)), E79.pct_matrix(cls.df, E79.MODELS))]),
                      "f": E.flag_rows(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])}
        cls.tbl = E.monthly(cls.df, cls.scales)

    def test_totals(self):
        t = self.tbl
        self.assertEqual(int(t["breaks"].sum()), len(self.df))
        for s, fl in self.scales.items():
            self.assertEqual(int(np.nansum(t[f"{s}_r0"])), int(fl["R0"].sum()))
            self.assertEqual(int(np.nansum(t[f"{s}_a15"])), int(fl["A15"].sum()))
            self.assertTrue((t[f"{s}_a15"].dropna() <= t[f"{s}_r0"].dropna()).all())          # A15 は R0 の内
            self.assertTrue((t[f"{s}_r0_hot"].dropna() <= t[f"{s}_r0"].dropna()).all())
            self.assertTrue((t[f"{s}_r0_codes"].dropna() <= t[f"{s}_r0"].dropna()).all())
        self.assertTrue(t["f_r0"].isna().any())                                               # 前向きは最初の窓を覆わない
        self.assertFalse(t["s_r0"].isna().any())
        self.assertTrue((t["hot_days"] <= t["days"]).all())

    def test_hot_days(self):
        per_day = pd.to_datetime(self.df["Date"]).value_counts()
        self.assertEqual(int(self.tbl["hot_days"].sum()), int((per_day >= E.MIN_BREAK).sum()))

    def test_report(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.tbl)
        out = buf.getvalue()
        for tag in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "csv:month,breaks"):
            self.assertIn(tag, out)
        self.assertEqual(sum(1 for ln in out.splitlines() if ln.startswith("csv:")), len(self.tbl) + 1)
        self.assertIn("s_r0", s)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
