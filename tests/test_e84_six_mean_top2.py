#!/usr/bin/env python3
"""
実験84（research/exp/e84_six_mean_top2.py）。

- 6本の平均（avg6）は 1.2σ・1.5σ の lgbm・xgb・cat の百分位の平均。avg6_top は同じ窓の候補全体の中で上位何%か（最上位 = 0）
- 後から見た月の上位2件: 各月の A15 の中で avg6 の上位2件（月に2件まで、A15 だけ）。無作為に選んだときの平均の分布
- その日のうちに決められる形: 線は訓練だけで「A15 が月に平均2件」になる高さ。各月の先着2件（日付の早い順）
- 四分位: 窓ごとに A15 の行を4つに分け、上の四分位ほど avg6 が高い
- 報告に今のやり方（R0 の上位2件・先着2件）との比べが出て、銘柄コードは出ない

  python3 tests/test_e84_six_mean_top2.py
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
import e84_six_mean_top2 as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


class TestSixMean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        cls.split = E79.window_split(cls.df, folds)
        cls.ft = E.build(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in cls.split.items()])

    def test_avg6(self):
        w, (tr, te, Ptr, Pte) = next(iter(self.split.items()))
        idx = [list(E79.MODELS).index(a) for a in E.SIX]
        np.testing.assert_allclose(self.ft["avg6"].to_numpy()[:len(te)], Pte[:, idx].mean(axis=1))
        self.assertEqual(len(E.SIX), 6)
        a15 = self.ft[self.ft["A15"]]
        self.assertTrue((a15["avg6r"] >= 95.0).all())                   # 6本とも95以上なら平均も95以上
        for _, g in self.ft.groupby("window"):
            self.assertAlmostEqual(g.loc[g["avg6"].idxmax(), "avg6_top"], 0.0)
            self.assertTrue(((g["avg6_top"] >= 0) & (g["avg6_top"] < 100)).all())

    def test_monthly_top(self):
        top = E.monthly_by(self.ft, "A15", "avg6")
        self.assertTrue(top["A15"].all())
        self.assertLessEqual(top.groupby("_m").size().max(), 2)
        a15 = self.ft[self.ft["A15"]].assign(_m=lambda x: pd.to_datetime(x["Date"]).dt.to_period("M"))
        for m, g in top.groupby("_m"):
            others = a15[a15["_m"] == m]
            self.assertGreaterEqual(g["avg6"].min(), others["avg6"].nlargest(len(g)).min() - 1e-12)
        rnd = E.random_monthly(self.ft, "A15", "tp10c", draws=50)
        self.assertEqual(len(rnd), 50)
        self.assertTrue(np.isfinite(rnd).all())

    def test_quartiles(self):
        q = E.quartiles_in_window(self.ft, "A15")
        self.assertTrue(q[~self.ft["A15"]].isna().all())
        self.assertEqual(set(q.dropna().unique()), {0.0, 1.0, 2.0, 3.0})
        for _, g in self.ft.assign(q=q).dropna(subset=["q"]).groupby("window"):
            med = g.groupby("q")["avg6"].median()
            self.assertTrue(med.is_monotonic_increasing)

    def test_thresholds_and_first_picks(self):
        thr = E.train_thresholds(self.df, self.split)
        self.assertEqual(sorted(thr), sorted(int(w) for w in self.split))
        for w, (tr, te, Ptr, Pte) in self.split.items():
            t = thr[int(w)]
            fl = E.E82.flags_of(E.E82.rounded(Ptr))["A15"]
            a6 = E.avg6_of(Ptr)[fl]
            self.assertGreaterEqual(int((a6 >= t["T"]).sum()), min(t["k"], len(a6)))   # 訓練で月に平均2件（同点は多め）
            self.assertLess(int((a6 > t["T"]).sum()), max(t["k"], 1))
            self.assertTrue(0 < t["share_all"] < 100)
        mask = E.threshold_mask(self.ft, thr)
        self.assertFalse((mask & ~self.ft["A15"].to_numpy()).any())
        picks = E.first_by(self.ft, mask, "avg6")
        self.assertLessEqual(picks.groupby("_m").size().max(), 2)
        sub = self.ft[mask].assign(_m=E.months_of)
        for m, g in picks.groupby("_m"):
            earliest = sub[sub["_m"] == m]["Date"].sort_values().iloc[:len(g)]
            self.assertTrue((g["Date"].sort_values().to_numpy() == earliest.to_numpy()).all())   # 先着

    def test_report_prints_no_codes(self):
        thr = E.train_thresholds(self.df, self.split)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, thr=thr, ff=self.ft, draws=50)
        out = buf.getvalue()
        for tag in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ まとめ", "分位点", "今のやり方"):
            self.assertIn(tag, out)
        self.assertIn("p10", s["A"]["top6"]["q"])
        self.assertIn("vs_cur", s["B"])
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
