#!/usr/bin/env python3
"""
実験85（research/exp/e85_tier_cascade.py）。

- 段: 1段目 = A15、2段目 = R0 で1段目でない、3段目 = LGB が95以上で R0 でない（重ならない）
- 段階的・16日から緩める: 15日までは1段目だけ、16日以降は同じ日の中で段の上から。月に2件まで
- 3段目を月末の5営業日だけにする形・後から見た段階的な上位2件・今のやり方（R0 の先着2件）
- 報告に銘柄コードが出ない

  python3 tests/test_e85_tier_cascade.py
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
import e85_tier_cascade as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


def mini(rows):
    """段の判定に要る列だけの小さな表。rows: (日付, 段, tkey, score)。"""
    out = []
    for i, (d, t, k, s) in enumerate(rows):
        out.append({"Code": f"{1000 + i}0", "Date": pd.Timestamp(d), "window": 3, "tier": t, "tkey": k, "score": s,
                    "R0": t in (1, 2), "A15": t == 1, "tp10c": 1.0, "hit10": False})
    return pd.DataFrame(out)


class TestCascade(unittest.TestCase):
    def test_relax_after_day16(self):
        g = mini([("2024-03-04", 2, 99.0, 0.9),     # 2段目: 15日までは買わない
                  ("2024-03-08", 1, 97.0, 0.5),     # 1段目: いつでも買う
                  ("2024-03-18", 3, 0.0, 0.8),      # 16日以降: 同じ日に2段目があればそちらが先
                  ("2024-03-18", 2, 96.0, 0.1),
                  ("2024-03-25", 1, 99.0, 0.9)])    # 枠が埋まった後なので買わない
        p = E.cascade(g)
        self.assertEqual(list(p["tier"]), [1, 2])
        self.assertEqual(list(p["Date"].dt.day), [8, 18])
        p3 = E.cascade(g, tier3_days=set())                                 # 3段目を使わない日だけ
        self.assertEqual(list(p3["tier"]), [1, 2])
        g2 = mini([("2024-03-18", 3, 0.0, 0.8)])
        self.assertEqual(len(E.cascade(g2, tier3_days=set())), 0)
        self.assertEqual(len(E.cascade(g2)), 1)

    def test_hindsight_and_last_days(self):
        g = mini([("2024-03-04", 2, 99.0, 0.9), ("2024-03-20", 1, 97.0, 0.5), ("2024-03-29", 1, 98.0, 0.1)])
        h = E.hindsight(g)
        self.assertEqual(list(h["tier"]), [1, 1])
        self.assertEqual(list(h["tkey"]), [98.0, 97.0])
        days = pd.bdate_range("2024-03-01", "2024-04-30")
        last = E.last_days_of_month(days)
        self.assertEqual(len(last), 10)
        self.assertIn(pd.Timestamp("2024-03-29"), last)
        self.assertNotIn(pd.Timestamp("2024-03-22"), last)


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        cls.ft = E.build(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
        cls.thr = E.E84.train_thresholds(cls.df, split)

    def test_tiers_are_disjoint(self):
        t = E.tiers(self.ft)
        self.assertTrue((t.loc[t["tier"] == 1, "A15"]).all())
        self.assertFalse((t.loc[t["tier"] == 2, "A15"]).any())
        self.assertTrue((t.loc[t["tier"] == 2, "R0"]).all())
        self.assertFalse((t.loc[t["tier"] == 3, "R0"]).any())
        self.assertTrue((t.loc[t["tier"] == 3, "L95"]).all())
        p = E.cascade(t)
        self.assertLessEqual(p.groupby("_m").size().max(), 2)
        early = p[pd.to_datetime(p["Date"]).dt.day < E.RELAX_DAY]
        self.assertTrue((early["tier"] == 1).all())                         # 15日までは1段目だけ

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, thr=self.thr, bar_days=list(pd.bdate_range("2021-04-01", "2024-08-30")), ff=self.ft)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ まとめ", "今のやり方", "段ごと", "分位点"):
            self.assertIn(tag, out)
        self.assertIn("main_vs_cur", s)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
