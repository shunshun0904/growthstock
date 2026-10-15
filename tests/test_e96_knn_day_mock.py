#!/usr/bin/env python3
"""
実験96（research/exp/e96_knn_day_mock.py）: ダッシュボード案の模型のための、判断日の候補ごとの近傍の集計。

- 日の選び方: 直近の候補2件・3件・4件以上（6件以下を優先）の日
- 候補の順: 今の並べ方（3本の最小の順）に A, B, C…。picked_M は A、picked_Q は近傍の25%点の最上位
- 近傍の集計: 雲（距離 × 帯）の度数の合計が近傍の数。帯の名前は決めたとおり。母集団の形までの距離は 0 以上
- 文脈: 十分位の境が単調、2次元の度数は MIN_CELL 未満を消す
- 出力に銘柄コード・日付が含まれない

  python3 tests/test_e96_knn_day_mock.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e91_expectation_monitor as E91  # noqa: E402
import e95_knn_viz_data as E95  # noqa: E402
import e96_knn_day_mock as E  # noqa: E402
import test_e90_wasserstein_knn_pick as T90  # noqa: E402

LINE_T, POOL_T = T90.LINE_T, T90.POOL_T


class TestBand(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(E.band_of(-12.0, False), "-10 未満")
        self.assertEqual(E.band_of(-3.0, False), "-10〜0")
        self.assertEqual(E.band_of(2.0, False), "0〜+5")
        self.assertEqual(E.band_of(7.0, False), "+5〜+10 未満")
        self.assertEqual(E.band_of(10.0, True), "+10 到達")


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.dist, cls.split, cls.starts, _ = T90.case(planted=6.0)
        cls.df = E91.attach_forward(cls.df, cls.split)
        cls.c, _ = E95.collect(cls.df, cls.dist, cls.split, cls.starts, line=LINE_T, pool_line=POOL_T)
        E.LINE, E.POOL_LINE = LINE_T, POOL_T
        cls.out = E.build(cls.df, cls.dist, cls.split, cls.starts, cls.c)

    def test_days_and_candidates(self):
        kinds = {d["kind"]: d for d in self.out["days"]}
        self.assertIn("2", kinds)
        self.assertEqual(kinds["2"]["n"], 2)
        if "3" in kinds:
            self.assertEqual(kinds["3"]["n"], 3)
        if "4+" in kinds:
            self.assertGreaterEqual(kinds["4+"]["n"], 4)
        for d in self.out["days"]:
            labels = [r["label"] for r in d["candidates"]]
            self.assertEqual(labels, [chr(ord("A") + j) for j in range(d["n"])])
            mins = [r["min15"] for r in d["candidates"]]
            self.assertEqual(mins, sorted(mins, reverse=True))
            self.assertTrue(d["candidates"][0]["picked_M"])
            self.assertEqual(sum(r["picked_Q"] for r in d["candidates"]), 1)
            best = max(d["candidates"], key=lambda r: r["est"]["p25"])
            self.assertTrue(best["picked_Q"] or any(r["picked_Q"] and r["est"]["p25"] == best["est"]["p25"] for r in d["candidates"]))
            for r in d["candidates"]:
                self.assertEqual(int(np.sum(r["cloud"]["counts"])), r["est"]["n"])
                self.assertEqual(r["cloud"]["y_labels"], E.Y_LABELS)
                self.assertIn(r["outcome_band"], E.Y_LABELS)
                self.assertGreaterEqual(min(r["to_hit"]["R"], r["to_hit"]["V"], r["to_loss"]["R"], r["to_loss"]["V"]), 0.0)
                self.assertLessEqual(r["est"]["p25"], r["est"]["p50"])

    def test_same_estimate_as_collect(self):
        """模型の近傍の25%点は、実験95 の collect（＝実験92）の見積もりと同じ。"""
        per = self.c.groupby("Date").size()
        day = per.index[(per == 2).to_numpy()][-1]
        rows = self.c[self.c["Date"] == day].sort_values(["min15", "score"], ascending=False)
        d = [x for x in self.out["days"] if x["kind"] == "2"][0]
        for r, (_, src) in zip(d["candidates"], rows.iterrows()):
            self.assertAlmostEqual(r["est"]["p25"], round(float(src["p25_50"]), 2), places=2)
            self.assertAlmostEqual(r["min15"], round(float(src["min15"]), 1), places=1)

    def test_context_and_no_identifiers(self):
        cx = self.out["context"]
        dec = cx["p25_deciles"]
        self.assertEqual(len(dec), 11)
        self.assertEqual(dec, sorted(dec))
        counts = np.array(cx["density"]["counts"])
        self.assertTrue(((counts == 0) | (counts >= E.MIN_CELL)).all())
        self.assertEqual(int(counts.sum()) + cx["density"]["suppressed_rows"], cx["density"]["n"])
        js = E95.to_json(self.out)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, js)
        for day in self.c["Date"].dt.strftime("%Y-%m-%d").unique()[:50]:
            self.assertNotIn(day, js)


if __name__ == "__main__":
    unittest.main()
