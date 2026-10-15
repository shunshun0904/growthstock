#!/usr/bin/env python3
"""
実験95（research/exp/e95_knn_viz_data.py）: 後工程の解釈のための集計（可視化の材料）。

- collect: 実験92 の collect と同じ候補・見積もり（p25_50 が一致）。距離の成分は 0 以上で、和が d_nn
- 度数: 合計が件数と合う。分位関数は単調。十分位の件数の合計が候補の数。マス目は MIN_CELL 未満を 0 にして消した件数を記す
- 日の中の順位: 日の数が合い、見積もりの順位1の行は各日に1つ以上
- 形: 60点の平均と件数。母集団の分け方（到達 / −10% 割れ / 全部）
- @viz の行: つなぎ直すと同じ JSON。NaN は出さない
- 報告（集計）に銘柄コードを出さない

  python3 tests/test_e95_knn_viz_data.py
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

import e91_expectation_monitor as E91  # noqa: E402
import e92_knn_quantile_pick as E92  # noqa: E402
import e95_knn_viz_data as E  # noqa: E402
import test_e90_wasserstein_knn_pick as T90  # noqa: E402

LINE_T, POOL_T = T90.LINE_T, T90.POOL_T


class TestCollect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.dist, cls.split, cls.starts, _ = T90.case(planted=6.0)
        cls.df = E91.attach_forward(cls.df, cls.split)
        cls.c, cls.info = E.collect(cls.df, cls.dist, cls.split, cls.starts, line=LINE_T, pool_line=POOL_T)

    def test_same_as_e92(self):
        c92, i92 = E92.collect(self.df, self.dist, self.split, self.starts, line=LINE_T, pool_line=POOL_T)
        self.assertEqual(self.info["days"], i92["days"])
        m = self.c[["row", "p25_50"]].merge(c92[["row", "p25_50"]], on="row", suffixes=("", "_92"))
        self.assertEqual(len(m), len(self.c))
        np.testing.assert_allclose(m["p25_50"], m["p25_50_92"])

    def test_distance_components(self):
        self.assertTrue((self.c["dR_nn"] >= 0).all() and (self.c["dV_nn"] >= 0).all())
        np.testing.assert_allclose(self.c["dR_nn"] + self.c["dV_nn"], self.c["d_nn"], atol=1e-9)

    def test_aggregates(self):
        s = E.summarize_line(self.c, self.info, self.df, self.dist, draws=200)
        n_days = self.info["days"]
        pk = s["picks"]
        self.assertEqual(sum(pk["Q"]["hist"]), n_days)
        self.assertEqual(sum(pk["M"]["hist"]), n_days)
        self.assertEqual(sum(pk["R"]["hist"]), n_days * 200)
        cv = pk["Q"]["curve"]
        self.assertTrue(all(a <= b for a, b in zip(cv, cv[1:])))
        self.assertEqual(sum(pk["pair"]["hist"]), n_days)
        self.assertAlmostEqual(pk["pair"]["up"] + pk["pair"]["same"] + pk["pair"]["down"], 100.0, places=0)
        est = s["estimate"]
        self.assertEqual(sum(b["n"] for b in est["calibration"]), len(self.c))
        dens = est["density"]
        counts = np.array(dens["counts"])
        self.assertTrue(((counts == 0) | (counts >= E.MIN_CELL)).all())
        self.assertEqual(int(counts.sum()) + dens["suppressed_rows"], len(self.c))
        rk = est["rank"]
        self.assertEqual(sum(rk["days"].values()), n_days)
        for cls, mat in rk["classes"].items():
            mat = np.array(mat)
            if rk["days"][cls]:
                self.assertGreaterEqual(int(mat[0].sum()), rk["days"][cls])      # 見積もりの順位1は日ごとに1つ以上
        sh = s["shapes"]
        self.assertEqual(len(sh["probs"]), 60)
        self.assertEqual(sh["picked_Q"]["QR"]["n"], n_days)
        self.assertEqual(len(sh["picked_Q"]["QR"]["mean"]), 60)
        self.assertGreater(sh["pool_all"]["QR"]["n"], sh["pool_hit"]["QR"]["n"])
        self.assertEqual(sum(s["distances"]["dR"]), len(self.c))
        self.assertEqual(len(s["selectors"]), 5)
        self.assertEqual(s["counts"]["per_window"][0]["days"] + sum(w["days"] for w in s["counts"]["per_window"][1:]), n_days)

    def test_json_roundtrip_and_no_codes(self):
        s = E.summarize_line(self.c, self.info, self.df, self.dist, draws=100)
        out = {"meta": {"x": 1}, "lines": {"80": s}}
        js = E.to_json(out)
        self.assertNotIn("NaN", js)
        lines = E.chunk_lines(js, size=300)
        self.assertTrue(all(ln.startswith("@viz") and ln[8] == "|" for ln in lines))
        shuffled = ["2026-10-15T00:00:00.0000000Z ■ 集計の JSON を @viz の行に分けて出す"] + \
                   ["2026-10-15T00:00:00.0000000Z " + ln for ln in reversed(lines)] + ["other line"]
        back = E.parse_chunks(shuffled)                               # 見出し・時刻の前置き・順番の乱れに強い
        self.assertEqual(back, E.parse_chunks(lines))
        self.assertEqual(back["lines"]["80"]["picks"]["M"]["hist"], s["picks"]["M"]["hist"])
        self.assertEqual(E.parse_chunks(lines)["lines"]["80"]["picks"]["Q"]["hist"], s["picks"]["Q"]["hist"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.print_brief(80.0, s)
        text = buf.getvalue() + js
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, text)


class TestHelpers(unittest.TestCase):
    def test_hist_and_curve(self):
        self.assertEqual(E.hist([-50, -1, 0, 9.99, 10.0], [-40, 0, 10.01]), [2, 3])
        self.assertEqual(E.hist([], [0, 1]), [0])
        self.assertEqual(E.curve([]), [None] * len(E.PCT_CURVE))
        self.assertEqual(len(E.curve([1.0, 2.0, 3.0])), len(E.PCT_CURVE))

    def test_rank_matrix_small(self):
        d1, d2 = pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-11")
        c = pd.DataFrame({"Date": [d1, d1, d2, d2, d2], "p25_50": [1.0, 2.0, 3.0, 1.0, 2.0], E.TP: [10.0, 0.0, 5.0, 5.0, -3.0]})
        rk = E.rank_matrix(c, "p25_50")
        self.assertEqual(rk["days"], {"2": 1, "3": 1, "4+": 0})
        self.assertEqual(rk["classes"]["2"][0][1], 1)          # 見積もり1位（2.0）の実際は2位（0.0）
        self.assertEqual(rk["classes"]["3"][0][0], 1)          # 見積もり1位（3.0）の実際は同点1位

    def test_density_suppression(self):
        rng = np.random.default_rng(0)
        c = pd.DataFrame({"p25_50": rng.normal(size=40), E.TP: rng.normal(size=40)})
        d = E.density2d(c, "p25_50", n_x=4, y_edges=[-40, 0, 10.01], min_cell=3)
        counts = np.array(d["counts"])
        self.assertEqual(counts.shape, (4, 2))
        self.assertTrue(((counts == 0) | (counts >= 3)).all())
        self.assertEqual(int(counts.sum()) + d["suppressed_rows"], 40)


if __name__ == "__main__":
    unittest.main()
