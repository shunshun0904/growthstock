#!/usr/bin/env python3
"""
実験88（research/exp/e88_breaks_returns.py）。

- 並べ替え検定: D = 印の側の平均 − 残りの平均。層の中だけで並べ替える（片側しか無い層は動かさない）。
  差を仕込めば片側 p が小さく、仕込まなければ小さくならない。片側が0件なら NaN
- 銘柄単位: 同じ層・同じ銘柄・同じ側の行を平均して1つにする
- 判定: D < 0・行の p < 0.05・銘柄単位の p < 0.05 の3つがそろったときだけ「支える」
- 順位相関: 発火数と収益が同じ順なら窓の平均 +1
- 合成データで報告が全部の段を出し、帯の件数の合計が規則の行の数と合い、銘柄コードを出さない

  python3 tests/test_e88_breaks_returns.py
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
import e88_breaks_returns as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


def frame(n_per=60, windows=(3, 4, 5), shift=0.0, seed=0):
    """窓ごとに n_per 行。発火は 2〜30 件、7件以下の行の収益に shift を足す。"""
    rng = np.random.default_rng(seed)
    rows = []
    for w in windows:
        nb = rng.integers(2, 31, size=n_per)
        r = rng.normal(2.0, 3.0, size=n_per) + np.where(nb < E.SKIP, shift, 0.0)
        d = pd.bdate_range(f"{2020 + w}-01-05", periods=n_per)
        rows.append(pd.DataFrame({"Code": [f"{1000 + (i % 25)}0" for i in range(n_per)], "Date": d, "window": w,
                                  "n_break": nb, "tp10c": r, "hit10": r >= 10}))
    return pd.concat(rows, ignore_index=True)


class TestStratPerm(unittest.TestCase):
    def test_d_and_counts(self):
        t = E.strat_perm(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, False, False]), np.zeros(4), n_perm=50)
        self.assertAlmostEqual(t["D"], 1.5 - 3.5)
        self.assertEqual((t["n_yes"], t["n_no"]), (2, 2))
        self.assertTrue(0 < t["p_less"] <= 1 and 0 < t["p_two"] <= 1)

    def test_only_mixed_strata_are_shuffled(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f = np.array([True, True, False, True, False, False])
        s = np.array([0, 0, 0, 1, 1, 2])                           # 層2 は片側だけ
        t = E.strat_perm(v, f, s, n_perm=20)
        self.assertEqual(t["strata_used"], 2)
        t1 = E.strat_perm(v, f, np.array([0, 0, 1, 2, 3, 4]), n_perm=20)    # どの層も片側だけ → 動かない
        self.assertEqual(t1["strata_used"], 0)
        self.assertAlmostEqual(t1["p_less"], 1.0)                 # 並べ替えた D はいつも実際と同じ

    def test_planted_and_null(self):
        g = frame(shift=-4.0)
        t = E.strat_perm(g["tp10c"].to_numpy(), g["n_break"].to_numpy() < E.SKIP, g["window"].to_numpy(), n_perm=400)
        self.assertLess(t["D"], 0)
        self.assertLess(t["p_less"], 0.05)
        g0 = frame(shift=0.0, seed=5)
        t0 = E.strat_perm(g0["tp10c"].to_numpy(), g0["n_break"].to_numpy() < E.SKIP, g0["window"].to_numpy(),
                          n_perm=400)
        self.assertGreater(t0["p_less"], 0.05)

    def test_one_side_empty(self):
        t = E.strat_perm(np.array([1.0, 2.0]), np.array([False, False]), np.zeros(2), n_perm=10)
        self.assertTrue(np.isnan(t["D"]))
        self.assertTrue(np.isnan(t["p_less"]))
        self.assertIn("0件", E.fmt_p(t))


class TestCodeLevel(unittest.TestCase):
    def test_collapse(self):
        g = pd.DataFrame({"tp10c": [1.0, 3.0, 5.0, np.nan, 7.0], "window": [3, 3, 3, 3, 4],
                          "Code": ["10010", "10010", "10010", "10010", "10010"]})
        flag = np.array([True, True, False, True, True])
        v, f, s = E.code_level(g, flag, "window")
        got = sorted(zip(s.tolist(), f.tolist(), v.tolist()))
        self.assertEqual(got, [(3, False, 5.0), (3, True, 2.0), (4, True, 7.0)])   # NaN の行は落ちる


class TestSkipTest(unittest.TestCase):
    def test_support_needs_all_three(self):
        st = E.skip_test(frame(n_per=120, shift=-5.0), n_perm=300)
        self.assertTrue(st["support"], st["row"])
        self.assertLess(st["code"]["p_less"], 0.05)
        self.assertEqual(set(st["per_window"]), {3, 4, 5})
        st0 = E.skip_test(frame(n_per=120, shift=+3.0, seed=2), n_perm=300)     # 7件以下のほうが良い → 支えない
        self.assertFalse(st0["support"])
        self.assertGreater(st0["row"]["D"], 0)

    def test_band_test_counts(self):
        g = frame(n_per=80, seed=3)
        bt = E.band_test(g, n_perm=50)
        nb = g["n_break"].to_numpy()
        self.assertEqual(bt["n_yes"], int((nb >= E.STRONG).sum()))
        self.assertEqual(bt["n_no"], int(((nb >= E.SKIP) & (nb < E.STRONG)).sum()))


class TestSpearman(unittest.TestCase):
    def test_monotone(self):
        g = frame(n_per=40, seed=4)
        g["tp10c"] = g["n_break"] * 0.5 + 1.0                          # 発火が多いほど良い（同じ順）
        sp = E.spearman_by_window(g, n_perm=100)
        self.assertEqual(sp["n"], 3)
        self.assertAlmostEqual(sp["mean"], 1.0, places=6)
        self.assertEqual(sp["pos"], 3)
        self.assertLess(sp["p_two"], 0.05)

    def test_too_few_rows(self):
        g = frame(n_per=5, seed=6)
        self.assertEqual(E.spearman_by_window(g, n_perm=10)["n"], 0)


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        cls.ft = E.E87.build(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
        cls.ff = E.E87.build(cls.df, [(np.arange(len(cls.df)), E79.pct_matrix(cls.df, E79.MODELS))])
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_report_sections_and_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, bar_days=self.bar_days, ff=self.ff, n_perm=60)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ 7.", "■ まとめ", "判定"):
            self.assertIn(tag, out)
        for k in ("days", "main", "bands", "trend", "other", "strong", "trade", "full"):
            self.assertIn(k, s)
        k = E.rule_mask(self.ft, "min_K15", 95.0)
        n_bands = sum(s["bands"]["K15"][b]["n"] for _, _, b in E.BANDS)
        self.assertEqual(n_bands, int(pd.to_numeric(self.ft.loc[k, "tp10c"], errors="coerce").notna().sum()))
        self.assertEqual(s["main"]["row"]["n_yes"] + s["main"]["row"]["n_no"], n_bands)
        self.assertEqual(s["trade"]["skip"]["quiet"], 0)           # 8件以上の日だけ買う形に7件以下の日は無い
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
