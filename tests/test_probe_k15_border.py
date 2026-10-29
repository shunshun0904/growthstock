#!/usr/bin/env python3
"""
「1.5σで際どい」の数字（research/exp/probe_k15_border.py）。

- 形の分け方が画面の borderShape と同じ（線 95・下 85。2本95以上は残りで3つ、1本以下は最小で2つ）。NaN は形なし
- 一番低いモデルは同点なら先のモデル（画面の laggard と同じ）
- 赤の注意の線: 20件以上で平均 0% 以下
- 画面（strategy.js）の線・規則の3本と同じ
- 合成データで、形ごとの件数の合計が母集団と合い、報告に銘柄コードが出ない

  python3 tests/test_probe_k15_border.py
"""
import contextlib
import io
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e79_model_clusters as E79  # noqa: E402
import probe_k15_border as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


class TestShape(unittest.TestCase):
    def test_shapes_like_screen(self):
        V = np.array([[96, 97, 98], [96, 97, 92], [96, 97, 87], [96, 97, 70],
                      [96, 91, 93], [80, 88, 86], [96, 80, 99.9], [95.0, 95.0, 94.9], [np.nan, 99, 99]], dtype=float)
        got = E.shape_of(V).tolist()
        self.assertEqual(got, ["all95", "two95_hi", "two95_mid", "two95_lo", "min90", "other", "two95_lo", "two95_hi",
                               None])

    def test_min_shapes(self):
        V = np.array([[99, 89, 86], [99, 84.9, 99.0 - 10], [90, 90, 90]], dtype=float)
        self.assertEqual(E.shape_of(V).tolist(), ["min85", "other", "min90"])

    def test_laggard_ties_first(self):
        g = ("lgbm_k15", "xgb_k15", "cat_k15")
        V = np.array([[90, 90, 99], [99, 80, 80], [97, 96, 95], [np.nan, 1, 2]], dtype=float)
        self.assertEqual(E.laggard_of(V, g).tolist(), ["lgbm_k15", "xgb_k15", "cat_k15", None])


class TestWarn(unittest.TestCase):
    def test_line(self):
        self.assertTrue(E.warn_of({"n": 20, "mean": 0.0}))
        self.assertTrue(E.warn_of({"n": 300, "mean": -1.2}))
        self.assertFalse(E.warn_of({"n": 19, "mean": -3.0}))         # 件数が少ない
        self.assertFalse(E.warn_of({"n": 50, "mean": 0.01}))
        self.assertFalse(E.warn_of({"n": 0, "mean": float("nan")}))


class TestSyncWithScreen(unittest.TestCase):
    def test_line_and_models(self):
        with open(os.path.join(ROOT, "src", "lib", "strategy.js"), encoding="utf-8") as fh:
            js = fh.read()
        m = re.search(r"export const BORDER = \{ line: (\d+), lo: (\d+) \}", js)
        self.assertIsNotNone(m)
        self.assertEqual((float(m.group(1)), float(m.group(2))), (E.LINE, E.NEAR_LO))
        r = re.search(r"export const RULE_MODELS = \[([^\]]*)\]", js)
        # 2026-10-29 から画面の規則は 1.5σ の3本（このプローブの主の組 K15。画面の数字は probe_k15_panel が同じ形で測る）。
        # 2026-10-12〜10-23 も K15、2026-10-23〜10-29 は T3（1.2σ の3本。probe_t3_panel）だった
        self.assertEqual(tuple(re.findall(r"'([^']+)'", r.group(1))), tuple(E.GROUPS["K15"]))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        parts = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
        cls.ft = E.add_shapes(E.E87.build(cls.df, parts), parts)

    def test_shapes_match_min_column(self):
        k = (self.ft["shape_K15"] == "all95").to_numpy()
        np.testing.assert_array_equal(k, self.ft["min_K15"].to_numpy(dtype=float) >= E.LINE)
        self.assertTrue(self.ft.loc[self.ft["shape_K15"] == "all95", "lag_K15"].isin(E.GROUPS["K15"]).all())

    def test_report(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, self.ft)
        out = buf.getvalue()
        for tag in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ まとめ", "strategy.js に写す形", "line: { n:"):
            self.assertIn(tag, out)
        sh = s["shapes"]
        self.assertEqual(sum(sh[k]["n"] for k, _ in E.SHAPES), sh["all"]["n"])
        self.assertEqual(sum(sh[k]["n"] for k in E.BORDER), sh["border"]["n"])
        t = s["screen"]
        for fam, _, _ in E.FAMILIES:
            self.assertEqual(sum(t[fam][a]["n"] for a in E.GROUPS["K15"]), t[fam]["all"]["n"])
        self.assertEqual(t["line"]["n"], sh["all95"]["n"])
        self.assertEqual(set(s["reds"]), {f"{fam}・一番低い {a}" for fam, _, _ in E.FAMILIES for a in E.GROUPS["K15"]
                                          if E.warn_of(t[fam][a])})
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
