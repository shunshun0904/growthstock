#!/usr/bin/env python3
"""
選定の基準を 1.2σ の3本 95以上にするときの画面の文言用の数字（research/exp/probe_t3_panel.py）。

- 画面の物差しの帯: 発火の帯の件数の合計が行全体と一致し、惜しい帯は最小 85〜95・発火8件以上の日だけ
- 前向き: 発火7件以下と8件以上の件数の合計が行全体と一致する。3枠の形は発火8件以上の日だけのほうが取引が多くならない
- 際どい形（1.2σ の3本）の件数の合計が形ごとの件数と一致し、画面の BORDER_STATS の形（lgbm / xgb / cat）で書ける
- 報告に銘柄コードが出ない

  python3 tests/test_probe_t3_panel.py
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

import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import probe_t3_panel as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


class TestPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = T83.synth_with_exits()
        split = E79.window_split(cls.df, folds)
        parts = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
        ft = E.E87.build(cls.df, parts)
        for code, _ in E.CODES:
            ft = E.PKB.add_spread(ft, parts, code)
        cls.ft, cls.parts = ft, parts
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_codes_and_line(self):
        self.assertEqual([c for c, _ in E.CODES], ["T3", "K15"])
        self.assertEqual(E.LINE, LT.AGREE_PCT)
        groups = dict((c, g) for c, _, g in E.E87.SETS)
        self.assertEqual(tuple(groups["T3"]), LT.RULE_MODELS)      # 1.2σ の3本 = 今の規則（2026-10-23 から）
        self.assertEqual(tuple(groups["K15"]), LT.K15)

    def test_panel_bands_add_up(self):
        for code, _ in E.CODES:
            p = E.panel(self.ft, code)
            n_rows = int(E.PKB.rule_mask(self.ft, code).sum())
            self.assertEqual(p["all"]["n"], n_rows)
            self.assertEqual(sum(s["n"] for s in p["bands"].values()), n_rows)
            mn = self.ft[f"min_{code}"].to_numpy(dtype=float)
            nb = self.ft["n_break"].to_numpy()
            near = (mn >= 85.0) & (mn < E.LINE) & (nb >= LT.PREV_SKIP_BREAKS)
            self.assertEqual(p["near"]["narrow"]["n"] + p["near"]["wide"]["n"], int(near.sum()))

    def test_forward_bands_and_slots(self):
        f = E.forward(self.ft, self.bar_days, "T3")
        self.assertEqual(f["quiet"]["n"] + f["busy"]["n"], f["rows"]["n"])
        # 発火8件以上の日だけの3枠の取引は、その日の行（規則の3本 95以上・発火8件以上）を超えない
        self.assertLessEqual(f["slots_busy8"]["n"], f["busy"]["n"])
        for key in ("slots_any", "slots_busy8"):
            self.assertIn("capital_pm", f[key])
            if f[key]["n"]:
                self.assertTrue(np.isfinite(f[key]["mean"]))

    def test_border_family_matches_screen_keys(self):
        ftx = E.PKX.add_shapes(self.ft, self.parts)
        t = E.PKX.family_table(ftx, "T3")
        for fam in ("two95", "min"):
            self.assertEqual(sum(t[fam][a]["n"] for a in LT.RULE_MODELS), t[fam]["all"]["n"])
        js = "\n".join(E.PKX.js_lines(t, "T3"))
        for a in LT.RULE_MODELS:
            self.assertIn(f" {a}: {{ n: ", js)                   # strategy.js の BORDER_STATS の鍵（lgbm / xgb / cat）
        self.assertNotIn("_k15", js)

    def test_report_has_no_codes(self):
        codes = set(self.df["Code"].astype(str))
        text = E.short(E.PKB.stats(self.ft[E.PKB.rule_mask(self.ft, "T3")]))
        p = E.panel(self.ft, "T3")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.print_panel(p, "1.2σ の3本 95以上", E.period_of(self.ft))
            E.print_forward(E.forward(self.ft, self.bar_days, "T3"), "1.2σ の3本 95以上")
        out = buf.getvalue() + text
        self.assertFalse(any(c in out for c in codes if len(c) >= 4))


if __name__ == "__main__":
    unittest.main()
