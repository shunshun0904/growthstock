#!/usr/bin/env python3
"""
選定の基準を 1.5σ の3本 95以上に戻すときの画面の文言用の数字（research/exp/probe_k15_panel.py）。

- 6本（1.2σ・1.5σ の木）だけで作った行の、規則の3本の最小・最大は、12本で作った行と同じ（百分位はモデルごとなので、ほかの
  モデルを読まなくても変わらない）
- 際どい形は 1.5σ の3本（lgbm_k15 / xgb_k15 / cat_k15）の鍵で、画面の BORDER_STATS の形で書ける
- まとめ: 前向きの行全体と画面の3枠の形の平均が、どちらも 1.5σ の3本のほうが上のときだけ「上」
- 通しで回す（読み込みは合成データに差し替え）: 報告の見出しがそろい、銘柄コードが出ない

  python3 tests/test_probe_k15_panel.py
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import probe_k15_panel as P  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


class TestProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.folds = T83.synth_with_exits()
        cls.ft, cls.ff, cls.parts = P.frames(cls.df, cls.folds)
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_models_and_sets(self):
        self.assertEqual(P.SIX, ("lgbm", "xgb", "cat", "lgbm_k15", "xgb_k15", "cat_k15"))
        self.assertEqual(sorted(c for c, _, _ in P.SETS), ["K15", "T3"])
        groups = dict((c, g) for c, _, g in P.SETS)
        self.assertEqual(tuple(groups["K15"]), LT.K15)
        self.assertEqual(P.LINE, LT.AGREE_PCT)

    def test_same_as_twelve_models(self):
        split = E79.window_split(self.df, self.folds)
        parts12 = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
        ft12 = P.E87.build(self.df, parts12)
        for code, _ in P.CODES:
            ft12 = P.PKB.add_spread(ft12, parts12, code)
            for c in (f"min_{code}", f"max_{code}"):
                np.testing.assert_array_equal(self.ft[c].to_numpy(dtype=float), ft12[c].to_numpy(dtype=float), err_msg=c)
        pd.testing.assert_frame_equal(self.ft[["Code", "Date", "n_break", "tp10c"]], ft12[["Code", "Date", "n_break", "tp10c"]])
        self.assertGreater(int(P.PKB.rule_mask(self.ft, "K15").sum()), 0)

    def test_border_keys(self):
        ftx = P.PKX.add_shapes(self.ft, self.parts, models=P.SIX)
        t = P.PKX.family_table(ftx, "K15")
        for fam in ("two95", "min"):
            self.assertEqual(sum(t[fam][a]["n"] for a in LT.K15), t[fam]["all"]["n"])
        js = "\n".join(P.PKX.js_lines(t, "K15"))
        for a in LT.K15:
            self.assertIn(f" {a}: {{ n: ", js)
        # 1.2σ の3本（2026-10-30 に規則を戻すときの BORDER_STATS）も同じ形で書ける
        t3 = P.PKX.family_table(ftx, "T3")
        for fam in ("two95", "min"):
            self.assertEqual(sum(t3[fam][a]["n"] for a in LT.BOOST), t3[fam]["all"]["n"])
        js3 = "\n".join(P.PKX.js_lines(t3, "T3"))
        for a in LT.BOOST:
            self.assertIn(f" {a}: {{ n: ", js3)

    def test_summary(self):
        def fwd(k_rows, k_slots, t_rows, t_slots):
            return {"K15": {"rows": {"mean": k_rows}, "slots_any": {"mean": k_slots}},
                    "T3": {"rows": {"mean": t_rows}, "slots_any": {"mean": t_slots}}}
        self.assertTrue(P.summary({}, fwd(2.7, 1.6, 2.6, 0.2))["k15_better"])
        self.assertFalse(P.summary({}, fwd(2.7, 0.1, 2.6, 0.2))["k15_better"])   # 3枠で下
        self.assertFalse(P.summary({}, fwd(2.5, 1.6, 2.6, 0.2))["k15_better"])   # 行全体で下

    def test_main_report(self):
        info0 = {"kept": len(self.df)}
        chk = {"with_close": len(self.df), "rows": len(self.df), "order_violations": 0}
        buf = io.StringIO()
        with mock.patch.object(P, "load", return_value=(self.df, self.folds, self.bar_days, info0, chk)), \
                contextlib.redirect_stdout(buf):
            self.assertEqual(P.main([]), 0)
        out = buf.getvalue()
        for h in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4b.", "■ 4d.", "■ 5.", "js:", "js3:", "1.5σ の3本 95以上（新しい規則）"):
            self.assertIn(h, out)
        codes = set(self.df["Code"].astype(str))
        self.assertFalse(any(c in out for c in codes if len(c) >= 4))


if __name__ == "__main__":
    unittest.main()
