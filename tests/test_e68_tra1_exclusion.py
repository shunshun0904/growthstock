#!/usr/bin/env python3
"""実験68（research/exp/e68_tra1_exclusion.py）の部品のテスト。"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e68_tra1_exclusion as E68  # noqa: E402


def population(n_fold=4, n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fold):
        for i in range(n):
            rows.append({"Code": f"{i:05d}", "Date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=f * 200 + i % 100),
                         "fold": f, "x": rng.normal(f, 1.0)})   # 窓ごとに分布が動く
    return pd.DataFrame(rows)


class TestExcludeMask(unittest.TestCase):
    def test_threshold_uses_only_prior_windows(self):
        ref = population()
        sel = ref.copy()
        m = E68.exclude_mask(sel, ref, "x", "top", 90.0)
        # 最初の窓は参照が無いので誰も外さない
        self.assertFalse(m[sel["fold"].to_numpy() == 0].any())
        # 2つ目の窓のしきい値は窓0 の上位10%（平均0の分布）→ 平均1の窓1 では多くが外れる
        f1 = sel["fold"].to_numpy() == 1
        self.assertGreater(m[f1].mean(), 0.3)
        # 全窓の分布で切った場合より多く外れる（前の窓だけを見ている証拠）
        thr_all = np.percentile(ref["x"], 90)
        self.assertGreater(m[f1].mean(), (sel.loc[f1, "x"] >= thr_all).mean())

    def test_bottom_side_and_missing_values(self):
        ref = population()
        sel = ref.copy()
        sel.loc[sel.index[:50], "x"] = np.nan
        m = E68.exclude_mask(sel, ref, "x", "bottom", 10.0)
        self.assertFalse(m[:50].any(), "値が無い行は外さない")
        self.assertTrue(m.any())


class TestEvaluate(unittest.TestCase):
    def test_counts_and_direction(self):
        ref = population()
        sel = ref.copy()
        rng = np.random.default_rng(1)
        # x が高いほど収益が低い世界
        sel["ret_o1_20"] = -0.02 * sel["x"] + rng.normal(0, 0.01, len(sel))
        sel["label"] = (sel["ret_o1_20"] > 0).astype(float)
        res = E68.evaluate(sel, ref, col="x")
        top = res[res["rule"] == "上位10% を外す"].iloc[0]
        self.assertEqual(int(top["n"]), len(sel))
        self.assertGreater(top["diff"], 0)                 # 高い側を外すと残りの平均が上がる
        self.assertLess(top["excl_mean"], top["before"])   # 外した側は平均より悪い
        ctrl = res[res["rule"] == "対照: 下位10% を外す"].iloc[0]
        self.assertLess(ctrl["diff"], 0)                   # 低い側を外すと下がる
        self.assertTrue((res["won"] <= res["nf"]).all())


if __name__ == "__main__":
    unittest.main()
