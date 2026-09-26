#!/usr/bin/env python3
"""
実験52（research/exp/e52_return_objective.py）の目的の作り方のテスト。

- 回帰の目的の刈り込みは訓練側の分位だけで決まる
- 日付内の百分位は 0〜1（1件だけの日は 0.5）、LTR の段階は 0〜4 の整数で日付内の順位に単調
- 木の形のパラメータから目的関数と不均衡の補正が外れている
- 4本の腕が学習でき、C は確率（0〜1）、ほかはスコア。LTR はグループの並びを守る

  python3 tests/test_e52_return_objective.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import e52_return_objective as E  # noqa: E402


class TestTargets(unittest.TestCase):
    def test_clip_uses_train_quantiles(self):
        r = np.r_[np.linspace(-0.3, 0.3, 98), 5.0, -2.0]
        y, lo, hi = E.clipped_return(r)
        self.assertLess(hi, 5.0)
        self.assertGreater(lo, -2.0)
        self.assertEqual(y.max(), hi)
        self.assertEqual(y.min(), lo)

    def test_within_date_pct_and_grade(self):
        dates = pd.Series(["d1", "d1", "d1", "d2", "d2", "d3"])
        r = np.array([0.1, -0.2, 0.3, 0.0, 0.5, 0.2])
        p = E.within_date_pct(dates, r)
        np.testing.assert_allclose(p, [0.5, 0.0, 1.0, 0.0, 1.0, 0.5])
        g = E.within_date_grade(dates, r)
        self.assertEqual(g.tolist(), [2, 0, 4, 0, 4, 2])
        self.assertTrue(((g >= 0) & (g < E.N_GRADES)).all())

    def test_tree_params_drop_objective_and_weight(self):
        p = E.tree_params(7)
        self.assertNotIn("objective", p)
        self.assertNotIn("scale_pos_weight", p)
        self.assertEqual(p["random_state"], 7)
        self.assertEqual(p["n_estimators"], 200)
        self.assertFalse(any(k.startswith("_") for k in p))


def frame(n_dates=60, per_day=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.bdate_range("2020-01-01", periods=n_dates)):
        for j in range(per_day):
            x1, x2 = rng.normal(), rng.normal()
            r = 0.05 * x1 + rng.normal(0, 0.05)
            rows.append({"Date": d, "Code": f"{1000 + j}", "x1": x1, "x2": x2,
                         "ret_o1_20": r, "label": int(r > 0.04)})
    return pd.DataFrame(rows)


class TestArms(unittest.TestCase):
    def test_all_arms_fit_and_predict(self):
        df = frame()
        tr, te = df.iloc[:400], df.iloc[400:]
        for arm in E.ARMS:
            s = E.fit_predict(arm, tr, te, ["x1", "x2"], seed=0)
            self.assertEqual(len(s), len(te), arm)
            self.assertTrue(np.isfinite(s).all(), arm)
            # 目的が x1 で決まる作りなので、どの腕も x1 と正の相関を持つ
            self.assertGreater(np.corrcoef(s, te["x1"])[0, 1], 0.3, arm)
        c = E.fit_predict("C", tr, te, ["x1", "x2"], seed=0)
        self.assertTrue(((c >= 0) & (c <= 1)).all())

    def test_ltr_keeps_group_order(self):
        """行を日付順に並べ替えてから group を作る（並びが崩れると lambdarank は別の問題を解く）。"""
        df = frame(n_dates=30)
        shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        te = df.iloc[:50]
        a = E.fit_predict("L", df, te, ["x1", "x2"], seed=0)
        b = E.fit_predict("L", shuffled, te, ["x1", "x2"], seed=0)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
