#!/usr/bin/env python3
"""
実験53（research/exp/e53_objective_tuning.py）のテスト。

- 探索空間は tuning.tune と同じ範囲で、目的関数と不均衡の補正を含まない
- 渡した木の形が fit_predict に届く（本番の木と違う結果になる）
- 発火数を C にそろえた選定は、窓ごとに C の件数と同じ件数を上から取る
- 探索の目的（上位10% の平均収益）は分割ごとに計算される。2試行の探索が通り、記録の形が整う

  python3 tests/test_e53_objective_tuning.py
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

import e52_return_objective as E52  # noqa: E402
import e53_objective_tuning as E  # noqa: E402
import tuning  # noqa: E402


def frame(n_dates=60, per_day=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.bdate_range("2020-01-01", periods=n_dates)):
        for j in range(per_day):
            x1, x2 = rng.normal(), rng.normal()
            r = 0.05 * x1 + rng.normal(0, 0.05)
            rows.append({"Date": d, "Code": f"{1000 + j}", "x1": x1, "x2": x2,
                         "ret_o1_20": r, "label": int(r > 0.04),
                         "log_market_cap": float(rng.normal(10, 1))})
    return pd.DataFrame(rows)


class _Trial:
    """optuna の trial の代わり（範囲の中央を返す）。"""
    def suggest_float(self, name, lo, hi, log=False):
        return float(np.sqrt(lo * hi)) if log else (lo + hi) / 2

    def suggest_int(self, name, lo, hi, log=False):
        return int(round(np.sqrt(lo * hi))) if log else (lo + hi) // 2


class TestSearchSpace(unittest.TestCase):
    def test_same_ranges_as_tuning_no_objective(self):
        p = E.search_params(_Trial())
        self.assertNotIn("objective", p)
        self.assertNotIn("scale_pos_weight", p)
        self.assertEqual(p["n_estimators"], tuning.SEARCH_N_ESTIMATORS)
        lo, hi = tuning.lr_range(tuning.SEARCH_N_ESTIMATORS)
        self.assertTrue(lo <= p["learning_rate"] <= hi)
        self.assertTrue(7 <= p["num_leaves"] <= 127)
        self.assertTrue(10 <= p["min_child_samples"] <= 300)

    def test_params_reach_fit_predict(self):
        df = frame()
        tr, te = df.iloc[:400], df.iloc[400:]
        a = E52.fit_predict("R", tr, te, ["x1", "x2"], 0)
        p = {"n_estimators": 20, "num_leaves": 4, "learning_rate": 0.3, "min_child_samples": 5,
             "objective": "binary", "scale_pos_weight": 3.0}
        b = E52.fit_predict("R", tr, te, ["x1", "x2"], 0, p)
        self.assertEqual(len(b), len(te))
        self.assertGreater(np.abs(a - b).max(), 1e-6)
        # objective / scale_pos_weight は腕が決める（渡しても外れる）
        c = E52.fit_predict("C", tr, te, ["x1", "x2"], 0, p)
        self.assertTrue(((c >= 0) & (c <= 1)).all())


class TestMatchedPicks(unittest.TestCase):
    def test_counts_follow_c(self):
        rng = np.random.default_rng(1)
        n = 3000
        o_c = pd.DataFrame({"fold": np.repeat([1, 2, 3], n // 3), "score": rng.uniform(size=n),
                            "ret_o1_20": rng.normal(0, 0.1, n), "label": rng.integers(0, 2, n)})
        counts = E.threshold_counts(o_c, 90)
        self.assertNotIn(1, counts)              # 窓1は参照分布が無い
        self.assertEqual(set(counts), {2, 3})
        o_a = o_c.copy()
        o_a["score"] = rng.uniform(size=n)
        m = E.matched_picks(o_a, counts)
        self.assertEqual(m["n"].tolist(), [counts[2], counts[3]])
        # 上から取っている（選ばれた最小スコア ≥ 選ばれなかった最大スコア）
        cur = o_a[o_a["fold"] == 2].sort_values("score", ascending=False)
        self.assertGreaterEqual(cur["score"].iloc[counts[2] - 1], cur["score"].iloc[counts[2]])

    def test_top_mean(self):
        s = np.arange(100, dtype=float)
        r = np.where(s >= 90, 0.2, -0.2)
        t = E.top_mean(s, r, 90)
        self.assertEqual(t["n"], 10)             # 90 パーセンタイル（89.1）を超える 90〜99 の10件
        self.assertAlmostEqual(t["ret"], 0.2)
        self.assertEqual(t["bad"], 0.0)


class TestTune(unittest.TestCase):
    def test_two_trials(self):
        df = frame(n_dates=120, per_day=12)
        rec = E.tune_arm("Q", df, ["x1", "x2"], n_trials=2)
        self.assertIn("params", rec)
        self.assertEqual(rec["params"]["n_estimators"], tuning.SEARCH_N_ESTIMATORS)
        self.assertNotIn("objective", rec["params"])
        cv = rec["_cv"]
        self.assertEqual(len(cv["top_ret_folds"]), E.N_SPLITS)
        self.assertEqual(len(cv["trials"]), 2)
        self.assertIn("prod_tree", cv)
        self.assertTrue(np.isfinite(cv["top_ret"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
