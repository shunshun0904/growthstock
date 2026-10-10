#!/usr/bin/env python3
"""実験72（research/exp/e72_nl_only.py）の部品のテスト。学習はしない。"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import ops_rule as OR  # noqa: E402
import e72_nl_only as E72  # noqa: E402


def _oof(seed: int, n_fold: int = 3, per: int = 600, strength: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fold):
        dates = pd.date_range("2024-01-01", periods=per, freq="D") + pd.Timedelta(days=f * per)
        s = rng.random(per)
        lab = (rng.random(per) < 0.2 + strength * s).astype(float)
        rows.append(pd.DataFrame({"Code": [str(1000 + i) for i in range(per)], "Date": dates, "fold": f,
                                  "label": lab, "score": s,
                                  "ret_o1_20": rng.normal(0.01, 0.1, per), "ret_o1_40": rng.normal(0.01, 0.15, per)}))
    return pd.concat(rows, ignore_index=True)


class TestTables(unittest.TestCase):
    def test_positive_rate_by_window(self):
        o = pd.DataFrame({"fold": [0, 0, 1, 1], "label": [1, 0, 1, 1]})
        p = E72.positive_rate_by_window(o)
        self.assertEqual(list(p["fold"]), [0, 1])
        np.testing.assert_allclose(p["pos"], [0.5, 1.0])

    def test_auc_table_has_all_arms_and_matches_pos(self):
        a = _oof(1)
        b = a.copy()
        b["score"] = 1 - b["score"]  # 逆順位
        tab = E72.auc_table({"T": a, "N": b})
        self.assertEqual(list(tab.columns), ["fold", "pos", "pr_T", "roc_T", "pr_N", "roc_N"])
        self.assertEqual(len(tab), 3)
        np.testing.assert_allclose(tab["pos"], a.groupby("fold")["label"].mean().to_numpy())
        self.assertTrue((tab["roc_T"] > 0.5).all())
        np.testing.assert_allclose(tab["roc_N"], 1 - tab["roc_T"])  # ROC は逆順位で 1 − x

    def test_pair_summary_identical_and_better(self):
        a = _oof(2)
        tab = E72.auc_table({"T": a, "N": a})
        s = E72.pair_summary(tab, "T", "N")
        self.assertEqual(s["pr"]["diff"], 0.0)
        self.assertEqual(s["pr"]["won"], 0)
        self.assertEqual(s["pr"]["n"], 3)
        b = a.copy()
        b["score"] = b["label"] + 0.01 * b["score"]  # ほぼ完全な順位
        tab2 = E72.auc_table({"T": a, "N": b})
        s2 = E72.pair_summary(tab2, "T", "N")
        self.assertGreater(s2["pr"]["diff"], 0)
        self.assertEqual(s2["pr"]["won"], 3)
        self.assertGreater(s2["roc"]["b"], 0.99)

    def test_precision_rows_match_consensus(self):
        a = _oof(3)
        t = E72.precision_rows({"T": a, "N": a})
        self.assertEqual(len(t), 4)
        row = t[(t["arm"] == "T") & (t["rule"] == "lgbm 単体 95以上")].iloc[0]
        c = OR.consensus({"lgbm": a}, 95.0, models=("lgbm",))
        self.assertEqual(int(row["n"]), c["n"])
        self.assertGreater(c["n"], 0)
        self.assertAlmostEqual(float(row["hit_own"]), c["label_rate"])
        self.assertAlmostEqual(float(row["ret20"]), c["ret20"])
        self.assertAlmostEqual(float(row["lift20"]), c["ret_o1_20"]["fold_mean"])
        self.assertTrue(0 <= row["win"] <= 1)

    def test_fmt_prec_handles_empty(self):
        self.assertIn("選定なし", E72.fmt_prec(pd.Series({"n": 0, "rule": "r"}), "x"))
        r = pd.Series({"n": 10, "rule": "r", "rate": 0.05, "hit_own": 0.4, "ret20": 2.5, "win": 0.6,
                       "big_up": 0.2, "big_down": 0.1, "lift20": 1.0, "won": 2, "folds": 3})
        line = E72.fmt_prec(r, "x")
        self.assertIn("40.0%", line)
        self.assertIn("2/3", line)

    def test_importance_is_normalized_and_sorted(self):
        rng = np.random.default_rng(0)
        n = 600
        x1 = rng.normal(size=n)
        df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=n, freq="D"),
                           "label": (x1 + 0.3 * rng.normal(size=n) > 0).astype(float),
                           "a": x1, "b": rng.normal(size=n), "c": rng.normal(size=n)})
        params = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10}
        imp = E72.importance(df, ["a", "b", "c"], params, df["Date"].max())
        self.assertAlmostEqual(float(imp.sum()), 1.0)
        self.assertEqual(imp.index[0], "a")
        self.assertTrue((imp.diff().dropna() <= 0).all())


if __name__ == "__main__":
    unittest.main()
