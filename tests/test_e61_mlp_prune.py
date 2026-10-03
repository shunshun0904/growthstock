#!/usr/bin/env python3
"""
実験61（research/exp/e61_mlp_prune.py）の列の選び方のテスト。重い計算は回さない。

- 単独の AUC が 0.5 ± 0.02 の列が「弱い」
- A は本番 MLP の寄与の上位のうち弱い列だけ外す、B は弱い列を全部外す、C は置換の崩れの下位半分を外す
- どの腕も列の順は本番の順のまま。P は全列

  python3 tests/test_e61_mlp_prune.py
"""
import os
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import tuning_multi as TM  # noqa: E402
import e61_mlp_prune as E  # noqa: E402

warnings.filterwarnings("ignore")


def toy(n=4000, seed=0):
    """数値6列。a・b はラベルを決める、c・d は無関係、e は a のコピー寄り、f はノイズ。"""
    rng = np.random.default_rng(seed)
    cols = ["ROE_q0", "mom_20d", "noise_c", "noise_d", "ROE_q1", "noise_f"]
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    X = np.column_stack([a, b, rng.standard_normal(n), rng.standard_normal(n), a + 0.3 * rng.standard_normal(n),
                         rng.standard_normal(n)])
    y = (1.5 * a - 1.0 * b + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return cols, X, y


class Select(unittest.TestCase):
    def setUp(self):
        self.cols, self.X, self.y = toy()
        self.m = TM.build("mlp", {"h1": 8, "two_layers": False, "alpha": 0.01, "learning_rate_init": 1e-2,
                                  "batch_size": 64}, self.y, self.cols, prep="v1")
        self.m.fit(self.X, self.y)

    def test_weak_columns(self):
        weak, auc = E.weak_columns(self.cols, self.X, self.y)
        self.assertNotIn("ROE_q0", weak)
        self.assertNotIn("mom_20d", weak)
        self.assertNotIn("ROE_q1", weak)                     # a の写しも効く
        self.assertGreaterEqual(len(weak), 2)                # ノイズ3列のうち2列以上は弱い帯に入る
        self.assertTrue(set(weak) <= {"noise_c", "noise_d", "noise_f"})
        self.assertGreater(auc[0], 0.7)

    def test_arms_keep_order_and_drop_as_defined(self):
        arms, table = E.select(self.cols, self.X, self.y, self.m, top_n=3)
        self.assertEqual(arms["P"], self.cols)
        for a in ("A", "B", "C"):
            self.assertEqual(arms[a], [c for c in self.cols if c in set(arms[a])])   # 順は本番のまま
        # B: 弱い列を全部外し、効く列は残る（ノイズ列は偶然 0.5±0.02 を外れうるので、判定どおりかで見る）
        self.assertTrue({"ROE_q0", "mom_20d"} <= set(arms["B"]))
        weak = set(table.loc[table["weak"], "col"])
        self.assertEqual(set(arms["B"]), set(self.cols) - weak)
        self.assertTrue(weak <= {"noise_c", "noise_d", "noise_f"})
        self.assertGreaterEqual(len(weak), 2)
        # A: 寄与の上位3列のうち弱い列だけ → 上位3列はほぼ効く列なので、外すのは高々1列
        self.assertGreaterEqual(len(arms["A"]), len(self.cols) - 1)
        # C: 置換の崩れの下位半分（6列なら 3列）を外し、効く列は残る
        self.assertEqual(len(arms["C"]), 3)
        self.assertTrue({"ROE_q0", "mom_20d"} <= set(arms["C"]))
        self.assertEqual(set(table.columns) >= {"col", "uni_auc", "share", "perm", "weak", "over", "unused"}, True)
        self.assertEqual(int(table["unused"].sum()), 3)

    def test_labels_and_constants(self):
        self.assertEqual(E.ARMS, ("P", "A", "B", "C", "At", "Ct"))
        self.assertEqual(E.TUNE_SELECT, {"At": "A", "Ct": "C"})
        self.assertTrue(set(E.LABELS) == set(E.ARMS))
        self.assertEqual(E.ALGO, "mlp")
        self.assertEqual(E.PREP, "v1")
        self.assertEqual(TM.preprocess_version("mlp"), "v1")

    def test_fit_mlp_is_usable_for_selection_and_reproducible(self):
        import pandas as pd
        df = pd.DataFrame(self.X, columns=self.cols)
        df["label"] = self.y
        params = {"h1": 8, "two_layers": False, "alpha": 0.01, "learning_rate_init": 1e-2, "batch_size": 64}
        m1 = E.fit_mlp(df, self.cols, params, seed=42)
        m2 = E.fit_mlp(df, self.cols, params, seed=42)
        self.assertEqual(TM.SEED, 0)                        # 学習のあと種を戻す
        p1 = m1.predict_proba(self.X)[:, 1]
        p2 = m2.predict_proba(self.X)[:, 1]
        np.testing.assert_allclose(p1, p2)                  # 同じ種なら同じモデル
        arms, table = E.select(self.cols, self.X, self.y, m1, top_n=3)
        self.assertEqual(len(arms["C"]), 3)
        self.assertTrue({"ROE_q0", "mom_20d"} <= set(arms["C"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
