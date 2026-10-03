#!/usr/bin/env python3
"""
実験60（research/exp/e60_model_tendency.py）の数え方のテスト。重い計算は回さない。

- MLP の対数オッズと入力勾配の手計算が scikit-learn の predict_proba と数値微分に一致する
- 積分勾配の合計が f(Z) − f(0) に一致する（完全性）
- 前処理の出力列 → 元の列 の対応（出力名から / 揺らして）が one-hot と欠損指示子を元の列に戻す
- ロジスティック回帰の寄与（係数×値）を元の列に畳むと decision_function と一致する
- 置換の物差し: 使っていない列を入れ替えても順位は崩れない、使っている列なら崩れる
- 単独の AUC

  python3 tests/test_e60_model_tendency.py
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import tuning_multi as TM  # noqa: E402
import e60_model_tendency as E  # noqa: E402

warnings.filterwarnings("ignore")


def toy(n=600, seed=0):
    """カテゴリ1列 + 数値4列（1列は欠損あり）。ラベルは数値列の線形和。"""
    rng = np.random.default_rng(seed)
    cols = ["s33_code", "ROE_q0", "mom_20d", "vol_20d", "alert_slratio"]
    X = np.column_stack([rng.choice([1050, 2050, 3050], n), rng.standard_normal(n),
                         rng.standard_normal(n), rng.gamma(2, 1, n), rng.standard_normal(n)]).astype(float)
    X[rng.random(n) < 0.3, 4] = np.nan
    logit = 1.5 * X[:, 1] - 1.0 * X[:, 2] + 0.3 * (X[:, 0] == 2050)
    y = (logit + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return cols, X, y


class MLPGrad(unittest.TestCase):
    def setUp(self):
        self.cols, self.X, self.y = toy()
        self.m = TM.build("mlp", {"h1": 8, "two_layers": True, "alpha": 0.01, "learning_rate_init": 1e-2,
                                  "batch_size": 64}, self.y, self.cols, prep="v1")
        self.m.fit(self.X, self.y)
        self.ct, self.clf = self.m.steps[0][1], self.m.steps[-1][1]
        self.Z = E.transform(self.ct, self.X[:50])

    def test_logit_matches_predict_proba(self):
        out, _ = E.mlp_logit_and_grad(self.clf, self.Z)
        p = self.clf.predict_proba(self.Z)[:, 1]
        np.testing.assert_allclose(1 / (1 + np.exp(-out)), p, atol=1e-10)

    def test_gradient_matches_finite_difference(self):
        _, G = E.mlp_logit_and_grad(self.clf, self.Z)
        eps = 1e-6
        for k in (0, 3, self.Z.shape[1] - 1):
            Zp, Zm = self.Z.copy(), self.Z.copy()
            Zp[:, k] += eps
            Zm[:, k] -= eps
            fd = (E.mlp_logit_and_grad(self.clf, Zp)[0] - E.mlp_logit_and_grad(self.clf, Zm)[0]) / (2 * eps)
            np.testing.assert_allclose(G[:, k], fd, atol=1e-5)

    def test_integrated_gradients_completeness(self):
        f1, _ = E.mlp_logit_and_grad(self.clf, self.Z)
        f0, _ = E.mlp_logit_and_grad(self.clf, np.zeros_like(self.Z))
        # relu の折れ目をまたぐぶんの誤差は段数を増やすと縮む（完全性）
        err = {s: np.abs(E.integrated_gradients(self.clf, self.Z, steps=s).sum(axis=1) - (f1 - f0)).max()
               for s in (8, 64, 512)}
        self.assertLess(err[512], err[8])
        np.testing.assert_allclose(E.integrated_gradients(self.clf, self.Z, steps=512).sum(axis=1), f1 - f0,
                                   rtol=0.02, atol=0.02)


class Owners(unittest.TestCase):
    def setUp(self):
        self.cols, self.X, self.y = toy()

    def test_v1_outputs_map_back_to_columns(self):
        m = TM.build("logit", {"C": 1.0, "penalty": "l2"}, self.y, self.cols, prep="v1")
        m.fit(self.X, self.y)
        ct = m.steps[0][1]
        own = E.output_owners(ct, self.X, self.cols, n_probe=300)
        Z = E.transform(ct, self.X[:5])
        self.assertEqual(len(own), Z.shape[1])
        self.assertTrue((own >= 0).all())
        counts = np.bincount(own, minlength=len(self.cols))
        self.assertEqual(counts[0], 3)                       # one-hot 3カテゴリ
        self.assertEqual(counts[4], 2)                       # 値 + 欠損指示子
        self.assertEqual(counts[1], 1)

    def test_probe_fallback_without_names(self):
        m = TM.build("logit", {"C": 1.0, "penalty": "l2"}, self.y, self.cols, prep="v1")
        m.fit(self.X, self.y)
        ct = m.steps[0][1]

        class NoNames:
            def __init__(self, inner):
                self.inner = inner

            def transform(self, X):
                return self.inner.transform(X)

        own = E.output_owners(NoNames(ct), self.X, self.cols, n_probe=300)
        want = E.output_owners(ct, self.X, self.cols, n_probe=300)
        np.testing.assert_array_equal(own, want)

    def test_logit_contributions_sum_to_decision(self):
        m = TM.build("logit", {"C": 1.0, "penalty": "l2"}, self.y, self.cols, prep="v1")
        m.fit(self.X, self.y)
        A = E.attributions("logit", m, self.X[:100], self.cols)
        self.assertEqual(A.shape, (100, len(self.cols)))
        want = m.decision_function(self.X[:100]) - m.steps[-1][1].intercept_[0]
        np.testing.assert_allclose(A.sum(axis=1), want, atol=1e-8)

    def test_tree_shap_sums_to_raw(self):
        m = TM.build("lgbm", {"num_leaves": 7, "learning_rate": 0.1}, self.y, self.cols)
        m.fit(self.X, self.y)
        A = E.tree_shap("lgbm", m.booster_, self.X[:50])
        raw = m.booster_.predict(self.X[:50], raw_score=True)
        bias = m.booster_.predict(self.X[:50], pred_contrib=True)[:, -1]
        np.testing.assert_allclose(A.sum(axis=1) + bias, raw, atol=1e-6)


class Reliance(unittest.TestCase):
    def test_permutation_reliance(self):
        cols, X, y = toy()
        m = TM.build("logit", {"C": 1.0, "penalty": "l2"}, y, cols, prep="v1")
        m.fit(X, y)
        base = E.raw_score("logit", m, X)
        r = E.permutation_reliance("logit", m, X, base, {"used": [1], "unused": [3]})
        self.assertGreater(r["used"], 0.3)                   # ROE はラベルを決めている列
        self.assertLess(r["unused"], r["used"])

    def test_univariate_auc(self):
        cols, X, y = toy()
        auc = E.univariate_auc(X, y)
        self.assertGreater(auc[1], 0.75)                     # ROE は正の向き
        self.assertLess(auc[2], 0.35)                        # mom は負の向き
        self.assertTrue(np.isfinite(auc[4]))                 # 欠損があっても出る

    def test_fold_to_columns_drops_unowned(self):
        cz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        own = np.array([0, 0, -1])
        np.testing.assert_array_equal(E.fold_to_columns(cz, own, 2), [[3.0, 0.0], [9.0, 0.0]])

    def test_spearman(self):
        self.assertAlmostEqual(E.spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])), 1.0)
        self.assertTrue(np.isnan(E.spearman(np.array([1, 1, 1.0]), np.array([1, 2, 3.0]))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
