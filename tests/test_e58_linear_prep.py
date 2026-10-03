#!/usr/bin/env python3
"""
実験58（research/exp/e58_linear_prep.py）の骨組みのテスト。重い計算は回さない。

- 腕は P（v1）と V（v2）で、前処理の版の対応が固定されている
- prep() は tuning_multi.PREPROCESS（とモデルごとの設定）を一時的に切り替え、例外が出ても必ず戻す
- width() は v1 / v2 の前処理後の列数を返す（one-hot と指示子を含む）
- pair_line() の差・SE・勝ち数の数え方

  python3 tests/test_e58_linear_prep.py
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

import features as F  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e58_linear_prep as E  # noqa: E402


class TestArms(unittest.TestCase):
    def test_arms_and_versions(self):
        self.assertEqual(E.ARMS, ("P", "V"))
        self.assertEqual(E.PREP, {"P": "v1", "V": "v2"})
        self.assertEqual(E.ALGOS, ("logit", "mlp"))
        self.assertEqual(E.SEEDS["logit"], (42,))
        self.assertEqual(len(E.SEEDS["mlp"]), 3)

    def test_prep_restores_the_module_setting(self):
        self.assertEqual(TM.PREPROCESS, "v1")
        by = dict(TM.PREPROCESS_BY_ALGO)
        with E.prep("v2"):
            self.assertEqual(TM.PREPROCESS, "v2")
            self.assertEqual(TM.PREPROCESS_BY_ALGO, {})          # 腕の版を両モデルに強制する
            self.assertEqual(TM.preprocess_version("mlp"), "v2")
        with E.prep("v1"):
            self.assertEqual(TM.preprocess_version("logit"), "v1")  # 本番の logit=v2 を外す
        self.assertEqual(TM.PREPROCESS, "v1")
        self.assertEqual(TM.PREPROCESS_BY_ALGO, by)
        with self.assertRaises(RuntimeError):
            with E.prep("v2"):
                raise RuntimeError("途中で落ちる")
        self.assertEqual(TM.PREPROCESS, "v1")
        self.assertEqual(TM.PREPROCESS_BY_ALGO, by)

    def test_width_counts_onehot_and_indicators(self):
        cols = ["s33_code", "ROE_q0", "has_dividend", "alert_slratio"]
        rng = np.random.default_rng(0)
        X = np.column_stack([rng.integers(0, 3, 50), rng.standard_normal(50),
                             rng.integers(0, 2, 50), rng.gamma(2, 2, 50)]).astype(float)
        X[:10, 3] = np.nan                                    # alert_slratio に欠損 -> 指示子が1本
        w1 = E.width(cols, "v1", X)
        w2 = E.width(cols, "v2", X)
        # one-hot 3 + 数値 3 + 指示子 1 = 7（v1 も v2 も同じ）
        self.assertEqual(w1, 7)
        self.assertEqual(w2, 7)
        self.assertEqual(TM.PREPROCESS, "v1")                 # width() が設定を戻している

    def test_pair_line(self):
        line = E.pair_line("x PR-AUC", np.array([0.20, 0.30, 0.25]), np.array([0.22, 0.29, 0.27]))
        self.assertIn("2/3", line)                            # 上回った窓の数
        self.assertIn("+0.0100", line)                        # 差の平均

    def test_production_columns_are_all_classified(self):
        import linear_preprocess as LP
        self.assertEqual(LP.unclassified(F.columns(F.DEFAULT_PRESET)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
