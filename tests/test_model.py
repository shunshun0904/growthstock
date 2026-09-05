#!/usr/bin/env python3
"""
research/train_model.py の評価ロジックの単体テスト。

特徴量を足すたびに「ベースラインをわずかに上回った」が出る。
その差が誤差かどうかを判定するのがブートストラップなので、
判定器そのものが壊れていないことを既知のケースで固定する。

  python3 tests/test_model.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from train_model import clean_score, evaluate, paired_bootstrap  # noqa: E402


def _synthetic(n=8000, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    return y, rng


class TestCleanScore(unittest.TestCase):
    def test_missing_becomes_worst(self):
        out = clean_score(np.array([3.0, np.nan, 1.0, np.inf]))
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.argmax(), 0)          # 3.0 が最良のまま
        self.assertLess(out[1], out[2])            # 欠測は最下位の 1.0 より下
        self.assertLess(out[3], out[2])

    def test_all_missing_does_not_crash(self):
        out = clean_score(np.array([np.nan, np.nan]))
        self.assertTrue(np.isfinite(out).all())

    def test_missing_never_ranks_above_a_real_score(self):
        """欠測を高評価にしてしまうと、予測できない銘柄を推奨してしまう。"""
        y = np.array([0, 1, 0, 0])
        r = evaluate("t", y, np.array([np.nan, 0.9, 0.1, 0.2]))
        self.assertEqual(r["precision@1%"], 1.0)   # 上位1件は本物の正例


class TestPairedBootstrap(unittest.TestCase):
    """既知の答えがあるケースで、判定が正しく出ることを固定する。"""

    def test_clearly_better_model_is_significant(self):
        y, rng = _synthetic(seed=1)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        strong = y * 2.0 + rng.normal(0, 1, len(y))
        out = paired_bootstrap(y, {"ref": ref, "strong": strong},
                               reference="ref", n_boot=200, seed=1)
        r = out[0]
        self.assertGreater(r["ci_low"], 0)
        self.assertGreater(r["p_better"], 0.99)

    def test_identical_scores_give_exactly_zero(self):
        """対応のある比較になっていれば、同じスコアの差は厳密に0になる。
        ここが0でなければリサンプルが対になっていない。"""
        y, rng = _synthetic(seed=2)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        out = paired_bootstrap(y, {"ref": ref, "copy": ref.copy()},
                               reference="ref", n_boot=100, seed=2)
        r = out[0]
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["ci_low"], 0.0)
        self.assertEqual(r["ci_high"], 0.0)

    def test_pure_noise_is_significantly_worse(self):
        y, rng = _synthetic(seed=3)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        out = paired_bootstrap(y, {"ref": ref, "noise": rng.normal(0, 1, len(y))},
                               reference="ref", n_boot=200, seed=3)
        self.assertLess(out[0]["ci_high"], 0)

    def test_tiny_difference_is_not_called_significant(self):
        """本命のケース。ほぼ同じ2つのモデルを『有意』と言わないこと。"""
        y, rng = _synthetic(seed=4)
        ref = y * 0.5 + rng.normal(0, 1, len(y))
        nudged = ref + rng.normal(0, 0.01, len(y))   # ごくわずかに違うだけ
        out = paired_bootstrap(y, {"ref": ref, "nudged": nudged},
                               reference="ref", n_boot=300, seed=4)
        r = out[0]
        self.assertLessEqual(r["ci_low"], 0)
        self.assertGreaterEqual(r["ci_high"], 0)

    def test_resamples_without_positives_are_redrawn(self):
        """正例が極端に少なくても NaN を返さない。"""
        rng = np.random.default_rng(5)
        y = np.zeros(500, dtype=int)
        y[:3] = 1
        ref = rng.normal(0, 1, 500)
        out = paired_bootstrap(y, {"ref": ref, "b": rng.normal(0, 1, 500)},
                               reference="ref", n_boot=50, seed=5)
        for key in ("diff", "ci_low", "ci_high", "p_better"):
            self.assertFalse(np.isnan(out[0][key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
