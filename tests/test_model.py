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

import pandas as pd  # noqa: E402

from train_model import clean_score, evaluate, paired_bootstrap  # noqa: E402
from walkforward import REFERENCE, make_folds, run, sign_test, summarize  # noqa: E402


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


class TestSignTest(unittest.TestCase):
    """符号検定。フォールド数が少ないので、何勝すれば有意かを把握しておく。"""

    def test_unanimous_is_significant(self):
        self.assertAlmostEqual(sign_test(8, 0), 2 / 256)
        self.assertLess(sign_test(9, 0), 0.05)

    def test_even_split_is_not_significant(self):
        self.assertEqual(sign_test(4, 4), 1.0)

    def test_symmetric(self):
        self.assertEqual(sign_test(7, 2), sign_test(2, 7))

    def test_nine_folds_needs_eight_wins(self):
        """9フォールドでは 8勝1敗 でようやく有意。7勝2敗では足りない。
        レポートの判定を読むときにこの厳しさを踏まえる必要がある。"""
        self.assertLess(sign_test(8, 1), 0.05)
        self.assertGreater(sign_test(7, 2), 0.05)

    def test_no_folds_is_nan(self):
        self.assertTrue(np.isnan(sign_test(0, 0)))


class TestMakeFolds(unittest.TestCase):
    def _folds(self, first="2017-09-29", last="2025-11-28", **kw):
        kw.setdefault("min_train_months", 36)
        kw.setdefault("test_months", 6)
        kw.setdefault("step_months", 6)
        kw.setdefault("embargo_days", 180)
        return make_folds(pd.Series(pd.to_datetime([first, last])), **kw)

    def test_test_windows_never_overlap(self):
        """重なると同じサンプルが2フォールドに入り、独立でなくなる。"""
        folds = self._folds()
        self.assertGreater(len(folds), 1)
        for a, b in zip(folds, folds[1:]):
            self.assertLess(a.test_end, b.test_start)

    def test_embargo_is_respected(self):
        """訓練最終日のラベルがテスト期間に食い込まないこと。"""
        for f in self._folds():
            gap = (pd.Timestamp(f.test_start) - pd.Timestamp(f.train_end)).days
            self.assertGreaterEqual(gap, int(180 * 1.45) - 1)

    def test_training_window_expands(self):
        folds = self._folds()
        ends = [pd.Timestamp(f.train_end) for f in folds]
        self.assertEqual(ends, sorted(ends))
        self.assertEqual(len({f.train_start for f in folds}), 1)

    def test_never_runs_past_the_data(self):
        for f in self._folds():
            self.assertLessEqual(pd.Timestamp(f.test_end), pd.Timestamp("2025-11-28"))

    def test_too_short_a_period_yields_no_folds(self):
        self.assertEqual(self._folds(last="2019-01-01"), [])


class TestWalkForwardEndToEnd(unittest.TestCase):
    """合成データで最後まで通ること。列名や集計の取り違えを検出する。"""

    def _dataset(self, n_dates=60, n_codes=120, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2017-09-29", periods=n_dates, freq="ME")
        rows = []
        for d in dates:
            r_high = rng.uniform(0, 95, n_codes)
            # r_high が高いほど正例になりやすい合成ラベル
            p = 1 / (1 + np.exp(-(r_high - 70) / 10))
            frame = {
                "Date": d,
                "Code": [f"{i:04d}" for i in range(n_codes)],
                "r_high": r_high,
                "r_high_r": rng.uniform(0, 1, n_codes),
                "volume_trend": rng.normal(0, 1, n_codes),
                "label": (rng.random(n_codes) < p).astype(int),
            }
            # ベースラインの8軸スコアが必要とする列
            for c in ("ROE_q0", "credit_ratio", "eps_growth_q0", "market_cap",
                      "op_margin_q0", "progress_vs_base", "sales_growth_q0",
                      "tv_ma20"):
                frame[c] = rng.normal(0, 1, n_codes)
            rows.append(pd.DataFrame(frame))
        return pd.concat(rows, ignore_index=True)

    def test_runs_and_summarises(self):
        df = self._dataset()
        folds = make_folds(pd.to_datetime(df["Date"]), min_train_months=24,
                           test_months=6, step_months=6, embargo_days=20)
        self.assertGreaterEqual(len(folds), 2)

        # 実在する列だけを使う最小プリセットを注入する
        import features as Fx
        Fx.GROUPS["_t"] = ["r_high", "volume_trend"]
        Fx.PRESETS["_t"] = ["_t"]
        try:
            res = run(df, ["_t"], folds)
        finally:
            del Fx.GROUPS["_t"], Fx.PRESETS["_t"]

        self.assertTrue(res["folds"])
        self.assertTrue(res["summary"])
        for s in res["summary"]:
            self.assertEqual(s["wins"] + s["losses"] <= s["n_folds"], True)
            self.assertGreaterEqual(s["best_diff"], s["worst_diff"])
        # 基準そのものは要約に出さない（自分との差は常に0で無意味）
        self.assertNotIn(REFERENCE, [s["name"] for s in res["summary"]])

    def test_summarise_counts_wins_correctly(self):
        per_fold = [
            {"results": [{"name": REFERENCE, "pr_auc": 0.3, "diff_vs_ref": 0.0,
                          "lift@5%": 1.0},
                         {"name": "m", "pr_auc": 0.4, "diff_vs_ref": +0.1,
                          "lift@5%": 1.2}]},
            {"results": [{"name": REFERENCE, "pr_auc": 0.3, "diff_vs_ref": 0.0,
                          "lift@5%": 1.0},
                         {"name": "m", "pr_auc": 0.2, "diff_vs_ref": -0.1,
                          "lift@5%": 0.8}]},
        ]
        s = summarize(per_fold)[0]
        self.assertEqual((s["wins"], s["losses"], s["n_folds"]), (1, 1, 2))
        self.assertAlmostEqual(s["mean_diff"], 0.0)
        self.assertAlmostEqual(s["worst_diff"], -0.1)
        self.assertAlmostEqual(s["best_diff"], +0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
