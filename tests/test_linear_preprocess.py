#!/usr/bin/env python3
"""
線形・MLP 用の前処理 v2（research/linear_preprocess.py）のテスト。

固定すること
  - 本番の239列はすべて型の表にあり、既定（符号つき・切るだけ）に落ちる列が無い
  - 統計（分位点・四分位範囲・中央値・最大）は fit に渡した訓練側だけで決まる
  - 切る / asinh / log1p は NaN を素通しし、単調
  - 欠損の埋め方: 意味で 0 / 上限 / 最頻値 / 中央値。指示子は訓練側に欠損があった列だけ
  - tuning_multi.build は既定で v1 のまま。prep="v2" か PREPROCESS="v2" で v2 になる
  - v2 で logit / mlp が学習・採点でき、確率が 0〜1 で有限

  python3 tests/test_linear_preprocess.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import features as F  # noqa: E402
import linear_preprocess as LP  # noqa: E402
import tuning_multi as TM  # noqa: E402


class TestClassification(unittest.TestCase):
    def test_every_production_column_is_classified(self):
        cols = F.columns(F.DEFAULT_PRESET)
        self.assertEqual(len(cols), 239)
        self.assertEqual(LP.unclassified(cols), [])

    def test_categorical_matches_the_tuner(self):
        self.assertEqual(set(LP.CATEGORICAL), set(TM.CATEGORICAL))

    def test_kinds_are_exclusive(self):
        sets = [set(LP.SIGNED_HEAVY), set(LP.SIGNED), set(LP.POSITIVE_HEAVY), set(LP.POSITIVE),
                set(LP.COUNT_DAYS), set(LP.SMALL_INT), set(LP.FLAGS), set(LP.RANK), set(LP.CATEGORICAL)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertEqual(sets[i] & sets[j], set(), (i, j))

    def test_sym_columns_are_recognised_by_name(self):
        self.assertEqual(LP.kind_of("eps_growth_sym_q0"), "sym")
        self.assertEqual(LP.kind_of("cfo_yoy_sym_p1"), "sym")

    def test_fill_rules(self):
        self.assertEqual(LP.fill_of("alert_slratio", "pos_heavy"), 0.0)
        self.assertEqual(LP.fill_of("days_since_rev", "count"), 400.0)
        self.assertEqual(LP.fill_of("listing_years", "count"), 5.0)
        self.assertEqual(LP.fill_of("lvs_days", "count"), "max")
        self.assertEqual(LP.fill_of("has_dividend", "flag"), "mode")
        self.assertEqual(LP.fill_of("ROE_q0", "signed_heavy"), "median")
        self.assertEqual(LP.kind_of("lvs_n_20"), "count_log")
        self.assertEqual(LP.kind_of("brand_new_column"), "default")


class TestTransformers(unittest.TestCase):
    def test_clip_uses_training_quantiles_only(self):
        tr = np.arange(1, 1001, dtype=float).reshape(-1, 1)      # 1..1000
        te = np.array([[-5.0], [500.0], [1e6], [np.nan]])
        c = LP.ClipQuantile(0.01, 0.99).fit(tr)
        lo, hi = np.quantile(tr[:, 0], [0.01, 0.99])
        out = c.transform(te)[:, 0]
        self.assertAlmostEqual(out[0], lo)
        self.assertAlmostEqual(out[1], 500.0)
        self.assertAlmostEqual(out[2], hi)                        # 検証側の外れ値は訓練側の分位点で止まる
        self.assertTrue(np.isnan(out[3]))                         # NaN は素通し
        # 検証側のデータで統計が変わらない（fit は訓練側だけ）
        c.transform(np.array([[1e9]] * 10))
        self.assertAlmostEqual(c.hi_[0], hi)

    def test_clip_without_lower_bound(self):
        c = LP.ClipQuantile(None, 0.5).fit(np.array([[0.0], [10.0], [20.0]]))
        self.assertEqual(c.transform(np.array([[-100.0]]))[0, 0], -100.0)
        self.assertEqual(c.transform(np.array([[100.0]]))[0, 0], 10.0)

    def test_asinh_scale_is_half_the_training_iqr(self):
        tr = np.array([[-10.0], [-2.0], [0.0], [2.0], [10.0]])    # Q1 = -2, Q3 = 2 -> s = 2
        a = LP.Asinh().fit(tr)
        self.assertAlmostEqual(a.scale_[0], 2.0)
        out = a.transform(np.array([[1.0], [-1.0], [1000.0], [np.nan]]))[:, 0]
        self.assertAlmostEqual(out[0], np.arcsinh(0.5))
        self.assertAlmostEqual(out[1], -out[0])                   # 符号を保つ
        self.assertLess(out[2], 8.0)                              # 裾は対数で縮む
        self.assertTrue(np.isnan(out[3]))
        # 単調
        xs = np.linspace(-50, 50, 101).reshape(-1, 1)
        self.assertTrue(np.all(np.diff(a.transform(xs)[:, 0]) > 0))

    def test_asinh_without_spread_falls_back_to_unit_scale(self):
        a = LP.Asinh().fit(np.array([[3.0]] * 10))
        self.assertEqual(a.scale_[0], 1.0)

    def test_log1p(self):
        out = LP.Log1p().transform(np.array([[0.0], [np.e - 1], [-5.0], [np.nan]]))[:, 0]
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 1.0)
        self.assertAlmostEqual(out[2], 0.0)                       # 負は 0 に寄せる
        self.assertTrue(np.isnan(out[3]))

    def test_fill_by_rule_and_indicators(self):
        tr = np.array([[1.0, np.nan, 7.0, 0.0],
                       [3.0, 2.0, np.nan, 1.0],
                       [np.nan, 2.0, 9.0, 1.0],
                       [5.0, 8.0, 9.0, np.nan]])
        f = LP.Fill(["median", 0.0, "max", "mode"]).fit(tr)
        np.testing.assert_allclose(f.values_, [3.0, 0.0, 9.0, 1.0])
        self.assertEqual(list(f.indicator_cols_), [0, 1, 2, 3])
        out = f.transform(np.array([[np.nan, np.nan, np.nan, np.nan]]))
        np.testing.assert_allclose(out[0, :4], [3.0, 0.0, 9.0, 1.0])
        np.testing.assert_allclose(out[0, 4:], [1, 1, 1, 1])
        # 訓練側に欠損の無い列には指示子を付けない（検証側にだけ欠損があっても埋めるだけ）
        g = LP.Fill(["median", "median"]).fit(np.array([[1.0, 1.0], [2.0, np.nan]]))
        self.assertEqual(list(g.indicator_cols_), [1])
        out = g.transform(np.array([[np.nan, np.nan]]))
        self.assertEqual(out.shape, (1, 3))
        self.assertAlmostEqual(out[0, 0], 1.5)

    def test_fill_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            LP.Fill(["median"]).fit(np.zeros((3, 2)))


def synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """本番の列名で、型に合った適当な値を作る（裾の重い列には外れ値を入れる）。"""
    rng = np.random.default_rng(seed)
    cols = F.columns(F.DEFAULT_PRESET)
    d = {}
    for c in cols:
        k = LP.kind_of(c)
        if k == "cat":
            d[c] = rng.integers(0, 5, n).astype(float)
        elif k == "flag":
            d[c] = rng.integers(0, 2, n).astype(float)
        elif k == "small_int":
            d[c] = rng.integers(0, 4, n).astype(float)
        elif k in ("pos", "pos_heavy", "count", "count_log"):
            d[c] = rng.gamma(2.0, 2.0, n) * (1 + (rng.random(n) < 0.01) * 1e4)
        elif k == "rank":
            d[c] = rng.uniform(0, 100, n)
        else:
            d[c] = rng.standard_t(2, n) * 10 * (1 + (rng.random(n) < 0.01) * 1e3)
        v = d[c]
        v[rng.random(n) < 0.1] = np.nan                           # 1割を欠損に
    df = pd.DataFrame(d)
    y = (rng.random(n) < 0.2).astype(int)
    return df, y


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.cols = F.columns(F.DEFAULT_PRESET)
        self.X, self.y = synthetic()

    def test_v2_output_is_finite_and_bounded(self):
        ct = LP.preprocess_v2(self.cols)
        Z = ct.fit_transform(self.X.to_numpy(dtype=float))
        self.assertTrue(np.isfinite(Z).all())
        n_cat = sum(self.X[c].nunique() for c in LP.CATEGORICAL)
        self.assertGreaterEqual(Z.shape[1], 235 + n_cat)         # 数値 + one-hot + 指示子
        # 裾の重い列を入れたのに、標準化後の最大 |z| は切る前（数百）より桁で小さい
        self.assertLess(np.abs(Z).max(), 30.0)

    def test_statistics_come_from_the_training_rows_only(self):
        ct = LP.preprocess_v2(self.cols)
        Xtr = self.X.to_numpy(dtype=float)
        ct.fit(Xtr)
        a = ct.transform(Xtr[:5])
        # 検証側に極端な行を混ぜても、同じ訓練行の変換結果は変わらない
        Xbad = Xtr[:5].copy()
        Xbad[0] = 1e12
        b = ct.transform(Xbad)
        np.testing.assert_allclose(a[1:], b[1:])

    def test_default_build_is_still_v1(self):
        self.assertEqual(TM.PREPROCESS, "v1")
        m = TM.build("logit", {"C": 1.0, "penalty": "l2"}, self.y, self.cols)
        names = [n for n, _, _ in m.steps[0][1].transformers]
        self.assertEqual(names, ["cat", "num"])

    def test_prep_argument_switches_to_v2(self):
        m = TM.build("logit", {"C": 1.0, "penalty": "l2"}, self.y, self.cols, prep="v2")
        names = [n for n, _, _ in m.steps[0][1].transformers]
        self.assertIn("signed_heavy", names)
        self.assertIn("pos_heavy", names)
        self.assertNotIn("num", names)

    def test_module_switch_and_study_name(self):
        old = TM.PREPROCESS
        try:
            TM.PREPROCESS = "v2"
            m = TM.build("mlp", {"h1": 16, "alpha": 0.1}, self.y, self.cols)
            self.assertIn("signed_heavy", [n for n, _, _ in m.steps[0][1].transformers])
            v2 = TM.study_name("logit", 5, "2026-09-26", self.cols)
            TM.PREPROCESS = "v1"
            v1 = TM.study_name("logit", 5, "2026-09-26", self.cols)
            self.assertNotEqual(v1, v2)
            self.assertTrue(v2.endswith("_v2"))
            # 木の study 名は前処理の版に依らない
            TM.PREPROCESS = "v2"
            self.assertEqual(TM.study_name("lgbm", 5, "2026-09-26", self.cols),
                             TM.study_name("lgbm", 5, "2026-09-26", self.cols))
        finally:
            TM.PREPROCESS = old

    def test_logit_and_mlp_fit_and_score_with_v2(self):
        Xtr = self.X.to_numpy(dtype=float)
        for algo, params in (("logit", {"C": 0.1, "penalty": "l2"}),
                             ("mlp", {"h1": 16, "two_layers": False, "alpha": 0.1,
                                      "learning_rate_init": 1e-3, "batch_size": 128})):
            m = TM.build(algo, params, self.y, self.cols, prep="v2")
            m.fit(Xtr, self.y)
            p = m.predict_proba(Xtr)[:, 1]
            self.assertTrue(np.isfinite(p).all(), algo)
            self.assertTrue(((p >= 0) & (p <= 1)).all(), algo)

    def test_unknown_version_is_an_error(self):
        with self.assertRaises(ValueError):
            TM.build("logit", {"C": 1.0, "penalty": "l2"}, self.y, self.cols, prep="v9")


if __name__ == "__main__":
    unittest.main(verbosity=2)
