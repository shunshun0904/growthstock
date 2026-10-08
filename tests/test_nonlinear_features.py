#!/usr/bin/env python3
"""
非線形時系列解析の特徴量（research/nonlinear_features.py、nl_* 32列）のテスト。

性質の分かっている合成系列で各指標の向きを固定し、先読みが無いことを確かめる。

  python3 tests/test_nonlinear_features.py
"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import nonlinear_features as NL  # noqa: E402


def white(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def ar1(n: int, phi: float, seed: int = 1) -> np.ndarray:
    e = np.random.default_rng(seed).normal(0.0, 1.0, n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


class TestKnownSeries(unittest.TestCase):
    """教科書どおりの値になるか（長い系列で確かめる）。"""

    def test_dfa_white_noise_is_half(self):
        a = np.mean([NL.dfa_alpha(white(2000, s), (4, 8, 16, 32, 64, 128)) for s in range(5)])
        self.assertAlmostEqual(a, 0.5, delta=0.08)

    def test_dfa_random_walk_is_one_and_half(self):
        a = NL.dfa_alpha(np.cumsum(white(2000)), (4, 8, 16, 32, 64, 128))
        self.assertGreater(a, 1.2)

    def test_dfa_persistent_series_is_above_half(self):
        self.assertGreater(NL.dfa_alpha(ar1(2000, 0.8), (4, 8, 16, 32, 64, 128)), 0.7)

    def test_variance_ratio(self):
        self.assertAlmostEqual(NL.variance_ratio(white(5000), 5), 1.0, delta=0.1)
        self.assertGreater(NL.variance_ratio(ar1(5000, 0.5), 5), 1.3)
        self.assertLess(NL.variance_ratio(ar1(5000, -0.5), 5), 0.7)

    def test_autocorr(self):
        self.assertAlmostEqual(NL.autocorr(ar1(5000, 0.7), 1), 0.7, delta=0.05)
        self.assertAlmostEqual(NL.autocorr(white(5000), 1), 0.0, delta=0.05)

    def test_permutation_entropy_bounds(self):
        self.assertGreater(NL.permutation_entropy(white(500), m=3), 0.95)
        # 単調列は型が1つなので 0
        self.assertEqual(NL.permutation_entropy(np.arange(100, dtype=float), m=3), 0.0)
        # 正弦波は型が少ない
        s = np.sin(np.linspace(0, 40 * math.pi, 400))
        self.assertLess(NL.permutation_entropy(s, m=3), 0.7)

    def test_sample_entropy_regular_vs_random(self):
        s = np.sin(np.linspace(0, 20 * math.pi, 400))
        self.assertLess(NL.sample_entropy(s), 0.5)
        self.assertGreater(NL.sample_entropy(white(400)), 1.5)
        self.assertTrue(math.isnan(NL.sample_entropy(np.zeros(100))))

    def test_lempel_ziv(self):
        rnd = (white(1000) > 0).astype(int)
        per = np.tile([0, 1], 500)
        self.assertGreater(NL.lempel_ziv(rnd), 0.8)
        self.assertLess(NL.lempel_ziv(per), 0.2)

    def test_spectral_entropy(self):
        self.assertGreater(NL.spectral_entropy(white(512)), 0.85)
        self.assertLess(NL.spectral_entropy(np.sin(np.linspace(0, 32 * math.pi, 512))), 0.3)

    def test_time_reversal_asymmetry_is_zero_for_gaussian(self):
        self.assertAlmostEqual(NL.time_reversal_asymmetry(white(20000)), 0.0, delta=0.03)
        self.assertAlmostEqual(NL.c3(white(20000)), 0.0, delta=0.03)

    def test_time_reversal_asymmetry_detects_irreversible_series(self):
        # 上がるときはゆっくり、下がるときは一気に（のこぎり波）は時間反転で変わる
        saw = np.tile(np.r_[np.linspace(0, 1, 9), [0.0]], 100)
        x = np.diff(saw)
        self.assertGreater(abs(NL.time_reversal_asymmetry(x)), 0.3)

    def test_skew_kurt(self):
        sk, ku = NL.skew_kurt(white(20000))
        self.assertAlmostEqual(sk, 0.0, delta=0.1)
        self.assertAlmostEqual(ku, 0.0, delta=0.2)
        sk2, ku2 = NL.skew_kurt(np.random.default_rng(3).standard_t(3, 20000))
        self.assertGreater(ku2, 2.0)

    def test_bds_is_small_for_iid_and_large_for_arch(self):
        z = [NL.bds_statistic(white(120, s)) for s in range(40)]
        self.assertLess(abs(np.mean(z)), 0.6)
        # GARCH 風: ボラが塊になる系列は i.i.d. から離れる
        rng = np.random.default_rng(9)
        vals = []
        for s in range(20):
            e = rng.normal(size=400)
            sig2 = np.ones(400)
            x = np.zeros(400)
            for t in range(1, 400):
                sig2[t] = 0.05 + 0.3 * x[t - 1] ** 2 + 0.65 * sig2[t - 1]
                x[t] = math.sqrt(sig2[t]) * e[t]
            vals.append(NL.bds_statistic(x[-120:]))
        self.assertGreater(np.mean(vals), np.mean(z) + 1.0)

    def test_rqa_periodic_is_deterministic(self):
        det_s, lam_s, _ = NL.rqa(np.sin(np.linspace(0, 12 * math.pi, 120)))
        det_w, lam_w, _ = NL.rqa(white(120))
        self.assertGreater(det_s, det_w)
        self.assertGreater(det_s, 0.9)
        for v in (det_s, lam_s, det_w, lam_w):
            self.assertTrue(0.0 <= v <= 1.0)

    def test_higuchi(self):
        self.assertAlmostEqual(NL.higuchi_fd(np.linspace(0, 1, 200)), 1.0, delta=0.05)
        self.assertGreater(NL.higuchi_fd(white(200)), 1.7)
        self.assertLess(NL.higuchi_fd(np.cumsum(white(200))), 1.7)

    def test_efficiency_ratio_and_crossings(self):
        self.assertAlmostEqual(NL.efficiency_ratio(np.linspace(1, 2, 61)), 1.0)
        zig = 100 + np.tile([0.0, 1.0], 60)[:61]
        self.assertLess(NL.efficiency_ratio(zig), 0.05)
        self.assertGreater(NL.ma_crossings(100 + np.cumsum(white(300)) * 0.01 + np.tile([0, 1.0], 150), 20, 120),
                           NL.ma_crossings(np.linspace(100, 200, 300), 20, 120))

    def test_burstiness(self):
        self.assertAlmostEqual(NL.burstiness(np.ones(60)), -1.0)
        spiky = np.r_[np.ones(59), [1000.0]]
        self.assertGreater(NL.burstiness(spiky), 0.5)

    def test_mutual_information_and_spearman(self):
        a = white(600)
        self.assertLess(NL.mutual_information(a, white(600, 5)), 0.05)
        self.assertGreater(NL.mutual_information(a, a + 0.1 * white(600, 6)), 0.3)
        self.assertAlmostEqual(NL.spearman(a, -a), -1.0, delta=1e-9)
        self.assertAlmostEqual(NL.spearman(a, white(600, 7)), 0.0, delta=0.1)


class TestWindowFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 300)))
        self.value = np.exp(rng.normal(20.0, 0.5, 300))

    def test_all_columns_present_and_finite(self):
        f = NL.features_for_window(self.close, self.value)
        self.assertEqual(list(f), NL.NL_COLS)
        for c in NL.NL_COLS:
            self.assertTrue(np.isfinite(f[c]), c)

    def test_only_the_last_n_bars_matter(self):
        f_all = NL.features_for_window(self.close, self.value)
        f_tail = NL.features_for_window(self.close[-NL.N_BARS:], self.value[-NL.N_BARS:])
        for c in NL.NL_COLS:
            self.assertAlmostEqual(f_all[c], f_tail[c], places=12, msg=c)

    def test_short_history_is_missing_not_zero(self):
        f = NL.features_for_window(self.close[-30:], self.value[-30:])
        self.assertTrue(all(math.isnan(f[c]) for c in NL.NL_COLS))

    def test_pinned_price_is_detected(self):
        close = np.r_[self.close[:-60], np.full(60, self.close[-61])]
        f = NL.features_for_window(close, self.value)
        self.assertEqual(f["nl_zero_ret_60"], 1.0)
        self.assertTrue(math.isnan(f["nl_skew_60"]))     # σ=0 は定義しない
        self.assertTrue(math.isnan(f["nl_acf1_60"]))

    def test_missing_bars_are_dropped(self):
        close = self.close.copy()
        close[-10] = np.nan
        f = NL.features_for_window(close, self.value)
        self.assertTrue(np.isfinite(f["nl_dfa_r120"]))
        self.assertTrue(np.isfinite(f["nl_zero_ret_60"]))

    def test_mostly_missing_window_is_missing(self):
        close = self.close.copy()
        close[-120:-10] = np.nan
        f = NL.features_for_window(close, self.value)
        self.assertTrue(math.isnan(f["nl_pe3_60"]))
        self.assertTrue(math.isnan(f["nl_dfa_r120"]))


def synthetic_bars(n_days: int = 320, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for code in ("10000", "20000"):
        px = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n_days)))
        vo = rng.integers(1000, 50000, n_days).astype(float)
        rows.append(pd.DataFrame({"Date": dates, "Code": code, "C": px, "AdjC": px,
                                  "Vo": vo, "Va": px * vo}))
    return pd.concat(rows, ignore_index=True)


class TestAttach(unittest.TestCase):
    def test_no_lookahead(self):
        bars = synthetic_bars()
        d = bars["Date"].unique()
        keys = pd.DataFrame({"Code": ["10000", "20000"], "Date": [d[300], d[280]]})
        base = NL.attach_nonlinear(keys, bars)
        # 未来の足を足しても（値を変えても）同じ日の列は変わらない
        future = bars.copy()
        fut = future["Date"] > pd.Timestamp(d[280])
        future.loc[fut, ["C", "AdjC"]] *= 3.0
        future.loc[fut, "Va"] *= 10.0
        again = NL.attach_nonlinear(keys[keys["Code"] == "20000"], future)
        a = base[base["Code"] == "20000"].iloc[0]
        b = again.iloc[0]
        for c in NL.NL_COLS:
            self.assertAlmostEqual(a[c], b[c], places=12, msg=c)

    def test_unknown_code_or_date_is_missing(self):
        bars = synthetic_bars()
        keys = pd.DataFrame({"Code": ["99999", "10000"],
                             "Date": [bars["Date"].iloc[300], pd.Timestamp("2030-01-01")]})
        out = NL.attach_nonlinear(keys, bars)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[NL.NL_COLS].isna().all().all())

    def test_parallel_matches_serial(self):
        bars = synthetic_bars()
        d = bars["Date"].unique()
        keys = pd.DataFrame({"Code": ["10000", "20000", "10000"],
                             "Date": [d[300], d[280], d[310]]})
        a = NL.attach_nonlinear(keys, bars, workers=1).sort_values(["Code", "Date"]).reset_index(drop=True)
        b = NL.attach_nonlinear(keys, bars, workers=2).sort_values(["Code", "Date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_column_list_and_descriptions_agree(self):
        self.assertEqual(set(NL.NL_DESC), set(NL.NL_COLS))
        self.assertEqual(len(NL.NL_COLS), 32)
        self.assertTrue(all(c.startswith("nl_") for c in NL.NL_COLS))


if __name__ == "__main__":
    unittest.main()
